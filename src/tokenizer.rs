//! Байтовый BPE-токенизатор (как у GPT-2/Llama 3), который обучается на твоём корпусе.
//!
//! Словарь: 0..256 — сырые байты, затем специальные токены, затем слияния BPE.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;

pub const ENDOFTEXT: &str = "<|endoftext|>";
pub const SYSTEM: &str = "<|system|>";
pub const USER: &str = "<|user|>";
pub const ASSISTANT: &str = "<|assistant|>";
pub const THINK: &str = "<|think|>";
pub const END_THINK: &str = "<|/think|>";
pub const END: &str = "<|end|>";

pub const SPECIAL_TOKENS: &[&str] = &[ENDOFTEXT, SYSTEM, USER, ASSISTANT, THINK, END_THINK, END];

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharClass {
    Word,
    Number,
    Space,
    Other,
}

fn class(c: char) -> CharClass {
    if c.is_whitespace() {
        CharClass::Space
    } else if c.is_ascii_digit() {
        CharClass::Number
    } else if c.is_alphanumeric() || c == '_' {
        CharClass::Word
    } else {
        CharClass::Other
    }
}

fn continues(cls: CharClass, c: char) -> bool {
    match cls {
        CharClass::Word => c.is_alphanumeric() || c == '_',
        CharClass::Number => c.is_ascii_digit(),
        CharClass::Other => class(c) == CharClass::Other,
        CharClass::Space => c.is_whitespace(),
    }
}

/// Режет текст на куски: идентификаторы, числа, знаки, пробелы.
/// Один пробел перед словом приклеивается к слову (" foo"), как в GPT-2.
/// Склейка всех кусков всегда равна исходному тексту.
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let start = chars[i].0;
        let mut j = i;
        if chars[j].1 == ' ' && j + 1 < n && !chars[j + 1].1.is_whitespace() {
            j += 1;
        }
        let cls = class(chars[j].1);
        if cls == CharClass::Space {
            let mut k = j;
            while k < n && chars[k].1.is_whitespace() {
                k += 1;
            }
            // Последний пробел отдаём следующему слову.
            if k < n && k - j > 1 && chars[k - 1].1 == ' ' {
                k -= 1;
            }
            j = k;
        } else {
            j += 1;
            while j < n && continues(cls, chars[j].1) {
                j += 1;
            }
        }
        let end = if j < n { chars[j].0 } else { text.len() };
        out.push(&text[start..end]);
        i = j;
    }
    out
}

fn merge_pair(ids: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

#[derive(Serialize, Deserialize)]
struct TokenizerFile {
    special_tokens: Vec<String>,
    merges: Vec<(u32, u32)>,
}

#[derive(Clone)]
pub struct Tokenizer {
    special_tokens: Vec<String>,
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    vocab: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn new(merges: Vec<(u32, u32)>) -> Self {
        let special_tokens: Vec<String> = SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect();
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for s in &special_tokens {
            vocab.push(s.as_bytes().to_vec());
        }
        let first_merge = vocab.len() as u32;
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, &(a, b)) in merges.iter().enumerate() {
            let mut bytes = vocab[a as usize].clone();
            bytes.extend_from_slice(&vocab[b as usize]);
            vocab.push(bytes);
            ranks.insert((a, b), first_merge + i as u32);
        }
        Self {
            special_tokens,
            merges,
            ranks,
            vocab,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn special(&self, name: &str) -> u32 {
        let idx = self
            .special_tokens
            .iter()
            .position(|s| s == name)
            .expect("неизвестный спецтокен");
        256 + idx as u32
    }

    pub fn is_special(&self, id: u32) -> bool {
        id >= 256 && (id as usize) < 256 + self.special_tokens.len()
    }

    /// Обучение BPE: на каждом шаге склеиваем самую частую пару соседних токенов.
    /// Счётчики пар обновляются инкрементально, лучшая пара берётся из кучи.
    pub fn train(chunk_counts: HashMap<String, u64>, vocab_size: usize, verbose: bool) -> Self {
        let base = 256 + SPECIAL_TOKENS.len();
        assert!(vocab_size > base, "vocab_size должен быть больше {base}");
        let mut words: Vec<Vec<u32>> = Vec::with_capacity(chunk_counts.len());
        let mut freqs: Vec<i64> = Vec::with_capacity(chunk_counts.len());
        for (chunk, count) in chunk_counts {
            if chunk.len() < 2 {
                continue;
            }
            words.push(chunk.bytes().map(u32::from).collect());
            freqs.push(count as i64);
        }

        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), HashSet<u32>> = HashMap::new();
        for (wi, w) in words.iter().enumerate() {
            for p in w.windows(2) {
                let p = (p[0], p[1]);
                *pair_counts.entry(p).or_default() += freqs[wi];
                where_.entry(p).or_default().insert(wi as u32);
            }
        }
        let mut heap: BinaryHeap<(i64, Reverse<(u32, u32)>)> =
            pair_counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

        let n_merges = vocab_size - base;
        let mut merges = Vec::with_capacity(n_merges);
        while merges.len() < n_merges {
            let Some((count, Reverse(pair))) = heap.pop() else {
                break;
            };
            // Запись в куче могла устареть — берём только актуальные.
            if pair_counts.get(&pair).copied() != Some(count) {
                continue;
            }
            if count < 2 {
                break;
            }
            let new_id = (base + merges.len()) as u32;
            merges.push(pair);

            let mut touched: HashSet<(u32, u32)> = HashSet::new();
            let word_ids = where_.remove(&pair).unwrap_or_default();
            for wi in word_ids {
                let wi_us = wi as usize;
                let f = freqs[wi_us];
                let old = std::mem::take(&mut words[wi_us]);
                for p in old.windows(2) {
                    let p = (p[0], p[1]);
                    if let Some(c) = pair_counts.get_mut(&p) {
                        *c -= f;
                        if *c <= 0 {
                            pair_counts.remove(&p);
                        }
                    }
                    if let Some(set) = where_.get_mut(&p) {
                        set.remove(&wi);
                    }
                    touched.insert(p);
                }
                let new = merge_pair(&old, pair, new_id);
                for p in new.windows(2) {
                    let p = (p[0], p[1]);
                    *pair_counts.entry(p).or_default() += f;
                    where_.entry(p).or_default().insert(wi);
                    touched.insert(p);
                }
                words[wi_us] = new;
            }
            for p in touched {
                if let Some(&c) = pair_counts.get(&p) {
                    heap.push((c, Reverse(p)));
                }
            }
            if verbose && merges.len() % 1000 == 0 {
                println!("  BPE: {}/{} слияний", merges.len(), n_merges);
            }
        }
        Self::new(merges)
    }

    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = chunk.bytes().map(u32::from).collect();
        while ids.len() >= 2 {
            let mut best: Option<(u32, (u32, u32))> = None;
            for p in ids.windows(2) {
                if let Some(&rank) = self.ranks.get(&(p[0], p[1]))
                    && best.is_none_or(|(r, _)| rank < r)
                {
                    best = Some((rank, (p[0], p[1])));
                }
            }
            match best {
                Some((rank, pair)) => ids = merge_pair(&ids, pair, rank),
                None => break,
            }
        }
        ids
    }

    /// Кодирует обычный текст. Спецтокены внутри текста НЕ распознаются
    /// (их вставляет только код), так что пользователь не сломает формат чата.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3);
        for chunk in pre_tokenize(text) {
            out.extend(self.encode_chunk(chunk));
        }
        out
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.vocab[id as usize]
    }

    #[allow(dead_code)]
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.vocab[i as usize].iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let f = TokenizerFile {
            special_tokens: self.special_tokens.clone(),
            merges: self.merges.clone(),
        };
        std::fs::write(path, serde_json::to_string(&f)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("не могу прочитать {}", path.display()))?;
        let f: TokenizerFile = serde_json::from_str(&s)?;
        if f.special_tokens != SPECIAL_TOKENS {
            bail!(
                "спецтокены в {} не совпадают с текущей версией",
                path.display()
            );
        }
        Ok(Self::new(f.merges))
    }
}

/// Кодировщик с кэшем кусков — для быстрой токенизации больших корпусов.
pub struct CachedEncoder<'a> {
    tok: &'a Tokenizer,
    cache: HashMap<String, Vec<u32>>,
}

impl<'a> CachedEncoder<'a> {
    pub fn new(tok: &'a Tokenizer) -> Self {
        Self {
            tok,
            cache: HashMap::new(),
        }
    }

    pub fn encode_into(&mut self, text: &str, out: &mut Vec<u32>) {
        for chunk in pre_tokenize(text) {
            if let Some(ids) = self.cache.get(chunk) {
                out.extend_from_slice(ids);
                continue;
            }
            let ids = self.tok.encode_chunk(chunk);
            out.extend_from_slice(&ids);
            if self.cache.len() < 1_000_000 {
                self.cache.insert(chunk.to_string(), ids);
            }
        }
    }
}

/// Потоковый декодер: отдаёт текст, только когда байты складываются в целые UTF-8 символы.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // Битая последовательность (не просто незаконченная) — заменяем.
                if e.error_len().is_some() {
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return s;
                }
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_tokenize_roundtrip() {
        let text = "fn main() {\n    let x = 42; // привет\n\tprintln!(\"{x}\");\n}\n\n  ";
        assert_eq!(pre_tokenize(text).concat(), text);
        assert!(pre_tokenize("let foo").contains(&" foo"));
    }

    #[test]
    fn train_encode_decode() {
        let text =
            "def add(a, b):\n    return a + b\n".repeat(50) + "fn add(a: i32) -> i32 { a }\n";
        let mut counts = HashMap::new();
        for c in pre_tokenize(&text) {
            *counts.entry(c.to_string()).or_insert(0) += 1;
        }
        let tok = Tokenizer::train(counts, 300, false);
        assert!(tok.vocab_size() > 263);
        let sample = "def add(x, y):\n    return x + y  # сумма 🙂";
        let ids = tok.encode(sample);
        assert!(ids.len() < sample.len());
        assert_eq!(tok.decode(&ids), sample);

        let path = std::env::temp_dir().join("ishka_tok_test.json");
        tok.save(&path).unwrap();
        let tok2 = Tokenizer::load(&path).unwrap();
        assert_eq!(tok2.encode(sample), ids);
    }

    #[test]
    fn stream_decoder_handles_split_utf8() {
        let mut d = StreamDecoder::default();
        let bytes = "ж".as_bytes();
        assert_eq!(d.push(&bytes[..1]), "");
        assert_eq!(d.push(&bytes[1..]), "ж");
    }
}
