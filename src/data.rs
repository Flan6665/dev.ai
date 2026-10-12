//! Данные: сбор файлов, токенизированный корпус для предобучения и диалоги для дообучения.

use crate::chat::{self, Message};
use crate::tokenizer::Tokenizer;
use crate::util::Rng;
use anyhow::{Context, Result, bail};
use burn::prelude::{Device, Int, Tensor, TensorData};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const DEFAULT_EXTS: &str = "rs,py,js,ts,tsx,jsx,go,c,h,cpp,hpp,cc,java,kt,cs,swift,rb,php,lua,sh,sql,html,css,scss,vue,svelte,toml,yaml,yml,json,md,txt,jsonl";

/// Все подходящие файлы из списка путей (файлы и папки), в стабильном порядке.
pub fn collect_files(inputs: &[PathBuf], exts: &str) -> Vec<PathBuf> {
    let exts: Vec<&str> = exts
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .collect();
    let mut files = Vec::new();
    for input in inputs {
        for entry in walkdir::WalkDir::new(input)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let p = entry.path();
            let skip_dir = p.components().any(|c| {
                let s = c.as_os_str().to_string_lossy();
                s == ".git"
                    || s == "node_modules"
                    || s == "target"
                    || s == "__pycache__"
                    || s == ".venv"
            });
            let ext_ok = p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| exts.contains(&e));
            if !skip_dir && ext_ok {
                files.push(p.to_path_buf());
            }
        }
    }
    files.sort();
    files
}

/// Документы из файла. Для .jsonl — по одному на строку (поле text/content/code),
/// иначе весь файл целиком. Бинарные и огромные файлы пропускаются.
pub fn read_documents(path: &Path, max_file_bytes: u64) -> Vec<String> {
    let Ok(meta) = std::fs::metadata(path) else {
        return vec![];
    };
    let is_jsonl = path.extension().is_some_and(|e| e == "jsonl");
    if !is_jsonl && meta.len() > max_file_bytes {
        return vec![];
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return vec![];
    };
    if !is_jsonl {
        return if text.trim().is_empty() {
            vec![]
        } else {
            vec![text]
        };
    }
    text.lines()
        .filter_map(|line| {
            let v: Value = serde_json::from_str(line).ok()?;
            ["text", "content", "code"]
                .iter()
                .find_map(|k| v.get(*k).and_then(Value::as_str))
                .map(str::to_string)
        })
        .filter(|s| !s.trim().is_empty())
        .collect()
}

pub struct Batch {
    pub x: Tensor<2, Int>,
    pub y: Tensor<2, Int>,
    pub mask: Option<Tensor<2>>,
}

fn ids_tensor(ids: Vec<u32>, batch: usize, t: usize, dev: &Device) -> Tensor<2, Int> {
    let ids: Vec<i64> = ids.into_iter().map(i64::from).collect();
    Tensor::from_data(TensorData::new(ids, [batch, t]), dev)
}

pub trait BatchSource {
    fn next_batch(
        &mut self,
        rng: &mut Rng,
        batch: usize,
        seq_len: usize,
        dev: &Device,
    ) -> Result<Batch>;
}

/// Токенизированный корпус (u16, little-endian), отображённый в память —
/// можно обучаться на десятках гигабайт без загрузки в RAM.
pub struct TokenFile {
    mmap: memmap2::Mmap,
}

impl TokenFile {
    pub fn open(path: &Path) -> Result<Self> {
        let f =
            std::fs::File::open(path).with_context(|| format!("нет файла {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        Ok(Self { mmap })
    }

    pub fn len(&self) -> usize {
        self.mmap.len() / 2
    }

    fn get(&self, i: usize) -> u32 {
        u16::from_le_bytes([self.mmap[2 * i], self.mmap[2 * i + 1]]) as u32
    }
}

impl BatchSource for TokenFile {
    fn next_batch(
        &mut self,
        rng: &mut Rng,
        batch: usize,
        seq_len: usize,
        dev: &Device,
    ) -> Result<Batch> {
        if self.len() <= seq_len + 1 {
            bail!(
                "корпус слишком маленький ({} токенов) для seq_len {seq_len}",
                self.len()
            );
        }
        let mut xs = Vec::with_capacity(batch * seq_len);
        let mut ys = Vec::with_capacity(batch * seq_len);
        for _ in 0..batch {
            let start = rng.below(self.len() - seq_len - 1);
            xs.extend((start..start + seq_len).map(|i| self.get(i)));
            ys.extend((start + 1..start + seq_len + 1).map(|i| self.get(i)));
        }
        Ok(Batch {
            x: ids_tensor(xs, batch, seq_len, dev),
            y: ids_tensor(ys, batch, seq_len, dev),
            mask: None,
        })
    }
}

/// Диалоги для дообучения (SFT): учим только ответы ассистента.
pub struct ChatData {
    examples: Vec<(Vec<u32>, Vec<f32>)>,
    pad: u32,
}

impl ChatData {
    pub fn from_conversations(tok: &Tokenizer, convs: &[Vec<Message>]) -> Self {
        let examples = convs
            .iter()
            .map(|c| chat::encode_chat(tok, c, None))
            .filter(|(ids, mask)| ids.len() >= 2 && mask.iter().any(|&m| m > 0.0))
            .collect();
        Self {
            examples,
            pad: tok.special(crate::tokenizer::ENDOFTEXT),
        }
    }

    pub fn load_jsonl(tok: &Tokenizer, path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("нет файла {}", path.display()))?;
        let mut convs = Vec::new();
        let mut bad = 0;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            match chat::parse_example(line) {
                Ok(c) => convs.push(c),
                Err(_) => bad += 1,
            }
        }
        if bad > 0 {
            eprintln!(
                "⚠ пропущено {bad} строк в непонятном формате в {}",
                path.display()
            );
        }
        let data = Self::from_conversations(tok, &convs);
        if data.examples.is_empty() {
            bail!("в {} нет ни одного годного примера", path.display());
        }
        Ok(data)
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn append(&mut self, other: ChatData) {
        self.examples.extend(other.examples);
    }
}

impl BatchSource for ChatData {
    fn next_batch(
        &mut self,
        rng: &mut Rng,
        batch: usize,
        seq_len: usize,
        dev: &Device,
    ) -> Result<Batch> {
        let picked: Vec<usize> = (0..batch).map(|_| rng.below(self.examples.len())).collect();
        let longest = picked
            .iter()
            .map(|&i| self.examples[i].0.len() - 1)
            .max()
            .unwrap_or(1);
        let t = longest.clamp(1, seq_len);
        let (mut xs, mut ys, mut ms) = (Vec::new(), Vec::new(), Vec::new());
        for i in picked {
            let (ids, mask) = &self.examples[i];
            let n = (ids.len() - 1).min(t);
            xs.extend_from_slice(&ids[..n]);
            ys.extend_from_slice(&ids[1..n + 1]);
            ms.extend_from_slice(&mask[1..n + 1]);
            for _ in n..t {
                xs.push(self.pad);
                ys.push(self.pad);
                ms.push(0.0);
            }
        }
        Ok(Batch {
            x: ids_tensor(xs, batch, t, dev),
            y: ids_tensor(ys, batch, t, dev),
            mask: Some(Tensor::from_data(TensorData::new(ms, [batch, t]), dev)),
        })
    }
}
