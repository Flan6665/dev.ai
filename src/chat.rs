//! Формат диалога и разбор обучающих примеров.
//!
//! Диалог в токенах:
//! `<|system|>…<|end|><|user|>…<|end|><|assistant|><|think|>размышления<|/think|>ответ<|end|>`

use crate::tokenizer::{self as tk, Tokenizer};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            thinking: None,
        }
    }
}

/// Если размышления лежат прямо в тексте как `<think>…</think>`, вынимаем их.
fn split_think(content: &str) -> (Option<String>, String) {
    let trimmed = content.trim_start();
    if let Some(rest) = trimmed.strip_prefix("<think>")
        && let Some(end) = rest.find("</think>")
    {
        let thinking = rest[..end].trim().to_string();
        let answer = rest[end + "</think>".len()..].trim_start().to_string();
        return (Some(thinking), answer);
    }
    (None, content.to_string())
}

fn str_field<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| v.get(*k).and_then(Value::as_str))
}

/// Разбирает одну строку JSONL. Понимает несколько популярных форматов:
/// - `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "thinking": "...", "content": "..."}]}`
/// - `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` (ShareGPT)
/// - `{"instruction": "...", "input": "...", "output": "..."}` (Alpaca)
/// - `{"prompt": "...", "response"|"completion": "...", "thinking"?: "..."}`
pub fn parse_example(line: &str) -> Result<Vec<Message>> {
    let v: Value = serde_json::from_str(line)?;
    let mut msgs = Vec::new();
    if let Some(arr) = v
        .get("messages")
        .or_else(|| v.get("conversations"))
        .and_then(Value::as_array)
    {
        for m in arr {
            let role = str_field(m, &["role", "from"]).unwrap_or("user");
            let role = match role {
                "human" | "user" => "user",
                "gpt" | "assistant" | "model" | "bot" => "assistant",
                "system" => "system",
                other => bail!("неизвестная роль {other}"),
            };
            let content = str_field(m, &["content", "value"])
                .unwrap_or("")
                .to_string();
            let thinking =
                str_field(m, &["thinking", "reasoning", "reasoning_content"]).map(str::to_string);
            msgs.push(Message {
                role: role.into(),
                content,
                thinking,
            });
        }
    } else if let Some(instr) = str_field(&v, &["instruction", "prompt", "question", "problem"]) {
        let mut user = instr.to_string();
        if let Some(inp) = str_field(&v, &["input"]).filter(|s| !s.is_empty()) {
            user = format!("{user}\n\n{inp}");
        }
        let Some(out) = str_field(
            &v,
            &["output", "response", "completion", "answer", "solution"],
        ) else {
            bail!("нет поля с ответом (output/response/completion/answer)");
        };
        msgs.push(Message::new("user", &user));
        let thinking = str_field(&v, &["thinking", "reasoning"]).map(str::to_string);
        msgs.push(Message {
            role: "assistant".into(),
            content: out.into(),
            thinking,
        });
    } else {
        bail!("непонятный формат примера");
    }
    for m in msgs
        .iter_mut()
        .filter(|m| m.role == "assistant" && m.thinking.is_none())
    {
        let (thinking, content) = split_think(&m.content);
        m.thinking = thinking;
        m.content = content;
    }
    if !msgs.iter().any(|m| m.role == "assistant") {
        bail!("в примере нет ответа ассистента");
    }
    Ok(msgs)
}

/// Режим размышлений при генерации.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Think {
    /// Модель решает сама.
    Auto,
    /// Обязательно начать с размышлений.
    On,
    /// Сразу отвечать.
    Off,
}

/// Кодирует диалог. Возвращает токены и маску: 1.0 — токен, который модель учится
/// предсказывать (только ответы ассистента), 0.0 — контекст.
pub fn encode_chat(
    tok: &Tokenizer,
    msgs: &[Message],
    generation_prompt: Option<Think>,
) -> (Vec<u32>, Vec<f32>) {
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    fn push(new: &[u32], m: f32, ids: &mut Vec<u32>, mask: &mut Vec<f32>) {
        ids.extend_from_slice(new);
        mask.extend(std::iter::repeat_n(m, new.len()));
    }
    for msg in msgs {
        match msg.role.as_str() {
            "assistant" => {
                push(&[tok.special(tk::ASSISTANT)], 0.0, &mut ids, &mut mask);
                push(&[tok.special(tk::THINK)], 1.0, &mut ids, &mut mask);
                if let Some(t) = &msg.thinking {
                    push(&tok.encode(t), 1.0, &mut ids, &mut mask);
                }
                push(&[tok.special(tk::END_THINK)], 1.0, &mut ids, &mut mask);
                push(&tok.encode(&msg.content), 1.0, &mut ids, &mut mask);
                push(&[tok.special(tk::END)], 1.0, &mut ids, &mut mask);
            }
            role => {
                let head = if role == "system" {
                    tk::SYSTEM
                } else {
                    tk::USER
                };
                push(&[tok.special(head)], 0.0, &mut ids, &mut mask);
                push(&tok.encode(&msg.content), 0.0, &mut ids, &mut mask);
                push(&[tok.special(tk::END)], 0.0, &mut ids, &mut mask);
            }
        }
    }
    if let Some(think) = generation_prompt {
        ids.push(tok.special(tk::ASSISTANT));
        match think {
            Think::Auto => {}
            Think::On => ids.push(tok.special(tk::THINK)),
            Think::Off => ids.extend([tok.special(tk::THINK), tok.special(tk::END_THINK)]),
        }
        mask.resize(ids.len(), 0.0);
    }
    (ids, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_formats() {
        let a = parse_example(
            r#"{"instruction":"Сложи","input":"2 и 2","output":"<think>2+2=4</think>4"}"#,
        )
        .unwrap();
        assert_eq!(a[0].content, "Сложи\n\n2 и 2");
        assert_eq!(a[1].thinking.as_deref(), Some("2+2=4"));
        assert_eq!(a[1].content, "4");

        let b = parse_example(
            r#"{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"yo"}]}"#,
        )
        .unwrap();
        assert_eq!(b[1].role, "assistant");
        assert!(parse_example(r#"{"messages":[{"role":"user","content":"x"}]}"#).is_err());
    }

    #[test]
    fn mask_covers_only_assistant() {
        let tok = Tokenizer::new(vec![]);
        let msgs = vec![Message::new("user", "ab"), Message::new("assistant", "cd")];
        let (ids, mask) = encode_chat(&tok, &msgs, None);
        assert_eq!(ids.len(), mask.len());
        // <|user|> a b <|end|> | <|assistant|> | <|think|> <|/think|> c d <|end|>
        assert_eq!(mask, vec![0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]);
        let (ids2, _) = encode_chat(&tok, &msgs[..1], Some(Think::Off));
        assert_eq!(ids2, ids[..7]);
    }
}
