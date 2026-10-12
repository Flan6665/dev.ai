//! Генерация: семплирование токенов с температурой, top-k, top-p и штрафом за повторы.

use crate::model::{KvCache, Model};
use crate::util::Rng;
use anyhow::Result;
use burn::prelude::{Device, Int, Tensor, TensorData};
use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            repeat_penalty: 1.1,
            max_tokens: 512,
        }
    }
}

fn sample(
    logits: &mut [f32],
    history: &[u32],
    banned: &[u32],
    p: &SamplingParams,
    rng: &mut Rng,
) -> u32 {
    for &b in banned {
        logits[b as usize] = f32::NEG_INFINITY;
    }
    if p.repeat_penalty != 1.0 {
        let recent = &history[history.len().saturating_sub(64)..];
        for &t in recent {
            let l = &mut logits[t as usize];
            *l = if *l > 0.0 {
                *l / p.repeat_penalty
            } else {
                *l * p.repeat_penalty
            };
        }
    }
    let argmax = || {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    };
    if p.temperature <= 0.0 {
        return argmax();
    }
    let mut idx: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if p.top_k > 0 {
        idx.truncate(p.top_k);
    }
    let max = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - max) / p.temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|x| *x /= sum);
    let mut cum = 0.0;
    let mut cut = probs.len();
    for (k, &pr) in probs.iter().enumerate() {
        cum += pr;
        if cum >= p.top_p {
            cut = k + 1;
            break;
        }
    }
    let total: f32 = probs[..cut].iter().sum();
    let mut r = rng.f32() * total;
    for k in 0..cut {
        r -= probs[k];
        if r <= 0.0 {
            return idx[k] as u32;
        }
    }
    idx[cut - 1] as u32
}

fn last_logits(logits: Tensor<3>) -> Vec<f32> {
    let [_, t, v] = logits.dims();
    logits
        .narrow(1, t - 1, 1)
        .reshape([v])
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("логиты должны быть f32")
}

fn ids(v: &[u32], device: &Device) -> Tensor<2, Int> {
    let v: Vec<i64> = v.iter().map(|&x| i64::from(x)).collect();
    let n = v.len();
    Tensor::from_data(TensorData::new(v, [1, n]), device)
}

/// Генерирует продолжение. `on_token` получает каждый новый токен и может
/// вернуть false, чтобы остановиться. Генерация прекращается на токене из `stop`.
/// `think` — пара (открыть, закрыть) размышления: открыть можно один раз, закрыть — только после открытия.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &Model,
    device: &Device,
    prompt: &[u32],
    p: &SamplingParams,
    rng: &mut Rng,
    stop: &[u32],
    banned: &[u32],
    think: Option<(u32, u32)>,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<Vec<u32>> {
    let max_ctx = model.max_seq_len();
    // Если промпт не влезает — оставляем конец (самое свежее).
    let keep = max_ctx.saturating_sub(p.max_tokens.min(max_ctx / 2)).max(1);
    let prompt = if prompt.len() > keep {
        &prompt[prompt.len() - keep..]
    } else {
        prompt
    };
    let mut cache = KvCache::new(model.n_layers());
    let mut logits = last_logits(model.forward(ids(prompt, device), Some(&mut cache))?);
    let mut history = prompt.to_vec();
    let mut out = Vec::new();
    // 0 — размышления не начаты, 1 — идут, 2 — закончены.
    let mut think_state = match (think, prompt.last()) {
        (Some((_, close)), Some(&last)) if last == close => 2,
        (Some((open, _)), Some(&last)) if last == open => 1,
        _ => 0,
    };
    let mut ban = banned.to_vec();
    for _ in 0..p.max_tokens {
        ban.truncate(banned.len());
        if let Some((open, close)) = think {
            match think_state {
                0 => ban.push(close),
                1 => ban.push(open),
                _ => ban.extend([open, close]),
            }
        }
        let next = sample(&mut logits, &history, &ban, p, rng);
        if let Some((open, close)) = think {
            if next == open {
                think_state = 1;
            } else if next == close {
                think_state = 2;
            }
        }
        if stop.contains(&next) {
            break;
        }
        out.push(next);
        history.push(next);
        if !on_token(next) || cache.len() + 1 > max_ctx {
            break;
        }
        logits = last_logits(model.forward(ids(&[next], device), Some(&mut cache))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_banned() {
        let mut rng = Rng::new(1);
        let p = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let mut l = vec![0.0, 5.0, 1.0];
        assert_eq!(sample(&mut l, &[], &[], &p, &mut rng), 1);
        let mut l = vec![0.0, 5.0, 1.0];
        assert_eq!(sample(&mut l, &[], &[1], &p, &mut rng), 2);
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        let mut l = vec![0.0, 5.0, 1.0];
        assert_eq!(sample(&mut l, &[], &[], &p, &mut rng), 1);
    }
}
