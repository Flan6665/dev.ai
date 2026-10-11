//! Папка модели: config.json, tokenizer.json, model.safetensors (+ optimizer.safetensors, state.json).

use crate::model::{Config, Model};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub const CONFIG: &str = "config.json";
pub const TOKENIZER: &str = "tokenizer.json";
pub const WEIGHTS: &str = "model.safetensors";
pub const OPTIMIZER: &str = "optimizer.safetensors";
pub const STATE: &str = "state.json";

/// Сколько модель уже обучалась (для продолжения с того же места).
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct TrainState {
    pub step: usize,
    pub tokens_seen: u64,
    pub best_val_loss: Option<f32>,
}

pub struct Loaded {
    pub model: Model,
    pub varmap: VarMap,
    pub tokenizer: Tokenizer,
    pub cfg: Config,
}

pub fn load(dir: &Path, device: &Device, dtype: DType) -> Result<Loaded> {
    let cfg: Config = serde_json::from_str(
        &std::fs::read_to_string(dir.join(CONFIG))
            .with_context(|| format!("нет {}/{CONFIG} — сначала `ishka init`", dir.display()))?,
    )?;
    let tokenizer = Tokenizer::load(&dir.join(TOKENIZER))?;
    if tokenizer.vocab_size() != cfg.vocab_size {
        bail!(
            "размер словаря токенизатора {} не совпадает с config {}",
            tokenizer.vocab_size(),
            cfg.vocab_size
        );
    }
    let varmap = VarMap::new();
    let model = Model::new(&cfg, VarBuilder::from_varmap(&varmap, dtype, device))?;
    let weights = dir.join(WEIGHTS);
    if weights.exists() {
        load_into(&varmap, &weights, device)?;
    } else {
        eprintln!("⚠ {} нет — модель со случайными весами.", weights.display());
    }
    Ok(Loaded {
        model,
        varmap,
        tokenizer,
        cfg,
    })
}

fn load_into(varmap: &VarMap, path: &Path, device: &Device) -> Result<()> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        let t = tensors
            .get(name)
            .with_context(|| format!("в {} нет тензора {name}", path.display()))?;
        var.set(&t.to_dtype(var.dtype())?)?;
    }
    Ok(())
}

/// Сохраняем через временный файл, чтобы не испортить веса, если процесс прервётся.
pub fn save_tensors(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    let tmp = path.with_extension("safetensors.tmp");
    candle_core::safetensors::save(tensors, &tmp)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn save_weights(dir: &Path, varmap: &VarMap) -> Result<()> {
    let tensors: HashMap<String, Tensor> = varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
        .collect();
    save_tensors(&tensors, &dir.join(WEIGHTS))
}

pub fn load_state(dir: &Path) -> TrainState {
    std::fs::read_to_string(dir.join(STATE))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_state(dir: &Path, st: &TrainState) -> Result<()> {
    std::fs::write(dir.join(STATE), serde_json::to_string_pretty(st)?)?;
    Ok(())
}

/// Создаёт новую модель со случайными весами.
pub fn init(dir: &Path, cfg: &Config, tokenizer: &Tokenizer) -> Result<()> {
    if dir.join(WEIGHTS).exists() {
        bail!(
            "в {} уже есть модель — не перезаписываю. Удали папку или выбери другую.",
            dir.display()
        );
    }
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(CONFIG), serde_json::to_string_pretty(cfg)?)?;
    tokenizer.save(&dir.join(TOKENIZER))?;
    let varmap = VarMap::new();
    Model::new(
        cfg,
        VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
    )?;
    save_weights(dir, &varmap)?;
    save_state(dir, &TrainState::default())?;
    Ok(())
}
