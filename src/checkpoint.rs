//! Папка модели: config.json, tokenizer.json, model.safetensors (+ optimizer.bin, state.json).

use crate::model::{Config, Model};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow, bail};
use burn::prelude::Device;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const CONFIG: &str = "config.json";
pub const TOKENIZER: &str = "tokenizer.json";
pub const WEIGHTS: &str = "model.safetensors";
pub const OPTIMIZER: &str = "optimizer.bin";
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
    pub tokenizer: Tokenizer,
    pub cfg: Config,
}

pub fn load_config(dir: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(dir.join(CONFIG))
        .with_context(|| format!("нет {}/{CONFIG} — сначала `ishka init`", dir.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// Загружает модель на устройство. Для обучения передай `device.autodiff()`.
pub fn load(dir: &Path, device: &Device) -> Result<Loaded> {
    let cfg = load_config(dir)?;
    let tokenizer = Tokenizer::load(&dir.join(TOKENIZER))?;
    if tokenizer.vocab_size() != cfg.vocab_size {
        bail!(
            "размер словаря токенизатора {} не совпадает с config {}",
            tokenizer.vocab_size(),
            cfg.vocab_size
        );
    }
    let mut model = Model::new(&cfg, device);
    let weights = dir.join(WEIGHTS);
    if weights.exists() {
        let mut store = SafetensorsStore::from_file(&weights);
        model
            .load_from(&mut store)
            .map_err(|e| anyhow!("не удалось загрузить {}: {e:?}", weights.display()))?;
    } else {
        eprintln!("⚠ {} нет — модель со случайными весами.", weights.display());
    }
    Ok(Loaded {
        model,
        tokenizer,
        cfg,
    })
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".tmp");
    PathBuf::from(p)
}

/// Сохраняем через временный файл, чтобы не испортить веса, если процесс прервётся.
pub fn save_weights(dir: &Path, model: &Model) -> Result<()> {
    let path = dir.join(WEIGHTS);
    let tmp = tmp_path(&path);
    let mut store = SafetensorsStore::from_file(&tmp);
    model
        .save_into(&mut store)
        .map_err(|e| anyhow!("не удалось сохранить веса: {e:?}"))?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

pub fn save_optimizer(dir: &Path, opt: &burn::optim::ModuleOptimizer) -> Result<()> {
    let path = dir.join(OPTIMIZER);
    let tmp = tmp_path(&path);
    opt.save(&tmp)
        .map_err(|e| anyhow!("не удалось сохранить оптимизатор: {e:?}"))?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
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
    let model = Model::new(cfg, &Device::flex());
    save_weights(dir, &model)?;
    save_state(dir, &TrainState::default())?;
    Ok(())
}
