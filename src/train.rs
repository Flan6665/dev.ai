//! Обучение: оптимизатор AdamW, шаг с накоплением градиентов, цикл с оценкой и сохранением.

use crate::checkpoint::{self, TrainState};
use crate::data::{BatchSource, ChatData, TokenFile};
use crate::model::{Model, lm_loss};
use crate::util::{Rng, human_count, parse_dtype, pick_device};
use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct AdamW {
    names: Vec<String>,
    vars: Vec<Var>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    pub t: usize,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

impl AdamW {
    pub fn new(varmap: &VarMap, weight_decay: f64) -> Result<Self> {
        let mut named: Vec<(String, Var)> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        let (names, vars): (Vec<_>, Vec<_>) = named.into_iter().unzip();
        let m = vars
            .iter()
            .map(|v| v.as_tensor().zeros_like())
            .collect::<candle_core::Result<Vec<_>>>()?;
        let v = vars
            .iter()
            .map(|v| v.as_tensor().zeros_like())
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            names,
            vars,
            m,
            v,
            t: 0,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
        })
    }

    pub fn step(&mut self, grads: &[Option<Tensor>], lr: f64) -> Result<()> {
        self.t += 1;
        let c1 = 1.0 - self.beta1.powi(self.t as i32);
        let c2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..self.vars.len() {
            let Some(g) = &grads[i] else { continue };
            let p = self.vars[i].as_tensor();
            let m = ((&self.m[i] * self.beta1)? + (g * (1.0 - self.beta1))?)?;
            let v = ((&self.v[i] * self.beta2)? + (g.sqr()? * (1.0 - self.beta2))?)?;
            let update = ((&m / c1)? / ((&v / c2)?.sqrt()? + self.eps)?)?;
            // Weight decay — только матрицам (не нормализациям).
            let decay = if p.rank() >= 2 {
                1.0 - lr * self.weight_decay
            } else {
                1.0
            };
            let new_p = ((p * decay)? - (update * lr)?)?;
            self.vars[i].set(&new_p)?;
            self.m[i] = m;
            self.v[i] = v;
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut map = HashMap::new();
        for (i, name) in self.names.iter().enumerate() {
            map.insert(format!("m.{name}"), self.m[i].clone());
            map.insert(format!("v.{name}"), self.v[i].clone());
        }
        map.insert("t".into(), Tensor::new(&[self.t as u32], &Device::Cpu)?);
        checkpoint::save_tensors(&map, path)
    }

    pub fn load(&mut self, path: &Path, device: &Device) -> Result<()> {
        let map = candle_core::safetensors::load(path, device)?;
        for (i, name) in self.names.iter().enumerate() {
            if let (Some(m), Some(v)) =
                (map.get(&format!("m.{name}")), map.get(&format!("v.{name}")))
            {
                self.m[i] = m.to_dtype(self.vars[i].dtype())?;
                self.v[i] = v.to_dtype(self.vars[i].dtype())?;
            }
        }
        if let Some(t) = map.get("t") {
            self.t = t.to_vec1::<u32>()?[0] as usize;
        }
        Ok(())
    }
}

/// Один шаг оптимизации: `accum` микро-батчей, градиенты суммируются,
/// затем обрезка по норме и шаг AdamW. Возвращает (loss, норма градиента).
#[allow(clippy::too_many_arguments)]
pub fn train_step(
    model: &Model,
    opt: &mut AdamW,
    data: &mut dyn BatchSource,
    rng: &mut Rng,
    micro_batch: usize,
    seq_len: usize,
    accum: usize,
    lr: f64,
    clip: f64,
) -> Result<(f32, f32)> {
    let mut acc: Vec<Option<Tensor>> = vec![None; opt.vars.len()];
    let mut total_loss = 0.0f32;
    for _ in 0..accum {
        let batch = data.next_batch(rng, micro_batch, seq_len, model.device())?;
        let logits = model.forward(&batch.x, None, true)?;
        let loss = lm_loss(&logits, &batch.y, batch.mask.as_ref())?;
        total_loss += loss.to_scalar::<f32>()?;
        let grads = (loss / accum as f64)?.backward()?;
        for (i, var) in opt.vars.iter().enumerate() {
            if let Some(g) = grads.get(var.as_tensor()) {
                acc[i] = Some(match acc[i].take() {
                    Some(a) => (a + g)?,
                    None => g.clone(),
                });
            }
        }
    }
    let mut sq = Tensor::zeros((), DType::F32, model.device())?;
    for g in acc.iter().flatten() {
        sq = (sq + g.to_dtype(DType::F32)?.sqr()?.sum_all()?)?;
    }
    let norm = sq.sqrt()?.to_scalar::<f32>()?;
    if !norm.is_finite() {
        bail!("градиент стал NaN/inf — уменьши learning rate");
    }
    if norm as f64 > clip {
        let s = clip / (norm as f64 + 1e-6);
        for g in acc.iter_mut().flatten() {
            *g = (&*g * s)?;
        }
    }
    opt.step(&acc, lr)?;
    Ok((total_loss / accum as f32, norm))
}

pub fn eval_loss(
    model: &Model,
    data: &mut dyn BatchSource,
    rng: &mut Rng,
    batch: usize,
    seq_len: usize,
    iters: usize,
) -> Result<f32> {
    let mut total = 0.0;
    for _ in 0..iters {
        let b = data.next_batch(rng, batch, seq_len, model.device())?;
        let logits = model.forward(&b.x, None, false)?;
        total += lm_loss(&logits, &b.y, b.mask.as_ref())?.to_scalar::<f32>()?;
    }
    Ok(total / iters as f32)
}

/// Косинусное расписание с прогревом.
pub fn lr_at(step: usize, warmup: usize, total: usize, max_lr: f64, min_lr: f64) -> f64 {
    if step < warmup {
        return max_lr * (step + 1) as f64 / warmup as f64;
    }
    let progress = ((step - warmup) as f64 / (total.saturating_sub(warmup)).max(1) as f64).min(1.0);
    min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
}

#[derive(clap::Args, Debug)]
pub struct TrainArgs {
    /// Папка модели (создаётся командой `init`).
    #[arg(long, default_value = "models/ishka")]
    pub model: PathBuf,
    /// Предобучение: папка от `prepare` (train.bin / val.bin).
    #[arg(long)]
    pub data: Option<PathBuf>,
    /// Дообучение на диалогах: файл .jsonl (можно несколько через запятую).
    #[arg(long, value_delimiter = ',')]
    pub sft: Vec<PathBuf>,
    /// Сколько шагов оптимизации сделать в этом запуске.
    #[arg(long, default_value_t = 10000)]
    pub steps: usize,
    #[arg(long, default_value_t = 3e-4)]
    pub lr: f64,
    #[arg(long, default_value_t = 0.1)]
    pub min_lr_ratio: f64,
    #[arg(long, default_value_t = 500)]
    pub warmup: usize,
    /// Примеров в микро-батче (уменьши, если не хватает видеопамяти).
    #[arg(long, default_value_t = 8)]
    pub micro_batch: usize,
    /// Накопление градиентов: эффективный батч = micro_batch * accum.
    #[arg(long, default_value_t = 4)]
    pub accum: usize,
    /// Длина контекста при обучении (по умолчанию — из config).
    #[arg(long)]
    pub seq_len: Option<usize>,
    #[arg(long, default_value_t = 0.1)]
    pub weight_decay: f64,
    #[arg(long, default_value_t = 1.0)]
    pub clip: f64,
    #[arg(long, default_value_t = 250)]
    pub eval_every: usize,
    #[arg(long, default_value_t = 1000)]
    pub save_every: usize,
    /// f32 (точнее) или bf16 (быстрее и меньше памяти, RTX 30xx и новее).
    #[arg(long, default_value = "f32")]
    pub dtype: String,
    /// Начать оптимизатор и счётчик шагов заново (например, при переходе к SFT).
    #[arg(long)]
    pub fresh: bool,
    #[arg(long)]
    pub cpu: bool,
    #[arg(long, default_value_t = 1337)]
    pub seed: u64,
}

pub fn run(args: TrainArgs) -> Result<()> {
    let device = pick_device(args.cpu)?;
    let dtype = parse_dtype(&args.dtype, &device)?;
    let loaded = checkpoint::load(&args.model, &device, dtype)?;
    let (model, varmap, tok, cfg) = (loaded.model, loaded.varmap, loaded.tokenizer, loaded.cfg);
    let seq_len = args.seq_len.unwrap_or(cfg.max_seq_len).min(cfg.max_seq_len);

    let (mut train_data, mut val_data): (Box<dyn BatchSource>, Option<Box<dyn BatchSource>>) =
        match (&args.data, args.sft.is_empty()) {
            (Some(dir), true) => {
                let train = TokenFile::open(&dir.join("train.bin"))?;
                println!(
                    "Предобучение: {} токенов в корпусе",
                    human_count(train.len())
                );
                let val = TokenFile::open(&dir.join("val.bin"))
                    .ok()
                    .filter(|v| v.len() > seq_len + 1);
                (
                    Box::new(train),
                    val.map(|v| Box::new(v) as Box<dyn BatchSource>),
                )
            }
            (None, false) => {
                let mut convs = Vec::new();
                for p in &args.sft {
                    let d = ChatData::load_jsonl(&tok, p)?;
                    println!("Дообучение: {} диалогов из {}", d.len(), p.display());
                    convs.push(d);
                }
                let data = convs.into_iter().reduce(|mut a, b| {
                    a.append(b);
                    a
                });
                (Box::new(data.unwrap()), None)
            }
            _ => bail!("укажи либо --data (предобучение), либо --sft (диалоги)"),
        };

    let mut opt = AdamW::new(&varmap, args.weight_decay)?;
    let mut state = if args.fresh {
        TrainState::default()
    } else {
        checkpoint::load_state(&args.model)
    };
    let opt_path = args.model.join(checkpoint::OPTIMIZER);
    if !args.fresh && opt_path.exists() {
        opt.load(&opt_path, &device)?;
        println!(
            "Продолжаю с шага {} (видела {} токенов)",
            state.step,
            human_count(state.tokens_seen as usize)
        );
    }

    println!(
        "Модель: {} параметров | {:?} | контекст {seq_len} | батч {}×{} | lr {}",
        human_count(cfg.n_params()),
        dtype,
        args.micro_batch,
        args.accum,
        args.lr
    );

    let mut rng = Rng::new(args.seed ^ state.step as u64);
    let start_step = state.step;
    let end_step = start_step + args.steps;
    let tokens_per_step = (args.micro_batch * args.accum * seq_len) as u64;
    let mut t0 = Instant::now();
    let mut log_loss = 0.0;
    let mut log_n = 0;
    let save = |state: &TrainState, opt: &AdamW| -> Result<()> {
        checkpoint::save_weights(&args.model, &varmap)?;
        opt.save(&opt_path)?;
        checkpoint::save_state(&args.model, state)?;
        Ok(())
    };

    for step in start_step..end_step {
        let lr = lr_at(
            step - start_step,
            args.warmup,
            args.steps,
            args.lr,
            args.lr * args.min_lr_ratio,
        );
        let (loss, norm) = train_step(
            &model,
            &mut opt,
            train_data.as_mut(),
            &mut rng,
            args.micro_batch,
            seq_len,
            args.accum,
            lr,
            args.clip,
        )?;
        state.step = step + 1;
        state.tokens_seen += tokens_per_step;
        log_loss += loss;
        log_n += 1;

        if state.step % 10 == 0 || state.step == end_step {
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "шаг {:>7} | loss {:.4} | grad {:.2} | lr {:.2e} | {:.0} ток/с",
                state.step,
                log_loss / log_n as f32,
                norm,
                lr,
                (tokens_per_step * log_n as u64) as f64 / dt
            );
            log_loss = 0.0;
            log_n = 0;
            t0 = Instant::now();
        }
        if let Some(val) = val_data.as_mut()
            && state.step % args.eval_every == 0
        {
            let vl = eval_loss(
                &model,
                val.as_mut(),
                &mut Rng::new(0),
                args.micro_batch,
                seq_len,
                20,
            )?;
            println!(
                "  ▸ проверка: val loss {vl:.4} (perplexity {:.1})",
                vl.exp()
            );
            state.best_val_loss = Some(state.best_val_loss.map_or(vl, |b| b.min(vl)));
            t0 = Instant::now();
        }
        if state.step % args.save_every == 0 {
            save(&state, &opt)?;
            println!("  ✓ сохранено в {}", args.model.display());
            t0 = Instant::now();
        }
    }
    save(&state, &opt)?;
    println!("Готово. Модель сохранена в {}", args.model.display());
    Ok(())
}
