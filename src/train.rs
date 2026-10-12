//! Обучение: AdamW, накопление градиентов, расписание lr, оценка и сохранение.

use crate::checkpoint::{self, TrainState};
use crate::data::{BatchSource, ChatData, TokenFile};
use crate::model::{Model, lm_loss};
use crate::util::{Rng, human_count, pick_device};
use anyhow::{Result, anyhow, bail};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::prelude::Device;
use std::path::PathBuf;
use std::time::Instant;

pub fn adamw(weight_decay: f64, clip: f64) -> ModuleOptimizer {
    AdamWConfig::new()
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .with_weight_decay(weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(clip as f32)))
        .init()
}

/// Один шаг оптимизации: `accum` микро-батчей, градиенты суммируются,
/// затем обрезка по норме и шаг AdamW. `device` — устройство с autodiff.
#[allow(clippy::too_many_arguments)]
pub fn train_step(
    model: Model,
    opt: &mut ModuleOptimizer,
    data: &mut dyn BatchSource,
    rng: &mut Rng,
    device: &Device,
    micro_batch: usize,
    seq_len: usize,
    accum: usize,
    lr: f64,
) -> Result<(Model, f32)> {
    let mut acc = GradientsAccumulator::new();
    let mut total_loss = 0.0f32;
    for _ in 0..accum {
        let batch = data.next_batch(rng, micro_batch, seq_len, device)?;
        let logits = model.forward(batch.x, None)?;
        let loss = lm_loss(logits, batch.y, batch.mask);
        let value = loss.clone().into_scalar::<f32>();
        if !value.is_finite() {
            bail!("loss стал NaN/inf — уменьши learning rate");
        }
        total_loss += value;
        let grads = (loss / accum as f64).backward();
        acc.accumulate(&model, GradientsParams::from_grads(grads, &model));
    }
    let model = opt.step(lr, model, acc.grads());
    Ok((model, total_loss / accum as f32))
}

/// Средний loss на проверочных данных. `model` — копия без autodiff (`valid()`).
pub fn eval_loss(
    model: &Model,
    data: &mut dyn BatchSource,
    rng: &mut Rng,
    device: &Device,
    batch: usize,
    seq_len: usize,
    iters: usize,
) -> Result<f32> {
    let mut total = 0.0;
    for _ in 0..iters {
        let b = data.next_batch(rng, batch, seq_len, device)?;
        let logits = model.forward(b.x, None)?;
        total += lm_loss(logits, b.y, b.mask).into_scalar::<f32>();
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
    /// Начать оптимизатор и счётчик шагов заново (например, при переходе к SFT).
    #[arg(long)]
    pub fresh: bool,
    /// Обучать на процессоре, даже если есть видеокарта.
    #[arg(long)]
    pub cpu: bool,
    #[arg(long, default_value_t = 1337)]
    pub seed: u64,
}

pub fn run(args: TrainArgs) -> Result<()> {
    let (device, _) = pick_device(args.cpu);
    let train_device = device.clone().autodiff();
    let loaded = checkpoint::load(&args.model, &train_device)?;
    let (mut model, tok, cfg) = (loaded.model, loaded.tokenizer, loaded.cfg);
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

    let mut opt = adamw(args.weight_decay, args.clip);
    let mut state = if args.fresh {
        TrainState::default()
    } else {
        checkpoint::load_state(&args.model)
    };
    let opt_path = args.model.join(checkpoint::OPTIMIZER);
    if !args.fresh && opt_path.exists() {
        opt = opt
            .load(&opt_path)
            .map_err(|e| anyhow!("не удалось загрузить оптимизатор: {e:?}"))?;
        println!(
            "Продолжаю с шага {} (видела {} токенов)",
            state.step,
            human_count(state.tokens_seen as usize)
        );
    }

    println!(
        "Модель: {} параметров | контекст {seq_len} | батч {}×{} | lr {}",
        human_count(cfg.n_params()),
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
    let save = |state: &TrainState, model: &Model, opt: &ModuleOptimizer| -> Result<()> {
        checkpoint::save_weights(&args.model, model)?;
        checkpoint::save_optimizer(&args.model, opt)?;
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
        let (m, loss) = train_step(
            model,
            &mut opt,
            train_data.as_mut(),
            &mut rng,
            &train_device,
            args.micro_batch,
            seq_len,
            args.accum,
            lr,
        )?;
        model = m;
        state.step = step + 1;
        state.tokens_seen += tokens_per_step;
        log_loss += loss;
        log_n += 1;

        if state.step % 10 == 0 || state.step == end_step {
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "шаг {:>7} | loss {:.4} | lr {:.2e} | {:.0} ток/с",
                state.step,
                log_loss / log_n as f32,
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
                &model.valid(),
                val.as_mut(),
                &mut Rng::new(0),
                &device,
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
            save(&state, &model, &opt)?;
            println!("  ✓ сохранено в {}", args.model.display());
            t0 = Instant::now();
        }
    }
    save(&state, &model, &opt)?;
    println!("Готово. Модель сохранена в {}", args.model.display());
    Ok(())
}
