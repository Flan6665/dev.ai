//! Веб-чат: общаешься с моделью в браузере, исправляешь её ответы и дообучаешь прямо оттуда.

use crate::chat::{self, Message, Think};
use crate::checkpoint;
use crate::data::ChatData;
use crate::generate::{SamplingParams, generate};
use crate::model::{Config, Model};
use crate::tokenizer::{self as tk, StreamDecoder, Tokenizer};
use crate::train::{adamw, lr_at, train_step};
use crate::util::{Rng, human_count, pick_device};
use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use burn::module::Module;
use burn::prelude::Device;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

struct Engine {
    /// Модель с градиентами — её дообучаем.
    model: Model,
    /// Её копия без градиентов — ей отвечаем (быстрее и меньше памяти).
    infer: Model,
    tok: Tokenizer,
    cfg: Config,
}

#[derive(Serialize, Clone, Default)]
struct TrainStatus {
    running: bool,
    step: usize,
    total: usize,
    loss: Option<f32>,
    message: String,
}

struct AppState {
    engine: Mutex<Engine>,
    model_dir: PathBuf,
    feedback: PathBuf,
    status: Mutex<TrainStatus>,
    device: Device,
    train_device: Device,
    gpu: bool,
}

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    #[arg(long, default_value = "models/ishka")]
    pub model: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// Куда сохранять примеры, которым ты учишь модель (по умолчанию <model>/feedback.jsonl).
    #[arg(long)]
    pub feedback: Option<PathBuf>,
    /// Работать на процессоре, даже если есть видеокарта.
    #[arg(long)]
    pub cpu: bool,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    let (device, gpu) = pick_device(args.cpu);
    let train_device = device.clone().autodiff();
    let l = checkpoint::load(&args.model, &train_device)?;
    let feedback = args
        .feedback
        .clone()
        .unwrap_or_else(|| args.model.join("feedback.jsonl"));
    let state = Arc::new(AppState {
        engine: Mutex::new(Engine {
            infer: l.model.valid(),
            model: l.model,
            tok: l.tokenizer,
            cfg: l.cfg,
        }),
        model_dir: args.model.clone(),
        feedback,
        status: Mutex::new(TrainStatus::default()),
        device,
        train_device,
        gpu,
    });
    let app = Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("../static/index.html")) }),
        )
        .route("/api/info", get(info))
        .route("/api/chat", post(chat_handler))
        .route("/api/teach", post(teach))
        .route("/api/train", post(train_handler))
        .route("/api/status", get(status))
        .with_state(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("Ишка запущена: открой http://{addr} в браузере");
    axum::serve(listener, app).await?;
    Ok(())
}

fn count_examples(path: &PathBuf) -> usize {
    std::fs::read_to_string(path)
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

async fn info(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    let train_state = checkpoint::load_state(&st.model_dir);
    let cfg = match st.engine.try_lock() {
        Ok(e) => Some(e.cfg.clone()),
        Err(_) => serde_json::from_str::<Config>(
            &std::fs::read_to_string(st.model_dir.join(checkpoint::CONFIG)).unwrap_or_default(),
        )
        .ok(),
    };
    Json(json!({
        "params": cfg.as_ref().map(|c| human_count(c.n_params())),
        "config": cfg,
        "device": if st.gpu { "GPU" } else { "CPU" },
        "pretrain_steps": train_state.step,
        "tokens_seen": human_count(train_state.tokens_seen as usize),
        "examples": count_examples(&st.feedback),
    }))
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    #[serde(default)]
    think: Option<String>,
    #[serde(default)]
    params: SamplingParams,
}

async fn chat_handler(
    State(st): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let (tx, rx) = tokio::sync::mpsc::channel::<serde_json::Value>(256);
    tokio::task::spawn_blocking(move || {
        let Ok(engine) = st.engine.try_lock() else {
            let _ = tx.blocking_send(json!({"type": "error", "text": "Сейчас идёт обучение или другой ответ — подожди немного."}));
            return;
        };
        let think = match req.think.as_deref() {
            Some("on") => Think::On,
            Some("off") => Think::Off,
            _ => Think::Auto,
        };
        let tok = &engine.tok;
        let (prompt, _) = chat::encode_chat(tok, &req.messages, Some(think));
        let (think_id, end_think_id) = (tok.special(tk::THINK), tok.special(tk::END_THINK));
        let stop = [tok.special(tk::END), tok.special(tk::ENDOFTEXT)];
        let banned = [
            tok.special(tk::SYSTEM),
            tok.special(tk::USER),
            tok.special(tk::ASSISTANT),
        ];
        let mut phase = if think == Think::On {
            "think"
        } else {
            "answer"
        };
        let mut decoder = StreamDecoder::default();
        let mut rng = Rng::from_time();
        let res = generate(
            &engine.infer,
            &st.device,
            &prompt,
            &req.params,
            &mut rng,
            &stop,
            &banned,
            Some((think_id, end_think_id)),
            |id| {
                let msg = if id == think_id {
                    phase = "think";
                    json!({"type": "phase", "phase": "think"})
                } else if id == end_think_id {
                    phase = "answer";
                    json!({"type": "phase", "phase": "answer"})
                } else if tok.is_special(id) {
                    return true;
                } else {
                    let text = decoder.push(tok.token_bytes(id));
                    if text.is_empty() {
                        return true;
                    }
                    json!({"type": phase, "text": text})
                };
                tx.blocking_send(msg).is_ok()
            },
        );
        let end = match res {
            Ok(ids) => json!({"type": "done", "tokens": ids.len()}),
            Err(e) => json!({"type": "error", "text": e.to_string()}),
        };
        let _ = tx.blocking_send(end);
    });
    let stream =
        ReceiverStream::new(rx).map(|v| Ok::<_, Infallible>(Event::default().data(v.to_string())));
    Sse::new(stream)
}

#[derive(Deserialize)]
struct TeachRequest {
    messages: Vec<Message>,
}

async fn teach(
    State(st): State<Arc<AppState>>,
    Json(req): Json<TeachRequest>,
) -> impl IntoResponse {
    let ok = req
        .messages
        .last()
        .is_some_and(|m| m.role == "assistant" && !m.content.trim().is_empty())
        && req.messages.iter().any(|m| m.role == "user");
    if !ok {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "нужен вопрос пользователя и непустой ответ"})),
        );
    }
    let line = json!({"messages": req.messages}).to_string();
    let res = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&st.feedback)
        .and_then(|mut f| writeln!(f, "{line}"));
    match res {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"examples": count_examples(&st.feedback)})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        ),
    }
}

#[derive(Deserialize)]
struct TrainRequest {
    #[serde(default = "default_steps")]
    steps: usize,
    #[serde(default = "default_lr")]
    lr: f64,
}

fn default_steps() -> usize {
    60
}

fn default_lr() -> f64 {
    1e-4
}

async fn train_handler(
    State(st): State<Arc<AppState>>,
    Json(req): Json<TrainRequest>,
) -> impl IntoResponse {
    {
        let mut s = st.status.lock().unwrap();
        if s.running {
            return (
                StatusCode::CONFLICT,
                Json(json!({"error": "обучение уже идёт"})),
            );
        }
        *s = TrainStatus {
            running: true,
            total: req.steps,
            message: "Готовлю примеры…".into(),
            ..Default::default()
        };
    }
    let st2 = st.clone();
    std::thread::spawn(move || {
        let result = teach_model(&st2, req.steps.clamp(1, 5000), req.lr);
        let mut s = st2.status.lock().unwrap();
        s.running = false;
        s.message = match result {
            Ok(()) => "Готово! Модель дообучена и сохранена.".into(),
            Err(e) => format!("Ошибка: {e}"),
        };
    });
    (StatusCode::OK, Json(json!({"started": true})))
}

fn teach_model(st: &AppState, steps: usize, lr: f64) -> Result<()> {
    let mut engine = st.engine.lock().unwrap();
    let mut data = ChatData::load_jsonl(&engine.tok, &st.feedback)?;
    let n = data.len();
    // Резервная копия весов до обучения — чтобы можно было откатиться.
    let weights = st.model_dir.join(checkpoint::WEIGHTS);
    if weights.exists() {
        std::fs::copy(
            &weights,
            st.model_dir.join("model.before-teach.safetensors"),
        )?;
    }
    let mut opt = adamw(0.0, 1.0);
    let mut rng = Rng::from_time();
    let micro_batch = n.clamp(1, 4);
    let seq_len = engine.cfg.max_seq_len;
    st.status.lock().unwrap().message = format!("Учусь на {n} примерах…");
    let mut model = engine.model.clone();
    for step in 0..steps {
        let cur_lr = lr_at(step, (steps / 10).max(1), steps, lr, lr * 0.1);
        let (m, loss) = train_step(
            model,
            &mut opt,
            &mut data,
            &mut rng,
            &st.train_device,
            micro_batch,
            seq_len,
            1,
            cur_lr,
        )?;
        model = m;
        let mut s = st.status.lock().unwrap();
        s.step = step + 1;
        s.loss = Some(loss);
    }
    st.status.lock().unwrap().message = "Сохраняю…".into();
    checkpoint::save_weights(&st.model_dir, &model)?;
    engine.infer = model.valid();
    engine.model = model;
    Ok(())
}

async fn status(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    Json(st.status.lock().unwrap().clone())
}
