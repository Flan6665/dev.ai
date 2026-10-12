use burn::prelude::Device;

/// Маленький быстрый ГПСЧ (SplitMix64) — для выборки батчей и семплирования.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn from_time() -> Self {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        Self::new(t)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Число в [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Число в [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Видеокарта через wgpu (DirectX 12 / Vulkan / Metal — нужен только обычный драйвер),
/// а если её нет — процессор. Возвращает устройство и признак «это видеокарта».
pub fn pick_device(cpu: bool) -> (Device, bool) {
    if !cpu {
        match Device::wgpu_options().init() {
            Ok(dev) => {
                eprintln!("✓ Работаю на видеокарте (wgpu).");
                return (dev, true);
            }
            Err(e) => {
                eprintln!("⚠ Видеокарта недоступна ({e:?}). Работаю на процессоре — это медленно.")
            }
        }
    }
    // Burn кое-где (например, при загрузке состояния оптимизатора) берёт устройство «по умолчанию»,
    // а оно — видеокарта. Без видеокарты указываем процессор явно.
    // SAFETY: вызывается при старте, до того как другие потоки читают окружение.
    unsafe { std::env::set_var("BURN_DEVICE", "flex") };
    (Device::flex(), false)
}

pub fn human_count(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}
