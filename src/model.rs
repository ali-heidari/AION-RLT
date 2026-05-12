use crate::get_config as CONFIG;
use anyhow::Ok;
use log::warn;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::distributions::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::u8;

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    pub snapshot: String,
    id: String,
}

impl Model {
    pub fn new(id: &str) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-0.1, 0.1);
        let this = Self {
            w1: Array2::random_using(
                (CONFIG().input_number, CONFIG().hidden_layers),
                uniform,
                &mut rng,
            ),
            b1: Array1::zeros(CONFIG().hidden_layers),
            w2: Array2::random_using(
                (CONFIG().hidden_layers, CONFIG().output_number),
                uniform,
                &mut rng,
            ),
            b2: Array1::zeros(CONFIG().output_number),
            snapshot: String::new(),
            id: id.to_owned(),
        };

        OpenOptions::new()
            .write(true)
            .create(true) // Creates if not exists
            .open(this.model_path())
            .ok();

        this
    }

    fn relu(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.max(0.0))
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = Self::relu(&(x.dot(&self.w1) + &self.b1));
        h.dot(&self.w2) + &self.b2
    }

    pub fn softmax2(x: &Array2<f32>, temperature: f32) -> Array2<f32> {
        let max_per_row = x.map_axis(Axis(1), |r| r.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let exps = x - &max_per_row.insert_axis(Axis(1));
        let exps = exps.mapv(|v| (v / temperature).exp());
        let sum_per_row = exps.sum_axis(Axis(1)).insert_axis(Axis(1));
        &exps / &sum_per_row
    }

    pub fn softmax(logits: &Array2<f32>, temperature: f32) -> Vec<f32> {
        // temperature > 0.0
        let inv_temp = 1.0 / temperature;
        let mut max = f32::NEG_INFINITY;
        for &v in logits {
            if v > max {
                max = v
            }
        }
        // subtract max for numerical stability
        let mut exps: Vec<f32> = logits
            .iter()
            .map(|&v| ((v - max) * inv_temp).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            // fallback to uniform
            let n = logits.len() as f32;
            return vec![1.0 / n; logits.len()];
        }
        exps.iter_mut().for_each(|x| *x /= sum);
        exps
    }

    pub fn reinforce(
        &mut self,
        x: &Array2<f32>,
        actions: &Array1<usize>,
        rewards: &Array1<f32>,
        lr: f32,
    ) -> f32 {
        let h_pre = x.dot(&self.w1) + &self.b1;
        let h = Self::relu(&h_pre);
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = Self::softmax2(&logits, 1.0);

        let mut grad_logits = Array2::<f32>::zeros(probs.raw_dim());
        for (i, &a) in actions.iter().enumerate() {
            grad_logits[[i, a]] = -rewards[i] * (1.0 - probs[[i, a]]);
        }

        let grad_w2 = h.t().dot(&grad_logits);
        let grad_b2 = grad_logits.sum_axis(Axis(0));
        let grad_h = grad_logits.dot(&self.w2.t());
        let grad_h_relu = grad_h * &h_pre.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let grad_w1 = x.t().dot(&grad_h_relu);
        let grad_b1 = grad_h_relu.sum_axis(Axis(0));

        self.w1 -= &(lr * grad_w1);
        self.b1 -= &(lr * grad_b1);
        self.w2 -= &(lr * grad_w2);
        self.b2 -= &(lr * grad_b2);

        if self.w1.iter().any(|x| x.is_nan()) {
            warn!("NaN detected in weights!");
        }
        rewards.mean().unwrap_or(0.0)
    }

    fn model_path(&self) -> String {
        let dir = "./models";
        if let Err(err) = std::fs::create_dir_all(dir) {
            warn!("Failed to create model directory '{}': {}", dir, err);
        }
        format!("{}/{}.{}", dir, self.id, CONFIG().model_name)
    }

    pub fn save_model(&self) -> anyhow::Result<()> {
        let json = serde_json::to_string(self)?;
        let mut file = File::create(self.model_path())?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load_model(&self) -> anyhow::Result<Model> {
        let data = std::fs::read_to_string(self.model_path())?.replace("null", "0.0");
        let model = match serde_json::from_str(&data) {
            std::result::Result::Ok(m) => m,
            Err(_) => Model::new(&self.id),
        };
        Ok(model)
    }

    pub fn sanitize(&mut self) {
        for x in self.w1.iter_mut() {
            if !x.is_finite() || x.abs() > 1e6 {
                *x = 0.0;
            }
        }
        for x in self.b1.iter_mut() {
            if !x.is_finite() || x.abs() > 1e6 {
                *x = 0.0;
            }
        }
        for x in self.w2.iter_mut() {
            if !x.is_finite() || x.abs() > 1e6 {
                *x = 0.0;
            }
        }
        for x in self.b2.iter_mut() {
            if !x.is_finite() || x.abs() > 1e6 {
                *x = 0.0;
            }
        }
    }
}
