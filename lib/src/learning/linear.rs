use candle_core::{Result, Tensor};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}
