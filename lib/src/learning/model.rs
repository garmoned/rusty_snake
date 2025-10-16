use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

pub struct Model {
    pub first: Linear,
    pub second: Linear,
}

impl Model {
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}
