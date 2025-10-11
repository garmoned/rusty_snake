use std::convert::TryInto;

use candle_core::error;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::conv2d_no_bias;
use candle_nn::linear;
use candle_nn::Conv2d;
use candle_nn::Conv2dConfig;
use candle_nn::Linear;
use candle_nn::Module;
use candle_nn::VarBuilder;
use candle_nn::VarMap;

use crate::models::Board;
use crate::models::Coord;
use crate::montecarlo::evaulator::Evaluator;
use crate::utils;

const BOARD_SIZE: usize = 11;
const NUM_SNAKES: usize = 2;

pub struct SimpleConv {
    cv1: Conv2d,
    ln1: Linear,
    ln2: Linear,
}

impl SimpleConv {
    fn new(vb: VarBuilder) -> Result<Self, candle_core::error::Error> {
        let cv1 =
            conv2d_no_bias(5, 20, 3, Conv2dConfig::default(), vb.pp("cv1"))?;
        let ln1 = linear(1620, 200, vb.pp("ln1"))?;
        let ln2 = linear(200, 2, vb.pp("ln2"))?;
        return Ok(Self { cv1, ln1, ln2 });
    }

    fn forward(&self, x: Tensor) -> Result<Tensor, candle_core::error::Error> {
        let x = self.cv1.forward(&x).unwrap();
        // Flatten everything but the batch dimension.
        let x = x.flatten(1, x.dims().len() - 1).unwrap();
        let x = x.relu().unwrap();
        let x = self.ln1.forward(&x).unwrap().relu().unwrap();
        let x = self.ln2.forward(&x).unwrap();
        // This should be a 1 dimensional matrix now.
        let x = candle_nn::ops::softmax(&x, 1)?;
        return Ok(x);
    }
}

// Uses a neural network to determine the winner from a
// given board state.
pub struct NNEvaulator {
    model: SimpleConv,
}

impl NNEvaulator {
    pub fn new(dev: &Device) -> Result<Self, error::Error> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F64, dev);
        return Ok(Self {
            model: SimpleConv::new(vs.clone())?,
        });
    }

    // Takes in a single vector of points and encodes them into an 11x11
    // tensor one hot encoded.
    pub fn one_hot_encode(&self, points: &Vec<Coord>) -> Tensor {
        let mut grid: Vec<f64> = vec![0.0; BOARD_SIZE * BOARD_SIZE];

        for coord in points {
            if !coord.in_bounds(
                BOARD_SIZE.try_into().unwrap(),
                BOARD_SIZE.try_into().unwrap(),
            ) {
                panic!("coord is out of bounds something terrible happened.");
            }
            let idx = coord.y() * (BOARD_SIZE) + coord.x();
            grid[idx] = 1.0;
        }
        return Tensor::from_vec(grid, (BOARD_SIZE, BOARD_SIZE), &Device::Cpu)
            .unwrap();
    }

    // The tensor will have a channel for :
    // 1. Each snake's body.
    // 2. Each snake's head.
    // 3. Food.
    //
    // Currently only has support for 2 snakes playing.
    // No support for obstacles and assumes a fixed board size of 10x10.
    //
    // The first snake layer represents the snake who is about to move.
    //
    // The model will have 2 outputs - which correspond to the snake layers.
    //
    // This means the tensor will be a 1x11x11.
    //
    // Snake 1 body layer 11x11.
    // Snake 1 head layer 11x11.
    // Snake 2 body layer 11x11.
    // Snake 2 head layer 11x11.
    // Food layer 11x11.
    fn generate_input_tensor(
        &self,
        board: &Board,
        current_snake: &str,
    ) -> Tensor {
        // Generate a channel layer for each snake and their head.
        let mut channels: Vec<Tensor> = vec![];
        for snake in &board.snakes {
            // It might be worth seeing if removing the head from the body
            // channel is helpful.
            channels.push(self.one_hot_encode(&snake.body));
            let head_channel = vec![snake.head.clone()];
            channels.push(self.one_hot_encode(&head_channel));
        }
        // Add the food channel last.
        channels.push(self.one_hot_encode(&board.food));
        let batch = [Tensor::stack(&channels, 0).unwrap()];
        return Tensor::stack(&batch, 0).unwrap();
    }
}

impl Evaluator for NNEvaulator {
    fn predict_winner(&self, board: &Board, current_snake: &str) -> String {
        let starter_snake = board.get_snake(current_snake);
        let mut board = board.clone();
        utils::fix_snake_order(&mut board, starter_snake.clone());

        let input_tensor = self.generate_input_tensor(&board, current_snake);
        let output = self.model.forward(input_tensor).unwrap();
        let output = output.flatten(0, output.dims().len() - 1).unwrap();
        let output = output.to_vec1::<f64>().unwrap();

        let mut max_idx = 0;
        let mut max_v = 0.0;
        for (idx, v) in output.iter().enumerate() {
            if v > &max_v {
                max_idx = idx;
                max_v = *v;
            }
        }
        return board.snakes[max_idx].id.clone();
    }
}

#[cfg(test)]
mod test {

    use core::f64;

    use crate::test_utils::scenarios::get_board;

    use super::*;

    // Verifies basic behavior about the snake.
    #[test]
    fn test_one_channel() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;

        let mut vec: Vec<Coord> = vec![];
        vec.push(Coord { x: 10, y: 10 });
        vec.push(Coord { x: 0, y: 0 });
        vec.push(Coord { x: 5, y: 5 });
        vec.push(Coord { x: 3, y: 10 });

        let nn_eval = NNEvaulator::new(&device)?;
        let tensor = nn_eval.one_hot_encode(&vec);
        let o_vec = tensor.to_vec2::<f64>()?;

        println!("nested tensor: {:?}", o_vec);

        assert_eq!(o_vec[10][10], 1.0);
        assert_eq!(o_vec[0][0], 1.0);
        assert_eq!(o_vec[5][5], 1.0);
        assert_eq!(o_vec[10][3], 1.0);

        Ok(())
    }

    #[test]
    fn generate_input_tensor() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let board = get_board();

        let nn_eval = NNEvaulator::new(&device)?;
        let tensor = nn_eval.generate_input_tensor(&board.board, &board.you.id);

        assert_eq!(tensor.dims(), [1, 5, 11, 11]);

        Ok(())
    }

    #[test]
    fn evaulate_board() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let board = get_board();
        let nn_eval = NNEvaulator::new(&device)?;

        // Expect that it returned a valid string
        assert!(nn_eval.predict_winner(&board.board, &board.you.id).len() > 0);

        Ok(())
    }
}
