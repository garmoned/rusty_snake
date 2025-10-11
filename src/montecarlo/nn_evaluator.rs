use std::convert::TryInto;

use candle_core::Device;
use candle_core::Tensor;

use crate::models::Board;
use crate::models::Coord;
use crate::utils;
use crate::{board, montecarlo::evaulator::Evaluator};

const BOARD_SIZE: usize = 11;
const NUM_SNAKES: usize = 2;

// Uses a neural network to determine the winner from a
// given board state.
pub struct NNEvaulator {}

impl NNEvaulator {
    pub fn new() -> Self {
        return Self {};
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
        // Sort the snakes such that the current snake comes first.
        let starter_snake = board.get_snake(current_snake);
        let mut board = board.clone();
        utils::fix_snake_order(&mut board, starter_snake.clone());

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
        return Tensor::stack(&channels, 2).unwrap();
    }
}

impl Evaluator for NNEvaulator {
    fn predict_winner(&self, board: &Board, current_snake: &str) -> String {
        todo!()
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
        let mut vec: Vec<Coord> = vec![];
        vec.push(Coord { x: 10, y: 10 });
        vec.push(Coord { x: 0, y: 0 });
        vec.push(Coord { x: 5, y: 5 });
        vec.push(Coord { x: 3, y: 10 });

        let nn_eval = NNEvaulator::new();
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
        let board = get_board();

        let nn_eval = NNEvaulator::new();
        let tensor = nn_eval.generate_input_tensor(&board.board, &board.you.id);

        assert_eq!(tensor.dims(), [11, 11, 5]);

        Ok(())
    }
}
