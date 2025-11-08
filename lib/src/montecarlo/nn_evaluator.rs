use core::f64;

use candle_core::error;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::conv2d_no_bias;
use candle_nn::linear;
use candle_nn::AdamW;
use candle_nn::Conv2d;
use candle_nn::Conv2dConfig;
use candle_nn::Linear;
use candle_nn::Module;
use candle_nn::Optimizer;
use candle_nn::VarBuilder;
use candle_nn::VarMap;

use crate::learning::simulation::DataLoader;
use crate::learning::simulation::Dir;
use crate::learning::simulation::MoveLog;
use crate::models::Board;
use crate::models::Coord;
use crate::montecarlo::evaulator::Evaluator;
use crate::montecarlo::evaulator::MovePolicy;
use crate::utils;

// This includes the 11x11 board plus the 2 extra rows and columns for the walls.
const BOARD_SIZE: usize = 14;

pub struct SimpleConv {
    cv1: Conv2d,
    cv2: Conv2d,
    ln1: Linear,
    ln2: Linear,

    // Stores parameters for inference and training like loss.
    var_map: VarMap,

    // Device the model is loaded onto.
    pub device: Device,
}

impl SimpleConv {
    pub fn new() -> Result<Self, candle_core::error::Error> {
        let var_map = VarMap::new();
        let device = Device::new_cuda(0)?;
        // Input matrix size = |batch|x5x11x11.
        let vb = VarBuilder::from_varmap(&var_map, DType::F64, &device);

        let cv1 =
            conv2d_no_bias(5, 32, 3, Conv2dConfig::default(), vb.pp("cv1"))?;
        let cv2 =
            conv2d_no_bias(32, 64, 3, Conv2dConfig::default(), vb.pp("cv2"))?;

        // Output matrix size 6400 = |batch|x64x(14-4)x(14-4);
        let ln1 = linear(6400, 200, vb.pp("ln1"))?;

        // The out dimensions correspond to -
        //
        // Value - 2 [First player win chance, Second player win chance].
        //
        // The policy - The 4 priors of each direction [UP, DOWN, LEFT, RIGHT].
        //
        let ln2 = linear(200, 6, vb.pp("ln2"))?;
        return Ok(Self {
            cv1,
            cv2,
            ln1,
            ln2,
            var_map,
            device,
        });
    }

    pub fn load_weights(
        &mut self,
        path: &str,
    ) -> Result<(), candle_core::error::Error> {
        self.var_map.load(path)
    }

    fn forward(&self, x: Tensor) -> Result<Tensor, candle_core::error::Error> {
        let x = self.cv1.forward(&x)?;
        let x = self.cv2.forward(&x)?;
        // Flatten everything but the batch dimension.
        let x = x.flatten(1, x.dims().len() - 1)?;
        let x = x.relu()?;
        let x = self.ln1.forward(&x)?.relu()?;

        // This will now be a batchx6x1 matrix.
        let x = self.ln2.forward(&x)?;
        return Ok(x);
    }

    fn log_to_label(&self, log: &MoveLog) -> Tensor {
        let mut vec = vec![0.0; 6];

        // Set the winner within the first two indices.
        let windex = log.get_winner_index() as usize;
        vec[windex] = 1.0;

        vec[2] = log.policy_prior(Dir::UP);
        vec[3] = log.policy_prior(Dir::DOWN);
        vec[4] = log.policy_prior(Dir::LEFT);
        vec[5] = log.policy_prior(Dir::RIGHT);

        let dim = vec.len();
        return Tensor::from_vec(vec, dim, &self.device).unwrap();
    }

    fn batch_log_to_label(&self, logs: &Vec<MoveLog>) -> Tensor {
        let tensors: Vec<Tensor> =
            logs.iter().map(|l| self.log_to_label(l)).collect();
        Tensor::stack(&tensors, 0).unwrap()
    }

    pub fn save(&self, path: &str) -> Result<(), error::Error> {
        self.var_map.save(path)
    }

    // Make this less tightly coupled.
    //
    // Lets assume this is a large amount of moves around 10,000.
    //
    // We will split it into training and validation so we can verify
    // that the model is training at all.
    pub fn train(
        &self,
        data: &Vec<MoveLog>,
    ) -> Result<(), candle_core::error::Error> {
        println!("Training on batch size {} ", data.len());
        let mut optimiser = AdamW::new_lr(self.var_map.all_vars(), 0.0001)?;
        let mut epoch = 0;
        let chunk_size = (data.len() / 100) + 1;
        println!("Training on breaking batch into chunks of {} ", chunk_size);

        for batch in data.chunks(chunk_size) {
            let boards = batch.iter().map(|l| return &l.board).collect();
            let batch_tensor =
                NNEvaulator::generate_input_tensor_batch(&boards);
            let batch_tensor = batch_tensor.to_device(&self.device)?;

            let inference = self.forward(batch_tensor)?;
            println!("Inference tensor {:?}", inference);
            let targets = self.batch_log_to_label(&batch);
            println!("Target tensor {:?}", targets);
            let loss = candle_nn::loss::binary_cross_entropy_with_logit(
                &inference, &targets,
            )?;
            println!("Caltulated loss");
            optimiser.backward_step(&loss)?;
            println!("Epoch: {epoch}, Loss: {:?}", loss.to_vec0::<f64>()?);
            epoch += 1;
        }

        Ok(())
    }
}

// Uses a neural network to determine the winner from a
// given board state.
pub struct NNEvaulator {
    model: SimpleConv,
}

impl NNEvaulator {
    pub fn new() -> Result<Self, error::Error> {
        return Ok(Self {
            model: SimpleConv::new()?,
        });
    }

    pub fn load_weights(
        &mut self,
        path: &str,
    ) -> Result<(), candle_core::error::Error> {
        self.model.load_weights(path)
    }

    // Takes in a single vector of points and encodes them into an 14x14
    // tensor one hot encoded.
    pub fn one_hot_encode(points: &Vec<Coord>) -> Tensor {
        let mut grid: Vec<f64> = vec![0.0; BOARD_SIZE * BOARD_SIZE];

        for coord in points {
            // We shift the x and y over 1 to project them from the range [-1, 12]x[-1, 12] to [0, 13][0, 13].
            let idx = coord.y_shifted(1) * (BOARD_SIZE) + coord.x_shifted(1);
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
    // This means the tensor will be a 1x14x14.
    //
    // Snake 1 body layer 14x14.
    // Snake 1 head layer 14x14.
    // Snake 2 body layer 14x14.
    // Snake 2 head layer 14x14.
    // Food layer 14x14.
    fn generate_input_tensor(board: &Board) -> Tensor {
        // Generate a channel layer for each snake and their head.
        let mut channels: Vec<Tensor> = vec![];
        for snake in &board.snakes {
            // It might be worth seeing if removing the head from the body
            // channel is helpful.
            channels.push(NNEvaulator::one_hot_encode(&snake.body));
            let head_channel = vec![snake.head.clone()];
            channels.push(NNEvaulator::one_hot_encode(&head_channel));
        }
        // Add the food channel last.
        channels.push(NNEvaulator::one_hot_encode(&board.food));
        return Tensor::stack(&channels, 0).unwrap();
    }

    fn generate_input_tensor_batch(boards: &Vec<&Board>) -> Tensor {
        let mut tensors = vec![];
        for b in boards {
            tensors.push(NNEvaulator::generate_input_tensor(b));
        }
        return Tensor::stack(&tensors, 0).unwrap();
    }

    fn generate_output(&self, board: &Board, current_snake: &str) -> Vec<f64> {
        let starter_snake = board.get_snake(current_snake);
        let mut board = board.clone();
        utils::fix_snake_order(&mut board, starter_snake.clone());

        let boards = vec![&board];

        let input_tensor = NNEvaulator::generate_input_tensor_batch(&boards);
        let input_tensor = input_tensor.to_device(&self.model.device).unwrap();
        let output = self.model.forward(input_tensor).unwrap();
        let output = output.flatten(0, output.dims().len() - 1).unwrap();
        output.to_vec1::<f64>().unwrap()
    }
}

impl Evaluator for NNEvaulator {
    fn predict_winner(&self, board: &Board, current_snake: &str) -> String {
        let output = self.generate_output(board, current_snake);
        let mut max_idx = 0;
        let mut max_v = 0.0;
        for (idx, v) in output.iter().enumerate() {
            // Only the first 2 indices are used for predicting winner.
            if idx > 1 {
                break;
            }
            if v > &max_v {
                max_idx = idx;
                max_v = *v;
            }
        }
        return board.snakes[max_idx].id.clone();
    }

    fn predict_best_moves(
        &self,
        board: &Board,
        current_snake: &str,
    ) -> Vec<super::evaulator::MovePolicy> {
        let output = self.generate_output(board, current_snake);
        let mut vec = vec![];
        // UP
        vec.push(MovePolicy {
            dir: (1, 0),
            p: output[2],
        });
        // DOWN
        vec.push(MovePolicy {
            dir: (-1, 0),
            p: output[3],
        });
        // LEFT
        vec.push(MovePolicy {
            dir: (0, -1),
            p: output[4],
        });
        // RIGHT
        vec.push(MovePolicy {
            dir: (0, 1),
            p: output[5],
        });
        vec
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
        vec.push(Coord { x: -1, y: 12 });

        let tensor = NNEvaulator::one_hot_encode(&vec);
        let o_vec = tensor.to_vec2::<f64>()?;

        println!("nested tensor: {:?}", o_vec);

        assert_eq!(o_vec[11][11], 1.0);
        assert_eq!(o_vec[1][1], 1.0);
        assert_eq!(o_vec[6][6], 1.0);
        assert_eq!(o_vec[11][4], 1.0);
        assert_eq!(o_vec[13][0], 1.0);

        Ok(())
    }

    #[test]
    fn generate_input_tensor() -> Result<(), Box<dyn std::error::Error>> {
        let board = get_board();

        let tensor = NNEvaulator::generate_input_tensor(&board.board);

        assert_eq!(tensor.dims(), [5, 14, 14]);

        Ok(())
    }

    #[test]
    fn generate_input_tensor_batch() -> Result<(), Box<dyn std::error::Error>> {
        let board = get_board();
        let boards = vec![&board.board, &board.board];
        let tensor = NNEvaulator::generate_input_tensor_batch(&boards);

        assert_eq!(tensor.dims(), [2, 5, 14, 14]);

        Ok(())
    }

    #[test]
    fn evaulate_board() -> Result<(), Box<dyn std::error::Error>> {
        let board = get_board();
        let nn_eval = NNEvaulator::new()?;

        // Expect that it returned a valid string
        assert!(nn_eval.predict_winner(&board.board, &board.you.id).len() > 0);

        Ok(())
    }

    #[test]
    fn predict_policy() -> Result<(), Box<dyn std::error::Error>> {
        let board = get_board();
        let nn_eval = NNEvaulator::new()?;

        // Expect that it returned a length 4 vec.
        assert!(
            nn_eval
                .predict_best_moves(&board.board, &board.you.id)
                .len()
                == 4
        );

        Ok(())
    }
}
