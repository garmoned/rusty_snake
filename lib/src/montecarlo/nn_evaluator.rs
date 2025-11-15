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

use crate::learning::simulation::Dir;
use crate::learning::simulation::MoveLog;
use crate::models::Board;
use crate::models::Coord;
use crate::montecarlo::evaulator::Evaluator;
use crate::montecarlo::evaulator::MovePolicy;
use crate::utils;

// This includes the 11x11 board plus the 2 extra rows and columns for the walls.
const BOARD_SIZE: usize = 14;

trait ModelUnit {
    fn forward(
        &self,
        x: Tensor,
        common_model: &CommonModel,
    ) -> Result<Tensor, candle_core::error::Error>;
    fn label_batch(&self, logs: &[MoveLog]) -> Tensor;
    fn loss_fn(
        &self,
        inference: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, candle_core::error::Error>;
    fn train(
        &self,
        data: &Vec<MoveLog>,
        common_model: &CommonModel,
    ) -> Result<(), candle_core::error::Error>;
}

pub struct CommonModel {
    cv1: Conv2d,
    cv2: Conv2d,
    ln1: Linear,

    pub pn: Linear,
    pub vn: Linear,

    pub var_map: VarMap,
    pub device: Device,
}

impl CommonModel {
    pub fn new() -> Result<Self, candle_core::error::Error> {
        let var_map = VarMap::new();
        let device = Device::new_cuda(0)?;
        // Input matrix size = |batch|x5x11x11.
        let vb = VarBuilder::from_varmap(&var_map, DType::F64, &device);

        let cv1 =
            conv2d_no_bias(5, 256, 3, Conv2dConfig::default(), vb.pp("cv1"))?;
        let cv2 =
            conv2d_no_bias(256, 64, 3, Conv2dConfig::default(), vb.pp("cv2"))?;

        // Output matrix size 6400 = |batch|x64x(14-4)x(14-4);
        let ln1 = linear(6400, 200, vb.pp("ln1"))?;
        // The out dimensions correspond to -
        //
        // The policy - The 4 priors of each direction [UP, DOWN, LEFT, RIGHT]
        //
        let pn = linear(200, 4, vb.pp("pn"))?;

        // The value output head
        //
        // Outputs a single value predicting the chance that the snake idx = 0
        // will win.
        //
        let vn = linear(200, 1, vb.pp("vn"))?;
        Ok(Self {
            cv1,
            cv2,
            ln1,
            pn,
            vn,
            var_map,
            device,
        })
    }

    pub fn load_weights(
        &mut self,
        path: &str,
    ) -> Result<(), candle_core::error::Error> {
        self.var_map.load(path)
    }

    pub fn save_weights(
        &self,
        path: &str,
    ) -> Result<(), candle_core::error::Error> {
        self.var_map.save(path)
    }

    fn common_forward(
        &self,
        x: Tensor,
    ) -> Result<Tensor, candle_core::error::Error> {
        let x = self.cv1.forward(&x)?;
        let x = self.cv2.forward(&x)?;
        // Flatten everything but the batch dimension.
        let x = x.flatten(1, x.dims().len() - 1)?;
        let x = x.relu()?;
        let x = self.ln1.forward(&x)?.relu()?;
        Ok(x)
    }

    fn train(
        &self,
        data: &Vec<MoveLog>,
        unit: &dyn ModelUnit,
    ) -> Result<(), candle_core::error::Error> {
        println!("Training on batch size {} ", data.len());
        let mut optimiser = AdamW::new_lr(self.var_map.all_vars(), 0.0001)?;
        let chunk_size = (data.len() / 100) + 1;
        println!("Training on breaking batch into chunks of {} ", chunk_size);
        for batch in data.chunks(chunk_size) {
            let boards = batch.iter().map(|l| return &l.board).collect();
            let batch_tensor =
                NNEvaulator::generate_input_tensor_batch(&boards);
            let batch_tensor = batch_tensor.to_device(&self.device)?;

            let inference = unit.forward(batch_tensor, &self)?;
            println!("Inference tensor {:?}", inference);
            let targets = unit.label_batch(&batch);
            println!("Target tensor {:?}", targets);
            let loss = unit.loss_fn(&inference, &targets)?;
            optimiser.backward_step(&loss)?;
            println!("Caltulated loss: {:?}", loss.to_vec0::<f64>());
        }
        Ok(())
    }
}

pub struct PolicyUnit {
    device: Device,
}

impl PolicyUnit {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    fn label_fn(&self, log: &MoveLog) -> Tensor {
        let mut vec = vec![0.0; 4];
        vec[utils::UP_IDX] = log.policy_prior(Dir::UP);
        vec[utils::DOWN_IDX] = log.policy_prior(Dir::DOWN);
        vec[utils::LEFT_IDX] = log.policy_prior(Dir::LEFT);
        vec[utils::RIGHT_IDX] = log.policy_prior(Dir::RIGHT);
        Tensor::from_vec(vec, 4, &self.device).unwrap()
    }
}

pub struct ValueUnit {
    device: Device,
}

impl ValueUnit {
    fn label_fn(&self, log: &MoveLog) -> Tensor {
        let mut vec = vec![0.0; 1];
        match log.get_winner_index() {
            Some(idx) => {
                // If the winner is the first snake, set the value to 1.0.
                if idx == 0 {
                    vec[0] = 1.0;
                } else {
                    // If the winner is not the first snake, set the value to -1.0.
                    vec[0] = -1.0;
                }
            }
            None => {
                // Ties are set to 0.0.
                vec[0] = 0.0;
            }
        }
        Tensor::from_vec(vec, 1, &self.device).unwrap()
    }
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl ModelUnit for PolicyUnit {
    fn forward(
        &self,
        x: Tensor,
        common_model: &CommonModel,
    ) -> Result<Tensor, candle_core::error::Error> {
        let x = common_model.common_forward(x)?;
        let x = common_model.pn.forward(&x)?;

        // Apply softmax so that all outcomes are scaled to sum to 1.
        let x = candle_nn::ops::softmax(&x, 1)?;
        Ok(x)
    }

    fn label_batch(&self, logs: &[MoveLog]) -> Tensor {
        let mut tensors = vec![];
        for log in logs {
            tensors.push(self.label_fn(log));
        }
        Tensor::stack(&tensors, 0).unwrap()
    }

    fn loss_fn(
        &self,
        inference: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, candle_core::error::Error> {
        candle_nn::loss::mse(inference, targets)
    }
    fn train(
        &self,
        data: &Vec<MoveLog>,
        common_model: &CommonModel,
    ) -> Result<(), candle_core::error::Error> {
        common_model.train(data, self)
    }
}

impl ModelUnit for ValueUnit {
    fn forward(
        &self,
        x: Tensor,
        common_model: &CommonModel,
    ) -> Result<Tensor, candle_core::error::Error> {
        let x = common_model.common_forward(x)?;
        let x = common_model.vn.forward(&x)?;
        // Apply tanh so that the value will fall between [-1,1]
        let x = x.tanh()?;
        Ok(x)
    }

    fn label_batch(&self, logs: &[MoveLog]) -> Tensor {
        let mut tensors = vec![];
        for log in logs {
            tensors.push(self.label_fn(log));
        }
        Tensor::stack(&tensors, 0).unwrap()
    }
    fn loss_fn(
        &self,
        inference: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, candle_core::error::Error> {
        candle_nn::loss::mse(inference, targets)
    }
    fn train(
        &self,
        data: &Vec<MoveLog>,
        common_model: &CommonModel,
    ) -> Result<(), candle_core::error::Error> {
        common_model.train(data, self)
    }
}

pub struct MultiOutputModel {
    common: CommonModel,
    policy_unit: PolicyUnit,
    value_unit: ValueUnit,
}
impl MultiOutputModel {
    pub fn new() -> Result<Self, error::Error> {
        let common = CommonModel::new()?;
        let policy_unit = PolicyUnit::new(common.device.clone());
        let value_unit = ValueUnit::new(common.device.clone());
        Ok(Self {
            common: common,
            policy_unit: policy_unit,
            value_unit: value_unit,
        })
    }
    pub fn load_weights(
        &mut self,
        path: &str,
    ) -> Result<(), candle_core::error::Error> {
        self.common.load_weights(path)
    }
    pub fn save_weights(
        &self,
        path: &str,
    ) -> Result<(), candle_core::error::Error> {
        self.common.save_weights(path)
    }

    pub fn policy_forward(
        &self,
        x: Tensor,
    ) -> Result<Tensor, candle_core::error::Error> {
        self.policy_unit.forward(x, &self.common)
    }

    pub fn value_forward(
        &self,
        x: Tensor,
    ) -> Result<Tensor, candle_core::error::Error> {
        self.value_unit.forward(x, &self.common)
    }

    pub fn train(
        &self,
        data: &Vec<MoveLog>,
    ) -> Result<(), candle_core::error::Error> {
        // Should eventually parallelize this but right now traning is not a bottleneck.
        self.policy_unit.train(data, &self.common)?;
        self.value_unit.train(data, &self.common)?;
        Ok(())
    }
}

// Uses a neural network to determine the winner from a
// given board state.
pub struct NNEvaulator {
    model: MultiOutputModel,
}

impl NNEvaulator {
    pub fn new() -> Result<Self, error::Error> {
        return Ok(Self {
            model: MultiOutputModel::new()?,
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
}

impl Evaluator for NNEvaulator {
    fn predict_winner(&self, board: &Board, _: &str) -> String {
        let input = NNEvaulator::generate_input_tensor_batch(&vec![board])
            .to_device(&self.model.common.device)
            .unwrap();

        // Output is in the format batch X value.
        let output = self.model.value_forward(input).unwrap();
        let output = output.squeeze(0).unwrap();
        let output = output.to_vec1::<f64>().unwrap()[0];
        if output > 0.1 {
            return board.snakes[0].id.clone();
        }
        if output < -0.1 {
            return board.snakes[1].id.clone();
        } else {
            return "tie".to_string();
        }
    }

    fn predict_best_moves(
        &self,
        board: &Board,
        _: &str,
    ) -> Vec<super::evaulator::MovePolicy> {
        let input = NNEvaulator::generate_input_tensor_batch(&vec![board])
            .to_device(&self.model.common.device)
            .unwrap();
        let output = self
            .model
            .policy_forward(input)
            .unwrap()
            .squeeze(0)
            .unwrap();
        let output = output.to_vec1::<f64>().unwrap();
        let mut policies = vec![];
        policies.push(MovePolicy {
            dir: utils::RIGHT,
            p: output[utils::RIGHT_IDX],
        });
        policies.push(MovePolicy {
            dir: utils::UP,
            p: output[utils::UP_IDX],
        });
        policies.push(MovePolicy {
            dir: utils::LEFT,
            p: output[utils::LEFT_IDX],
        });
        policies.push(MovePolicy {
            dir: utils::DOWN,
            p: output[utils::DOWN_IDX],
        });
        policies
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
