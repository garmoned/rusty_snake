use core::f32;

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
use rand::seq::SliceRandom;
use rand::thread_rng;

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
        x: &Tensor,
        scalar: Tensor,
        common_model: &CommonModel,
    ) -> Result<Tensor, candle_core::error::Error>;
    fn label_batch(&self, logs: &[MoveLog]) -> Tensor;
    fn loss_fn(
        &self,
        inference: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, candle_core::error::Error>;
}

pub struct CommonModel {
    cv1: Conv2d,
    cv2: Conv2d,

    // Normalization layer.
    ln_norm: candle_nn::LayerNorm,

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
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let cv1 =
            conv2d_no_bias(6, 256, 3, Conv2dConfig::default(), vb.pp("cv1"))?;
        let cv2 =
            conv2d_no_bias(256, 64, 3, Conv2dConfig::default(), vb.pp("cv2"))?;

        let ln_norm = candle_nn::layer_norm(6400, 1e-5, vb.pp("ln_norm"))?;

        // Output matrix size 6400 = |batch|x64x(14-4)x(14-4) + 4 input scalar dimension.
        let ln1 = linear(6404, 200, vb.pp("ln1"))?;

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
            ln_norm,
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

    fn get_optimizer(&self) -> Result<AdamW, candle_core::error::Error> {
        AdamW::new_lr(self.var_map.all_vars(), 0.0001)
    }

    fn common_forward(
        &self,
        x: &Tensor,
        // In the form Batch X Scalars.
        scalars: Tensor,
    ) -> Result<Tensor, candle_core::error::Error> {
        let x = self.cv1.forward(x)?;
        let x = x.relu()?;
        let x = self.cv2.forward(&x)?;
        // Flatten everything but the batch dimension.
        let x = x.flatten(1, x.dims().len() - 1)?;
        let x = x.relu()?;
        let x = self.ln_norm.forward(&x)?;

        // X should now be in the shape of BATCH * 6400 + 4 to include
        // the secondary scalar input.
        let x = candle_core::Tensor::cat(&[x, scalars], 1)?;
        let x = self.ln1.forward(&x)?.relu()?;
        Ok(x)
    }

    fn train(
        &self,
        data: &mut [MoveLog],
        units: Vec<&dyn ModelUnit>,
        optimiser: &mut AdamW,
    ) -> Result<(), candle_core::error::Error> {
        println!("Training on batch size {} ", data.len());
        let chunk_size = 256;
        println!("Training on breaking batch into chunks of {} ", chunk_size);

        data.shuffle(&mut thread_rng());

        for batch in data.chunks(chunk_size) {
            let boards = batch.iter().map(|l| &l.board).collect();
            let batch_tensor =
                NNEvaulator::generate_input_tensor_batch(&boards);
            let batch_tensor = batch_tensor.to_device(&self.device)?;

            let mut loss: Option<Tensor> = None;

            for unit in units.as_slice() {
                let batch_scalar =
                    NNEvaulator::generate_input_scalar_batch(&boards)
                        .to_device(&self.device)?;

                let inference =
                    unit.forward(&batch_tensor, batch_scalar, self)?;
                let targets = unit.label_batch(batch);
                let local_loss = unit.loss_fn(&inference, &targets)?;
                match loss {
                    Some(g_loss) => loss = Some(g_loss.add(&local_loss)?),
                    None => loss = Some(local_loss),
                }
            }

            match loss {
                Some(g_loss) => {
                    optimiser.backward_step(&g_loss)?;
                    println!("Caltulated loss: {:?}", g_loss.to_vec0::<f32>())
                }
                None => panic!("No loss calculated"),
            }
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
        let mut vec: Vec<f32> = vec![0.0; 4];
        vec[utils::UP_IDX] = log.policy_prior(Dir::UP) as f32;
        vec[utils::DOWN_IDX] = log.policy_prior(Dir::DOWN) as f32;
        vec[utils::LEFT_IDX] = log.policy_prior(Dir::LEFT) as f32;
        vec[utils::RIGHT_IDX] = log.policy_prior(Dir::RIGHT) as f32;
        Tensor::from_vec(vec, 4, &self.device).unwrap()
    }
}

pub struct ValueUnit {
    device: Device,
}

impl ValueUnit {
    fn label_fn(&self, log: &MoveLog) -> Tensor {
        let mut vec: Vec<f32> = vec![0.0; 1];
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

/// Computes Cross-Entropy loss for soft targets (probability distributions).
///
/// * `logits`: The raw output from the neural network (BEFORE Softmax). Shape: [Batch, Actions]
/// * `targets`: The MCTS probabilities (sum to 1.0). Shape: [Batch, Actions]
pub fn soft_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<Tensor, candle_core::error::Error> {
    // 1. Convert raw logits to Log-Probabilities
    // This is more numerically stable than doing softmax -> log
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;

    // 2. Calculate: target * log_probs
    // We do element-wise multiplication.
    let weighted_log_probs = targets.mul(&log_probs)?;

    // 3. Sum across the action dimension (dim 1)
    // This gives us the negative log-likelihood for each sample in the batch.
    let sum_across_actions = weighted_log_probs.sum(1)?;

    // 4. Negate (because loss must be positive)
    let neg_log_likelihood = sum_across_actions.neg()?;

    // 5. Average across the batch dimension (dim 0) to get a scalar loss
    let mean_loss = neg_log_likelihood.mean(0)?;

    Ok(mean_loss)
}

impl ModelUnit for PolicyUnit {
    fn forward(
        &self,
        x: &Tensor,
        scalars: Tensor,
        common_model: &CommonModel,
    ) -> Result<Tensor, candle_core::error::Error> {
        let x = common_model.common_forward(x, scalars)?;
        let x = common_model.pn.forward(&x)?;

        // Return raw logits (no softmax).
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
        // Use soft cross entropy loss for policy.
        let loss = soft_cross_entropy(inference, targets)?;
        Ok(loss)
    }
}

impl ModelUnit for ValueUnit {
    fn forward(
        &self,
        x: &Tensor,
        scalars: Tensor,
        common_model: &CommonModel,
    ) -> Result<Tensor, candle_core::error::Error> {
        let x = common_model.common_forward(x, scalars)?;
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
            common,
            policy_unit,
            value_unit,
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
        x: &Tensor,
        scalars: Tensor,
    ) -> Result<Tensor, candle_core::error::Error> {
        self.policy_unit.forward(x, scalars, &self.common)
    }

    pub fn value_forward(
        &self,
        x: &Tensor,
        scalars: Tensor,
    ) -> Result<Tensor, candle_core::error::Error> {
        self.value_unit.forward(x, scalars, &self.common)
    }

    pub fn get_optimizer(&self) -> Result<AdamW, error::Error> {
        self.common.get_optimizer()
    }

    pub fn train(
        &self,
        data: &mut [MoveLog],
        optimiser: &mut AdamW,
    ) -> Result<(), candle_core::error::Error> {
        let units: Vec<&dyn ModelUnit> =
            vec![&self.policy_unit, &self.value_unit];
        self.common.train(data, units, optimiser)?;
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
        Ok(Self {
            model: MultiOutputModel::new()?,
        })
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
        let mut grid: Vec<f32> = vec![0.0; BOARD_SIZE * BOARD_SIZE];

        for coord in points {
            // We shift the x and y over 1 to project them from the range [-1, 12]x[-1, 12] to [0, 13][0, 13].
            let idx = coord.y_shifted(1) * (BOARD_SIZE) + coord.x_shifted(1);
            grid[idx] = 1.0;
        }
        Tensor::from_vec(grid, (BOARD_SIZE, BOARD_SIZE), &Device::Cpu)
            .unwrap()
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
            channels.push(NNEvaulator::one_hot_encode(&snake.body));
            let head_channel = vec![snake.head.clone()];
            channels.push(NNEvaulator::one_hot_encode(&head_channel));
        }
        // Add the food channel last.
        channels.push(NNEvaulator::one_hot_encode(&board.food));
        let mut walls = vec![];
        for i in -1..=12 {
            walls.push(Coord { x: i, y: -1 });
            walls.push(Coord { x: -1, y: i });
            walls.push(Coord { x: i, y: 12 });
            walls.push(Coord { x: 12, y: i });
        }
        channels.push(NNEvaulator::one_hot_encode(&walls));
        Tensor::stack(&channels, 0).unwrap()
    }

    // Generates a 1D tensor in the format of
    // [My_Length, My_Health, Enemy_Length, Enemy_Health].
    // Will have a length of 4.
    fn generate_scalar_inputs(board: &Board) -> Tensor {
        let mut scalars: Vec<f32> = vec![];
        for snake in &board.snakes {
            // Length normalized to max length of entire board.
            scalars.push(snake.body.len() as f32 / (11.0 * 11.0));
            // Health normalized to max health of 100.
            scalars.push((snake.health as f32) / 100.0);
        }
        Tensor::from_vec(scalars, 4, &Device::Cpu).unwrap()
    }

    fn generate_input_scalar_batch(boards: &Vec<&Board>) -> Tensor {
        let mut tensors = vec![];
        for b in boards {
            tensors.push(NNEvaulator::generate_scalar_inputs(b));
        }
        Tensor::stack(&tensors, 0).unwrap()
    }

    fn generate_input_tensor_batch(boards: &Vec<&Board>) -> Tensor {
        let mut tensors = vec![];
        for b in boards {
            tensors.push(NNEvaulator::generate_input_tensor(b));
        }
        Tensor::stack(&tensors, 0).unwrap()
    }
}

impl Evaluator for NNEvaulator {
    fn predict_winner(&self, board: &Board, _: &str) -> String {
        let input = NNEvaulator::generate_input_tensor_batch(&vec![&board])
            .to_device(&self.model.common.device)
            .unwrap();
        let scalars = NNEvaulator::generate_input_scalar_batch(&vec![&board])
            .to_device(&self.model.common.device)
            .unwrap();

        // Output is in the format batch X value.
        let output = self.model.value_forward(&input, scalars).unwrap();
        let output = output.squeeze(0).unwrap();
        let output = output.to_vec1::<f32>().unwrap()[0];
        // println!("value of board {}", output);
        if output > 0.1 {
            return board.snakes[0].id.clone();
        }
        if output < -0.1 {
            board.snakes[1].id.clone()
        } else {
            "tie".to_string()
        }
    }

    fn predict_value(
        &self,
        board: &Board,
        target_snake: &str,
        _current_snake: &str, // unused for the neural net evaluator.
    ) -> f32 {
        // Currently there is no way to tell the neural network to predict the
        // winner of anyone else besides the current snake.
        //
        // So we set the current snake to the target snake.

        let snake = board.get_snake(target_snake);
        let mut copy_board = board.clone();
        utils::fix_snake_order(&mut copy_board, snake.clone());

        let input = NNEvaulator::generate_input_tensor_batch(&vec![&board])
            .to_device(&self.model.common.device)
            .unwrap();
        let scalars = NNEvaulator::generate_input_scalar_batch(&vec![&board])
            .to_device(&self.model.common.device)
            .unwrap();

        // Output is in the format batch X value.
        let output = self.model.value_forward(&input, scalars).unwrap();
        let output = output.squeeze(0).unwrap();
        output.to_vec1::<f32>().unwrap()[0]
    }

    fn predict_best_moves(
        &self,
        board: &Board,
        _: &str,
    ) -> Vec<super::evaulator::MovePolicy> {
        let input = NNEvaulator::generate_input_tensor_batch(&vec![&board])
            .to_device(&self.model.common.device)
            .unwrap();
        let scalars = NNEvaulator::generate_input_scalar_batch(&vec![&board])
            .to_device(&self.model.common.device)
            .unwrap();
        let output = self.model.policy_forward(&input, scalars).unwrap();
        // Apply softmax to the output to generate a probability distribution.
        let output = candle_nn::ops::softmax(&output, 1).unwrap();
        let output = output.squeeze(0).unwrap();
        let output = output.to_vec1::<f32>().unwrap();
        let mut policies = vec![];
        policies.push(MovePolicy {
            dir: utils::RIGHT,
            p: output[utils::RIGHT_IDX] as f64,
        });
        policies.push(MovePolicy {
            dir: utils::UP,
            p: output[utils::UP_IDX] as f64,
        });
        policies.push(MovePolicy {
            dir: utils::LEFT,
            p: output[utils::LEFT_IDX] as f64,
        });
        policies.push(MovePolicy {
            dir: utils::DOWN,
            p: output[utils::DOWN_IDX] as f64,
        });
        policies
    }
}

#[cfg(test)]
mod test {

    use core::f32;

    use crate::config::{self, MonteCarloConfig};
    use crate::montecarlo;
    use crate::test_utils::scenarios::{
        get_board, get_scenario, AVOID_DEATH_ADVANCED, AVOID_DEATH_GET_FOOD,
    };

    use crate::utils::dir_to_string;

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
        let o_vec = tensor.to_vec2::<f32>()?;

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

        assert_eq!(tensor.dims(), [6, 14, 14]);

        Ok(())
    }

    #[test]
    fn generate_input_tensor_batch() -> Result<(), Box<dyn std::error::Error>> {
        let board = get_board();
        let boards = vec![&board.board, &board.board];
        let tensor = NNEvaulator::generate_input_tensor_batch(&boards);

        assert_eq!(tensor.dims(), [2, 6, 14, 14]);

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

    #[test]
    fn verify_policy_layer() -> Result<(), Box<dyn std::error::Error>> {
        let game_state = get_scenario(AVOID_DEATH_GET_FOOD);
        let mut evaulator = NNEvaulator::new()?;

        print!("Board \n{}-\n", game_state.board.to_string());
        println!("You - id: {}", game_state.you.id);
        let weights = std::path::Path::new("../data/models/basic.safetensor");

        let mut config = MonteCarloConfig::default();
        config.evaulator = config::Evaluator::NEURAL;

        let mut tree = montecarlo::tree::Tree::new(
            config,
            game_state.board.clone(),
            game_state.board.snakes[0].clone(),
        );

        let tree_moves = tree.get_best_move_with_policy();

        if weights.exists() {
            println!("loading existing weights");
            evaulator.load_weights("../data/models/basic.safetensor")?;
        }
        let moves =
            evaulator.predict_best_moves(&game_state.board, &game_state.you.id);

        println!("best move for: {}", game_state.you.id);

        let best_move =
            moves.iter().max_by(|m1, m2| m1.p.total_cmp(&m2.p)).unwrap();

        let mut m_board = game_state.board.clone();

        m_board.execute(&game_state.you.id, best_move.dir, false);

        println!("Is terminal {}", m_board.is_terminal());

        println!("Winner is {:?}", m_board.get_endstate());

        print!("Board \n{}-\n", m_board.to_string());

        let best_move = dir_to_string(best_move.dir);

        evaulator.predict_winner(&m_board, &game_state.you.id);

        for m in &moves {
            println!("policy move {} - {:?}", dir_to_string(m.dir), m.p)
        }

        println!(" ------ ");

        for m in &tree_moves.1 {
            println!("tree move {} - {:?}", dir_to_string(m.dir), m.p)
        }

        assert_eq!(best_move, "up");

        Ok(())
    }

    #[test]
    fn verify_value_layer() -> Result<(), Box<dyn std::error::Error>> {
        let game_state = get_scenario(AVOID_DEATH_ADVANCED);
        let mut evaulator = NNEvaulator::new()?;

        let weights = std::path::Path::new("../data/models/basic.safetensor");

        if weights.exists() {
            evaulator.load_weights("../data/models/basic.safetensor")?;
        }

        let winner =
            evaulator.predict_winner(&game_state.board, &game_state.you.id);

        assert_eq!(winner, "a425095b-604b-4b67-a281-4dac219f4bee");

        Ok(())
    }

    #[test]
    fn back_prop_one_move_log() -> Result<(), Box<dyn std::error::Error>> {
        let game_state = get_scenario(AVOID_DEATH_GET_FOOD);
        let evaulator = NNEvaulator::new()?;

        let best_moves =
            evaulator.predict_best_moves(&game_state.board, &game_state.you.id);

        let mut config = MonteCarloConfig::default();
        config.evaulator = config::Evaluator::NEURAL;
        let mut tree = montecarlo::tree::Tree::new(
            config,
            game_state.board.clone(),
            game_state.you.clone(),
        );

        let tree_best_moves = tree.get_best_move_with_policy();

        for m in &best_moves {
            // Print out the move and policy value
            println!("move {} - policy value {}", dir_to_string(m.dir), m.p);
        }

        println!(" ------ ");

        for m in &tree_best_moves.1 {
            // Print out the move and policy value
            println!(
                "tree move {} - policy value {}",
                dir_to_string(m.dir),
                m.p
            );
        }

        let move_log = MoveLog {
            player: game_state.you.id.clone(),
            board: game_state.board.clone(),
            winner: Some(game_state.board.snakes[0].id.clone()),
            policy: tree_best_moves.1,
        };

        evaulator.model.train(
            &mut vec![move_log],
            &mut evaulator.model.get_optimizer()?,
        )?;

        Ok(())
    }
}
