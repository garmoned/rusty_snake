use crate::montecarlo::evaulator::{Evaluator, MovePolicy};
use crate::montecarlo::nn_evaluator::NNEvaulator;
use crate::utils;
use crate::{
    config::MonteCarloConfig,
    models::Board,
    montecarlo::{nn_evaluator::MultiOutputModel, tree::Tree},
};
use candle_nn::AdamW;
use rand::random;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::sync::Mutex;
use std::usize;
use std::{collections::HashMap, fs};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MoveLog {
    // The player who just made the move.
    pub player: String,
    // The board state before the move was made (the state the move was taken on).
    pub board: Board,
    // The player who won from this board state.
    pub winner: Option<String>,
    // Calculated policy prediction for each move.
    pub policy: Vec<MovePolicy>,
}
pub enum Dir {
    RIGHT,
    LEFT,
    UP,
    DOWN,
}

impl MoveLog {
    // Returns the index of the winner if found, otherwise None.
    pub fn get_winner_index(&self) -> Option<u32> {
        match &self.winner {
            Some(id) => self
                .board
                .snakes
                .iter()
                .enumerate()
                .find(|(_, bs)| bs.id == *id)
                .map(|(idx, _)| idx as u32),
            None => None,
        }
    }

    fn find_policy(&self, dir: (i32, i32)) -> f64 {
        for pdir in &self.policy {
            if pdir.dir == dir {
                return pdir.p;
            }
        }
        0.0
    }

    pub fn policy_prior(&self, dir: Dir) -> f64 {
        match dir {
            Dir::RIGHT => self.find_policy(utils::RIGHT),
            Dir::LEFT => self.find_policy(utils::LEFT),
            Dir::UP => self.find_policy(utils::UP),
            Dir::DOWN => self.find_policy(utils::DOWN),
        }
    }
}

pub struct DataIterator {
    file_iterator: Box<dyn Iterator<Item = String>>,
    chunk_size: usize,
}

impl DataIterator {}

impl Iterator for DataIterator {
    type Item = Vec<MoveLog>;
    fn next(&mut self) -> Option<Self::Item> {
        let chunk: Vec<MoveLog> = self
            .file_iterator
            .by_ref()
            .take(self.chunk_size)
            .map(|f| {
                println!("file path {}", f);
                serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap()
            })
            .collect();
        if !chunk.is_empty() {
            Some(chunk)
        } else {
            None
        }
    }
}

const RB_SIZE: usize = 50_000;

struct MoveLogger {
    current_game: Vec<MoveLog>,
    winners: HashMap<String, f32>,
    games_played: f32,
    buffer: VecDeque<MoveLog>,
}

impl MoveLogger {
    pub fn new() -> MoveLogger {
        Self {
            current_game: vec![],
            winners: HashMap::new(),
            games_played: 0.0,
            buffer: VecDeque::new(),
        }
    }

    pub fn get_move_view(&mut self) -> &mut [MoveLog] {
        self.buffer.make_contiguous()
    }

    // Used to merge two move loggers together after.
    // Used to merge batches together after running lots of simulations
    pub fn merge(&mut self, other: MoveLogger) {
        if self.current_game.len() > 0 {
            panic!("Should not be merging with a game in progress");
        }
        self.games_played += other.games_played;
        for (winner, times) in &other.winners {
            let times = times + self.winners.get(winner).unwrap_or(&0.0);
            self.winners.insert(winner.to_owned(), times);
        }
        self.push_to_buffer(other.buffer.into());
    }

    pub fn log_move(
        &mut self,
        player: &str,
        board: &Board,
        policy: Vec<MovePolicy>,
    ) {
        self.current_game.push(MoveLog {
            player: player.to_string(),
            board: board.clone(),
            winner: None,
            policy: policy,
        });
    }

    pub fn log_win(&mut self, winner: &str) {
        for mv in &mut self.current_game {
            mv.winner = Some(winner.to_string());
        }
        self.winners.insert(
            winner.to_owned(),
            self.winners.get(winner).unwrap_or(&0.0) + 1.0,
        );
        let game = std::mem::take(&mut self.current_game);
        self.push_to_buffer(game);
        self.games_played += 1.0;
    }

    // Should push to the buffer and push off older moves if the buffer is full.
    fn push_to_buffer(&mut self, vec: Vec<MoveLog>) {
        self.buffer.extend(vec);
        while self.buffer.len() > RB_SIZE {
            self.buffer.pop_front();
        }
    }

    pub fn log_tie(&mut self) {
        self.current_game.clear();
        self.games_played += 1.0;
    }

    pub fn print_stats(&self) {
        println!("total games played {}", self.games_played);

        for (p, wins) in &self.winners {
            println!(
                "{} won {} % of total games",
                p,
                (wins / self.games_played) * 100.0
            )
        }
    }
}

struct NNAgent {
    starting_snake_id: String,
    config: MonteCarloConfig,
    evaulator: Arc<Mutex<dyn Evaluator>>,
}

impl NNAgent {
    pub fn new(snake: String, evaulator: Arc<Mutex<dyn Evaluator>>) -> Self {
        let config = MonteCarloConfig::default();
        Self {
            starting_snake_id: snake.clone(),
            config: config,
            evaulator,
        }
    }
}

unsafe impl Sync for NNAgent {}

unsafe impl Send for NNAgent {}

impl Agent for NNAgent {
    fn id(&self) -> &str {
        &self.starting_snake_id
    }

    fn get_best_move_with_policy(
        &self,
        board: Board,
    ) -> ((i32, i32), Vec<MovePolicy>) {
        let starting_snake = board.get_snake(&self.starting_snake_id);
        let mut tree = Tree::new_with_evaulator(
            self.config.clone(),
            board.clone(),
            starting_snake.clone(),
            self.evaulator.clone(),
        );
        tree.get_best_move_with_policy()
    }
}

struct BasicAgent {
    starting_snake_id: String,
    config: MonteCarloConfig,
}

impl BasicAgent {
    pub fn new(snake: String) -> Self {
        Self {
            starting_snake_id: snake.clone(),
            config: MonteCarloConfig::default(),
        }
    }
}

impl Agent for BasicAgent {
    fn id(&self) -> &str {
        &self.starting_snake_id
    }

    fn get_best_move_with_policy(
        &self,
        board: Board,
    ) -> ((i32, i32), Vec<MovePolicy>) {
        let starting_snake = board.get_snake(&self.starting_snake_id);
        let mut tree = Tree::new(
            self.config.clone(),
            board.clone(),
            starting_snake.clone(),
        );
        tree.get_best_move_with_policy()
    }
}

trait Agent: Sync + Send {
    fn id(&self) -> &str;
    fn get_best_move_with_policy(
        &self,
        board: Board,
    ) -> ((i32, i32), Vec<MovePolicy>);
}

#[derive(Deserialize, Serialize, Debug, Clone, Hash, PartialEq, Eq)]
pub enum RunMode {
    // Runs with training between self plays.
    //
    // Default mode.
    Train,
    // Runs the neural net montecarlo against a pure monte carlo.
    BenchMark,
}

impl std::str::FromStr for RunMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "train" => Ok(RunMode::Train),
            "bench_mark" => Ok(RunMode::BenchMark),
            _ => Err(format!("Invalid RunMode: {}", s)),
        }
    }
}

pub struct TrainerConfig {
    // How many games to play in a batch.
    pub batch_size: u64,
    // How many batches to run.
    pub batches: u64,
    // The mode to run the trainer in.
    //
    // This determines whether we train between batches,
    // only generate move data, or use saved data from a previous run.
    pub run_mode: RunMode,
}

// Trains a given model through self play.
pub struct Trainer {
    // The configured agents.
    // Will eventually hold a shared reference to the model being trained.
    agents: Vec<Box<dyn Agent + Sync + Send>>,

    // Stores the moves to be trained on between batches.
    move_logger: MoveLogger,

    // Configuration for the trainer.
    config: TrainerConfig,

    // The initial board state that all games will start from.
    init_board: Board,

    // The network to train.
    //
    // This should be an interface so we can train different models.
    model: MultiOutputModel,

    // Optimiser to be used throughout the entire RL training process.
    optimiser: AdamW,

    // The shared nn evaluator used by multiple agents.
    evaluator: Arc<Mutex<NNEvaulator>>,
}

pub struct GameRun {}

impl Trainer {
    pub fn new(
        board: &Board,
        config: TrainerConfig,
    ) -> Result<Self, candle_core::error::Error> {
        let mut agent: Vec<Box<dyn Agent + Sync + Send>> = vec![];

        let evaulator = Arc::new(Mutex::new(NNEvaulator::new().unwrap()));

        if config.run_mode == RunMode::BenchMark {
            for (i, snake) in board.snakes.iter().enumerate() {
                // Make the first agent use the neural net.
                if i == 0 {
                    println!(
                        "Snake with id {} using neural network",
                        snake.id.clone()
                    );
                    agent.push(Box::from(NNAgent::new(
                        snake.id.clone(),
                        evaulator.clone(),
                    )));
                } else {
                    agent.push(Box::from(BasicAgent::new(snake.id.clone())));
                }
            }
        } else {
            // When training run with a full neural network for self play on both sides.
            for snake in &board.snakes {
                agent.push(Box::from(NNAgent::new(
                    snake.id.clone(),
                    evaulator.clone(),
                )));
            }
        }

        let training_model = MultiOutputModel::new()?;
        let optimiser = training_model.get_optimizer()?;

        let weight_path =
            std::path::Path::new("./data/models/basic.safetensor");
        if weight_path.exists() {
            evaulator
                .lock()
                .unwrap()
                .load_weights(weight_path.to_str().unwrap())
                .unwrap();
        }

        Ok(Self {
            agents: agent,
            move_logger: MoveLogger::new(),
            config: config,
            init_board: board.clone(),
            // We should clean up these unwraps.
            model: training_model,
            optimiser: optimiser,
            evaluator: evaulator,
        })
    }

    fn train(&mut self) -> Result<(), candle_core::error::Error> {
        let mut buffer_view = self.move_logger.get_move_view();
        self.model.train(&mut buffer_view, &mut self.optimiser)?;
        println!("Training run finished");
        self.model.save_weights("./data/models/basic.safetensor")?;

        // After a full training run - reload all of the NN evaluators off of the saved weights.
        self.evaluator
            .lock()
            .unwrap()
            .load_weights("./data/models/basic.safetensor")
    }

    fn play_game(&self, move_logger: &mut MoveLogger) {
        let mut board = self.init_board.clone();
        let mut state = board.get_endstate();
        while !state.is_terminal() {
            for i in 0..self.agents.len() {
                let agent = &self.agents[i];
                let is_last = i == self.agents.len() - 1;
                // Store the board before the move.
                let prior_board = board.clone();
                let move_p = agent.get_best_move_with_policy(board.clone());
                state = board.execute(agent.id(), move_p.0, is_last);
                // Log the move that was made on the board.
                move_logger.log_move(agent.id(), &prior_board, move_p.1);

                // Break immediately if the game ended after this move.
                if state.is_terminal() {
                    break;
                }

                // Every 20 moves or so throw a food in.
                if random::<u32>() % 20 == 0 {
                    board.add_food();
                }
            }
        }
        match state {
            crate::board::EndState::Winner(winner) => {
                move_logger.log_win(&winner);
            }
            crate::board::EndState::Playing => {
                panic!("Ended in non-terminal state")
            }
            crate::board::EndState::Tie => {
                println!("Ended in a tie");
                move_logger.log_tie()
            }
        }
    }

    pub fn play_batch(&mut self) {
        // Run the entire batch in parallel.
        let (sender, reciever) = channel();
        (0..self.config.batch_size).into_par_iter().for_each_with(
            sender,
            |s, _| {
                let mut move_logger = MoveLogger::new();
                self.play_game(&mut move_logger);
                let _ = s.send(move_logger);
            },
        );
        for logs in reciever.into_iter() {
            self.move_logger.merge(logs);
        }
        println!(
            "Finished self play with {} total games played",
            self.move_logger.games_played
        );
    }

    pub fn online_train(&mut self) {
        for _ in 0..self.config.batches {
            self.play_batch();
            // Train on the move data collected during the batch.
            let _ = self.train();
        }
    }

    pub fn bench_mark(&mut self) {
        for _ in 0..self.config.batches {
            self.play_batch();
        }
        self.move_logger.print_stats();
    }

    pub fn run(&mut self) {
        match self.config.run_mode {
            RunMode::Train => self.online_train(),
            RunMode::BenchMark => self.bench_mark(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::montecarlo::evaulator::MovePolicy;
    use crate::test_utils::scenarios::get_board;
    use crate::utils;

    #[test]
    fn test_move_log_policy_prior() {
        let move_log = MoveLog {
            player: "test_snake".to_string(),
            board: get_board().board,
            winner: None,
            policy: vec![
                MovePolicy {
                    dir: utils::RIGHT,
                    p: 0.4,
                },
                MovePolicy {
                    dir: utils::LEFT,
                    p: 0.3,
                },
                MovePolicy {
                    dir: utils::UP,
                    p: 0.2,
                },
                MovePolicy {
                    dir: utils::DOWN,
                    p: 0.1,
                },
            ],
        };

        assert_eq!(move_log.policy_prior(Dir::RIGHT), 0.4);
        assert_eq!(move_log.policy_prior(Dir::LEFT), 0.3);
        assert_eq!(move_log.policy_prior(Dir::UP), 0.2);
        assert_eq!(move_log.policy_prior(Dir::DOWN), 0.1);
    }

    #[test]
    fn test_move_log_get_winner_index() {
        let board = get_board().board;
        let snake_id = board.snakes[0].id.clone();

        let move_log = MoveLog {
            player: snake_id.clone(),
            board: board.clone(),
            winner: Some(snake_id.clone()),
            policy: vec![],
        };

        let winner_index = move_log.get_winner_index();
        assert!(winner_index.is_some());
        assert_eq!(winner_index.unwrap(), 0);
    }

    #[test]
    fn test_move_log_get_winner_index_none() {
        let board = get_board().board;

        let move_log = MoveLog {
            player: "test".to_string(),
            board: board,
            winner: None,
            policy: vec![],
        };

        assert!(move_log.get_winner_index().is_none());
    }

    #[test]
    fn test_move_logger_new() {
        let logger = MoveLogger::new();
        assert_eq!(logger.current_game.len(), 0);
        assert_eq!(logger.games_played, 0.0);
        assert_eq!(logger.buffer.len(), 0);
    }

    #[test]
    fn test_move_logger_log_move() {
        let mut logger = MoveLogger::new();
        let board = get_board().board;
        let policy = vec![MovePolicy {
            dir: utils::RIGHT,
            p: 1.0,
        }];

        logger.log_move("test_snake", &board, policy.clone());

        assert_eq!(logger.current_game.len(), 1);
        assert_eq!(logger.current_game[0].player, "test_snake");
        assert_eq!(logger.current_game[0].policy.len(), 1);
        assert_eq!(logger.current_game[0].winner, None);
    }

    #[test]
    fn test_move_logger_log_win() {
        let mut logger = MoveLogger::new();
        let board = get_board().board;
        let policy = vec![MovePolicy {
            dir: utils::RIGHT,
            p: 1.0,
        }];

        logger.log_move("snake1", &board, policy.clone());
        logger.log_move("snake2", &board, policy.clone());
        logger.log_win("snake1");

        // All moves should have winner set
        assert_eq!(logger.current_game.len(), 0); // Should be cleared after log_win
        assert_eq!(logger.games_played, 1.0);
        assert_eq!(logger.buffer.len(), 2);
        assert_eq!(logger.winners.get("snake1"), Some(&1.0));
    }

    #[test]
    fn test_move_logger_log_tie() {
        let mut logger = MoveLogger::new();
        let board = get_board().board;
        let policy = vec![MovePolicy {
            dir: utils::RIGHT,
            p: 1.0,
        }];

        logger.log_move("snake1", &board, policy.clone());
        logger.log_tie();

        // Current game should be cleared
        assert_eq!(logger.current_game.len(), 0);
        assert_eq!(logger.games_played, 1.0);
        assert_eq!(logger.buffer.len(), 0); // Ties don't add to buffer
    }

    #[test]
    fn test_move_logger_merge() {
        let mut logger1 = MoveLogger::new();
        let mut logger2 = MoveLogger::new();
        let board = get_board().board;
        let policy = vec![MovePolicy {
            dir: utils::RIGHT,
            p: 1.0,
        }];

        logger1.log_move("snake1", &board, policy.clone());
        logger1.log_win("snake1");

        logger2.log_move("snake2", &board, policy.clone());
        logger2.log_win("snake2");

        logger1.merge(logger2);

        assert_eq!(logger1.games_played, 2.0);
        assert_eq!(logger1.buffer.len(), 2);
        assert_eq!(logger1.winners.get("snake1"), Some(&1.0));
        assert_eq!(logger1.winners.get("snake2"), Some(&1.0));
    }

    #[test]
    #[should_panic(expected = "Should not be merging with a game in progress")]
    fn test_move_logger_merge_panics_with_game_in_progress() {
        let mut logger1 = MoveLogger::new();
        let mut logger2 = MoveLogger::new();
        let board = get_board().board;
        let policy = vec![MovePolicy {
            dir: utils::RIGHT,
            p: 1.0,
        }];

        logger1.log_move("snake1", &board, policy.clone());
        logger2.log_move("snake2", &board, policy.clone());
        logger2.log_win("snake2");

        // logger1 has a game in progress, should panic
        logger1.merge(logger2);
    }

    #[test]
    fn test_run_mode_from_str() {
        assert_eq!("train".parse::<RunMode>().unwrap(), RunMode::Train);
        assert_eq!(
            "bench_mark".parse::<RunMode>().unwrap(),
            RunMode::BenchMark
        );
        assert!("invalid".parse::<RunMode>().is_err());
    }

    #[test]
    fn test_agent_id() {
        let board = get_board().board;
        let snake_id = board.snakes[0].id.clone();
        let evaluator = Arc::new(Mutex::new(
            crate::montecarlo::evaulator::RREvaulator::new(std::rc::Rc::new(
                crate::montecarlo::tree::SnakeTracker::new(&board),
            )),
        ));

        let nn_agent = NNAgent::new(snake_id.clone(), evaluator);
        assert_eq!(nn_agent.id(), snake_id);

        let basic_agent = BasicAgent::new(snake_id.clone());
        assert_eq!(basic_agent.id(), snake_id);
    }

    // Critical test: Verify that board states are logged BEFORE the move is executed
    // This tests the fix where we log prior_board instead of board after execute()
    #[test]
    fn test_board_state_logged_before_move() {
        let mut logger = MoveLogger::new();
        let mut board = get_board().board;

        // Get initial snake position
        let snake_id = board.snakes[0].id.clone();
        let initial_head = board.get_snake(&snake_id).head.clone();

        // Store board state before move
        let prior_board = board.clone();

        // Create a simple policy
        let policy = vec![MovePolicy {
            dir: utils::RIGHT,
            p: 1.0,
        }];

        // Execute a move (this modifies the board)
        let is_last = false;
        board.execute(&snake_id, utils::RIGHT, is_last);

        // Log with the prior board (before the move)
        logger.log_move(&snake_id, &prior_board, policy);

        // Verify the logged board state matches the state BEFORE the move
        let logged_move = &logger.current_game[0];
        let logged_snake = logged_move.board.get_snake(&snake_id);
        assert_eq!(logged_snake.head, initial_head);

        // Verify the current board state is different (after the move)
        let current_snake = board.get_snake(&snake_id);
        assert_ne!(current_snake.head, initial_head);
    }

    // Test that policy values correspond to the board state they were calculated on
    #[test]
    fn test_policy_corresponds_to_logged_board_state() {
        let mut logger = MoveLogger::new();
        let mut board = get_board().board;
        let snake_id = board.snakes[0].id.clone();

        // Store board before move
        let prior_board = board.clone();

        // Get valid moves for the prior board state
        let valid_moves = prior_board.get_valid_moves(&snake_id);
        assert!(!valid_moves.is_empty());

        // Create a policy that matches the valid moves
        let policy: Vec<MovePolicy> = valid_moves
            .iter()
            .map(|dir| MovePolicy {
                dir: *dir,
                p: 1.0 / valid_moves.len() as f64,
            })
            .collect();

        // Execute move
        if !valid_moves.is_empty() {
            board.execute(&snake_id, valid_moves[0], false);
        }

        // Log with prior board and policy
        logger.log_move(&snake_id, &prior_board, policy.clone());

        // Verify the logged board state has the same valid moves as the policy
        let logged_move = &logger.current_game[0];
        let logged_valid_moves = logged_move.board.get_valid_moves(&snake_id);

        // All policy directions should be valid moves for the logged board state
        for policy_move in &policy {
            assert!(
                logged_valid_moves.contains(&policy_move.dir),
                "Policy move {:?} should be valid for logged board state",
                policy_move.dir
            );
        }
    }

    #[test]
    fn test_move_logger_buffer_size_limit() {
        let mut logger = MoveLogger::new();
        let board = get_board().board;
        let policy = vec![MovePolicy {
            dir: utils::RIGHT,
            p: 1.0,
        }];

        // Add more moves than RB_SIZE
        for _ in 0..(RB_SIZE + 100) {
            logger.log_move("snake1", &board, policy.clone());
            logger.log_win("snake1");
        }

        // Buffer should be capped at RB_SIZE
        assert_eq!(logger.buffer.len(), RB_SIZE);
    }
}
