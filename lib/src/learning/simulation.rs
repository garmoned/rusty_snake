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
    // The resulting board state.
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
        return 0.0;
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
        if chunk.len() > 0 {
            return Some(chunk);
        }
        None
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
        return &self.starting_snake_id;
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
        return &self.starting_snake_id;
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
                let move_p = agent.get_best_move_with_policy(board.clone());
                state = board.execute(agent.id(), move_p.0, is_last);
                // Log the player made the move.
                move_logger.log_move(agent.id(), &board, move_p.1);
                // Every 10 moves or so throw a food in.
                if random::<i32>() % 10 == 0 {
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
