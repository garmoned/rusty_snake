use rand::random;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::channel;
use std::usize;
use std::{collections::HashMap, fs};

use crate::config::Evaluator;
use crate::montecarlo::evaulator::MovePolicy;
use crate::utils;
use crate::{
    config::MonteCarloConfig,
    models::Board,
    montecarlo::{nn_evaluator::MultiOutputModel, tree::Tree},
};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MoveLog {
    // The player who just made the move.
    pub player: String,
    // The resulting board state.
    pub board: Board,
    // The player who won from this board state.
    pub winner: String,
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
        for (idx, bs) in self.board.snakes.iter().enumerate() {
            if bs.id == self.winner {
                return Some(idx as u32);
            }
        }
        // If the winner is not found, return None.
        return None;
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

struct MoveLogger {
    current_batch: Vec<MoveLog>,
    current_game: Vec<MoveLog>,
    batch_number: usize,
    winners: HashMap<String, f32>,
    games_played: f32,
}

impl MoveLogger {
    pub fn new() -> MoveLogger {
        Self {
            current_batch: vec![],
            current_game: vec![],
            batch_number: 0,
            winners: HashMap::new(),
            games_played: 0.0,
        }
    }

    // Used to merge two move loggers together after.
    // Used to merge batches together after running lots of simulations
    pub fn merge(&mut self, other: &MoveLogger) {
        self.current_batch.extend(other.current_batch.clone());
        self.current_game.extend(other.current_game.clone());
        self.games_played += other.games_played;

        for (winner, times) in &other.winners {
            let times = times + self.winners.get(winner).unwrap_or(&0.0);
            self.winners.insert(winner.to_owned(), times);
        }
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
            winner: "".to_string(),
            policy: policy,
        });
    }

    pub fn log_win(&mut self, winner: &str) {
        for mv in &mut self.current_game {
            mv.winner = winner.to_string();
            self.current_batch.push(mv.clone());
        }
        self.current_game.clear();
        let init = self.winners.get(winner).unwrap_or(&0.0) + 1.0;
        self.winners.insert(winner.to_owned(), init);
        self.games_played += 1.0;
    }

    pub fn finish_game(&mut self) {
        for mv in &mut self.current_game {
            self.current_batch.push(mv.clone());
        }
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

    pub fn clear_all(&mut self) {
        self.current_batch.clear();
        self.current_game.clear();
    }
}

struct Agent {
    starting_snake_id: String,
    config: MonteCarloConfig,
}

impl Agent {
    pub fn new_nn(snake: String) -> Self {
        let mut config = MonteCarloConfig::default();
        config.evaulator = Evaluator::NEURAL;
        Self {
            starting_snake_id: snake.clone(),
            config: config,
        }
    }
    pub fn new(snake: String) -> Self {
        Self {
            starting_snake_id: snake.clone(),
            config: MonteCarloConfig::default(),
        }
    }
    pub fn get_move(&self, board: Board) -> (i32, i32) {
        let starting_snake = board.get_snake(&self.starting_snake_id);
        let mut tree = Tree::new(
            self.config.clone(),
            board.clone(),
            starting_snake.clone(),
        );
        tree.get_best_move()
    }

    pub fn get_best_move_with_policy(
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

    pub fn id(&self) -> &str {
        return &self.starting_snake_id;
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Hash, PartialEq, Eq)]
pub enum RunMode {
    // Runs with training between self plays.
    //
    // Default mode.
    Train,
    // Runs without training between batches.
    // Only generates move data.
    DryRun,
    // Runs using the saved data from a previous run.
    Offline,

    // Runs the neural net montecarlo against a pure monte carlo.
    BenchMark,
}

impl std::str::FromStr for RunMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "train" => Ok(RunMode::Train),
            "dry_run" => Ok(RunMode::DryRun),
            "offline" => Ok(RunMode::Offline),
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
    agents: Vec<Agent>,

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
}

impl Trainer {
    pub fn new(board: &Board, config: TrainerConfig) -> Self {
        let mut agent = vec![];

        if config.run_mode == RunMode::BenchMark {
            for (i, snake) in board.snakes.iter().enumerate() {
                // Make the first agent use the neural net.
                if i == 0 {
                    println!(
                        "Snake with id {} using neural network",
                        snake.id.clone()
                    );
                    agent.push(Agent::new_nn(snake.id.clone()));
                } else {
                    agent.push(Agent::new(snake.id.clone()));
                }
            }
        } else {
            // When training run with a full neural network for self play on both sides.
            for snake in &board.snakes {
                agent.push(Agent::new_nn(snake.id.clone()));
            }
        }

        Self {
            agents: agent,
            move_logger: MoveLogger::new(),
            config: config,
            init_board: board.clone(),
            // We should clean up these unwraps.
            model: MultiOutputModel::new().unwrap(),
        }
    }

    fn train(&mut self) {
        if self.config.run_mode == RunMode::DryRun
            || self.config.run_mode == RunMode::BenchMark
        {
            println!("Skipping model training, in dryrun mode");
        }
        self.model.train(&self.move_logger.current_batch).unwrap();
        println!("Training run finished");
        let r = self.model.save_weights("./data/models/basic.safetensor");
        match r {
            Ok(_) => println!("Model saved successfully"),
            Err(err) => {
                println!("Model failed to save: {:?}", err);
            }
        };
        self.move_logger.clear_all();
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
                move_logger.finish_game();
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
                s.send(move_logger).unwrap();
            },
        );
        for logs in reciever.into_iter() {
            self.move_logger.merge(&logs);
        }
        println!(
            "Finished playing batch {} with {} games played",
            self.move_logger.batch_number, self.move_logger.games_played
        );
    }

    pub fn online_train(&mut self) {
        for _ in 0..self.config.batches {
            self.play_batch();
            // Train on the move data collected during the batch.
            self.train();
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
            RunMode::DryRun => self.online_train(),
            RunMode::Offline => self.train(),
            RunMode::BenchMark => self.bench_mark(),
        }
    }
}
