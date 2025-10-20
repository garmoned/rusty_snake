use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::channel;
use std::{collections::HashMap, fs};

use crate::{
    config::MonteCarloConfig,
    models::Board,
    montecarlo::{nn_evaluator::SimpleConv, tree::Tree},
};

#[derive(Deserialize, Serialize, Debug, Clone, Hash)]
pub struct MoveLog {
    // The player who just made the move.
    pub player: String,
    // The resulting board state.
    pub board: Board,
    // The player who won from this board state.
    pub winner: String,
}

impl MoveLog {
    // Returns a one hot encoded winner based
    // on the order of the snakes in the board state.
    pub fn get_winner_index(&self) -> u32 {
        for (idx, bs) in self.board.snakes.iter().enumerate() {
            if bs.id == self.winner {
                return idx as u32;
            }
        }
        return 0;
    }
}

pub enum ReadMode {
    BATCH(usize),
    ALL,
}

pub struct DataLoader {
    // Where to read and write from on disk.
    mem_path: String,
    readmode: ReadMode,
}

struct FileIterator {
    // The number of the current batch.
    batch_number: u64,
    // The index of the current move log in the batch.
    batch_index: u64,
    mem_path: String,
}

// Iterates over only files from a specific batch.
struct BatchFileIterator {
    batch_number: usize,
    batch_index: u64,
    mem_path: String,
}

impl BatchFileIterator {
    fn new(mem_path: String, batch: usize) -> Self {
        Self {
            batch_number: batch,
            batch_index: 0,
            mem_path,
        }
    }

    fn cur_file_path(&self) -> String {
        format!(
            "{}/batch_{}_{}.json",
            self.mem_path, self.batch_number, self.batch_index
        )
    }
}

impl Iterator for BatchFileIterator {
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        if std::fs::exists(self.cur_file_path()).unwrap() {
            self.batch_index += 1;
            return Some(self.cur_file_path());
        }
        return None;
    }
}

impl FileIterator {
    fn new(mem_path: String) -> Self {
        Self {
            mem_path: mem_path,
            batch_index: 0,
            batch_number: 0,
        }
    }
    fn cur_file_path(&self) -> String {
        format!(
            "{}/batch_{}_{}.json",
            self.mem_path, self.batch_number, self.batch_index
        )
    }
}

impl Iterator for FileIterator {
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        if std::fs::exists(self.cur_file_path()).unwrap() {
            self.batch_index += 1;
            return Some(self.cur_file_path());
        }
        // Check if we are at the next batch entirely.
        self.batch_index = 0;
        self.batch_number += 1;
        if std::fs::exists(self.cur_file_path()).unwrap() {
            self.batch_index += 1;
            return Some(self.cur_file_path());
        }
        return None;
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
                serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap()
            })
            .collect();
        if chunk.len() > 0 {
            return Some(chunk);
        }
        None
    }
}

impl DataLoader {
    fn new(mem_path: String) -> Self {
        Self {
            readmode: ReadMode::ALL,
            mem_path,
        }
    }

    pub fn set_readmode(&mut self, mode: ReadMode) {
        self.readmode = mode
    }

    // Reads all data in chunks.
    pub fn read_in_chunks(&self, chunk_size: usize) -> DataIterator {
        match self.readmode {
            ReadMode::BATCH(batch) => DataIterator {
                file_iterator: Box::from(BatchFileIterator::new(
                    self.mem_path.clone(),
                    batch,
                )),
                chunk_size,
            },
            ReadMode::ALL => DataIterator {
                file_iterator: Box::from(FileIterator::new(
                    self.mem_path.clone(),
                )),
                chunk_size,
            },
        }
    }

    pub fn len(&self) -> usize {
        let mut file_count = 0;
        for entry in fs::read_dir(&self.mem_path).unwrap() {
            let entry = entry.unwrap();
            if entry.metadata().unwrap().is_file() {
                file_count += 1;
            }
        }
        return file_count;
    }

    fn write(
        &self,
        move_log: &MoveLog,
        batch_number: usize,
        batch_index: usize,
    ) {
        let file_path =
            format!("./data/moves/batch_{}_{}.json", batch_number, batch_index);
        let json = serde_json::to_string(&move_log).unwrap();
        fs::write(file_path, json).unwrap();
    }
}

struct MoveLogger {
    current_batch: Vec<MoveLog>,
    current_game: Vec<MoveLog>,
    batch_number: usize,
    winners: HashMap<String, f32>,
    games_played: f32,
    data_loader: DataLoader,
}

impl MoveLogger {
    pub fn new() -> MoveLogger {
        Self {
            current_batch: vec![],
            current_game: vec![],
            batch_number: 0,
            winners: HashMap::new(),
            games_played: 0.0,
            data_loader: DataLoader::new("./data/moves/".to_string()),
        }
    }

    // Used to merge two move loggers together after.
    // Used to merge batches together after running lots of simulations
    pub fn merge(&mut self, other: &MoveLogger) {
        self.current_batch.extend(other.current_batch.clone());
        self.current_game.extend(other.current_game.clone());
        self.winners.extend(other.winners.clone());
        self.games_played += other.games_played;

        for (winner, times) in &other.winners {
            let times = times + self.winners.get(winner).unwrap_or(&0.0);
            self.winners.insert(winner.to_owned(), times);
        }
    }

    pub fn log_move(&mut self, player: &str, board: &Board) {
        self.current_game.push(MoveLog {
            player: player.to_string(),
            board: board.clone(),
            winner: "".to_string(),
        });
    }

    pub fn dump_batch(&mut self) {
        for (n, ml) in self.current_batch.iter().enumerate() {
            self.data_loader.write(ml, self.batch_number, n);
        }
        self.batch_number += 1;
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
}

impl std::str::FromStr for RunMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "train" => Ok(RunMode::Train),
            "dry_run" => Ok(RunMode::DryRun),
            "offline" => Ok(RunMode::Offline),
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
    model: SimpleConv,
}

impl Trainer {
    pub fn new(board: &Board, config: TrainerConfig) -> Self {
        let mut agent = vec![];
        for snake in &board.snakes {
            agent.push(Agent::new(snake.id.clone()));
        }
        Self {
            agents: agent,
            move_logger: MoveLogger::new(),
            config: config,
            init_board: board.clone(),
            // We should clean up these unwraps.
            model: SimpleConv::new().unwrap(),
        }
    }

    fn train(&mut self) {
        if self.config.run_mode == RunMode::DryRun {
            println!("Skipping model training, in dryrun mode");
        }
        self.model.train(&self.move_logger.data_loader).unwrap();
        // Clear out all of the data after training.
        println!("Training run finished");
        let r = self.model.save("./data/models/basic.safetensor");
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
                let dir = agent.get_move(board.clone());
                state = board.execute(agent.id(), dir, is_last);
                // Log the player made the move.
                move_logger.log_move(agent.id(), &board);
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
                // Nothing happens on a tie.
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
        self.move_logger.dump_batch();
    }

    pub fn run(&mut self) {
        for _ in 0..self.config.batches {
            self.play_batch();
            // Train on the move data collected during the batch.
            self.train();
        }
        self.move_logger.print_stats();
    }
}
