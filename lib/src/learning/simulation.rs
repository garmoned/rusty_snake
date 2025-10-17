use core::time;
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

struct MoveLogger {
    current_batch: Vec<MoveLog>,
    current_game: Vec<MoveLog>,
    batch_number: u64,

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
        self.batch_number += 1;
        for (n, ml) in self.current_batch.iter().enumerate() {
            let file_path =
                format!("./data/moves/batch{}_{}.json", self.batch_number, n);
            let json = serde_json::to_string(&ml).unwrap();
            fs::write(file_path, json).unwrap();
        }
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

// Trains a given model through self play.
pub struct Trainer {
    // The configured agents.
    // Will eventually hold a shared reference to the model being trained.
    agents: Vec<Agent>,

    // Stores the moves to be trained on between batches.
    move_logger: MoveLogger,

    // Configuration
    //
    // How many batches.
    batch_size: u64,
    // How large each batch is.
    batches: u64,

    // The initial board state that all games will start from.
    init_board: Board,

    // The network to train.
    //
    // This should be an interface so we can train different models.
    model: SimpleConv,
}

impl Trainer {
    pub fn new(board: &Board) -> Self {
        let mut agent = vec![];
        for snake in &board.snakes {
            agent.push(Agent::new(snake.id.clone()));
        }
        Self {
            agents: agent,
            move_logger: MoveLogger::new(),
            batch_size: 4,
            batches: 1,
            init_board: board.clone(),
            // We should clean up these unwraps.
            model: SimpleConv::new().unwrap(),
        }
    }

    fn train(&mut self) {
        self.model.train(&self.move_logger.current_batch).unwrap();
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
        let (sender, reciever) = channel();
        (0..self.batch_size)
            .into_par_iter()
            .for_each_with(sender, |s, _| {
                let mut move_logger = MoveLogger::new();
                self.play_game(&mut move_logger);
                s.send(move_logger).unwrap();
            });
        for logs in reciever.into_iter() {
            self.move_logger.merge(&logs);
        }
        self.move_logger.dump_batch();
    }

    pub fn run(&mut self) {
        for _ in 0..self.batches {
            self.play_batch();
            // Train on the move data collected during the batch.
            self.train();
        }
        self.move_logger.print_stats();
    }
}
