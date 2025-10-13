use std::{any::Any, usize};

use crate::{
    config::MonteCarloConfig,
    models::{Battlesnake, Board},
    montecarlo::{nn_evaluator::SimpleConv, tree::Tree},
};

#[derive(Debug, Clone, Hash)]
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
}

impl MoveLogger {
    pub fn new() -> MoveLogger {
        Self {
            current_batch: vec![],
            current_game: vec![],
        }
    }

    pub fn log_move(&mut self, player: &str, board: &Board) {
        self.current_game.push(MoveLog {
            player: player.to_string(),
            board: board.clone(),
            winner: "".to_string(),
        });
    }

    pub fn log_win(&mut self, winner: &str) {
        for mv in &mut self.current_game {
            mv.winner = winner.to_string();
            self.current_batch.push(mv.clone());
        }
        self.current_game.clear();
    }

    pub fn clear_all(&mut self) {
        self.current_batch.clear();
        self.current_game.clear();
    }
}

struct Agent {
    snake: Battlesnake,
    config: MonteCarloConfig,
}

impl Agent {
    pub fn new(snake: &Battlesnake) -> Self {
        Self {
            snake: snake.clone(),
            config: MonteCarloConfig::default(),
        }
    }
    pub fn get_move(&self, board: &Board) -> (i32, i32) {
        let mut tree =
            Tree::new(self.config.clone(), board.clone(), self.snake.clone());
        tree.get_best_move()
    }

    pub fn id(&self) -> &str {
        return &self.snake.id;
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
            agent.push(Agent::new(snake));
        }
        Self {
            agents: agent,
            move_logger: MoveLogger::new(),
            batch_size: 10,
            batches: 1,
            init_board: board.clone(),

            // We should clean up these unwraps.
            model: SimpleConv::new().unwrap(),
        }
    }

    fn train(&mut self) {
        self.model.train(&self.move_logger.current_batch).unwrap();
    }

    pub fn play_game(&mut self) {
        let mut board = self.init_board.clone();
        let mut state = board.get_endstate();
        while !state.is_terminal() {
            for i in 0..self.agents.len() {
                let agent = &self.agents[i];
                let is_last = i == self.agents.len() - 1;
                let dir = agent.get_move(&board);
                state = board.execute(agent.id(), dir, is_last);

                // Log the player made the move.
                self.move_logger.log_move(agent.id(), &board);
            }
        }
        match state {
            crate::board::EndState::Winner(winner) => {
                self.move_logger.log_win(&winner);
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
        // These games could all be potentially parralelized.
        let batch_size = self.batch_size;
        for i in 0..self.batch_size {
            self.play_game();
            println!("Finished {i} out of {batch_size} games")
        }
    }

    pub fn run(&mut self) {
        for _ in 0..self.batches {
            self.play_batch();
            // Train on the move data collected during the batch.
            self.train();
            // Clear all moves between batches.
            self.move_logger.clear_all();
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::test_utils::scenarios::{get_scenario, TEST_REQUEST};

    // Verifies basic behavior about the snake.
    #[test]
    fn test_training_run() {
        let game_state = get_scenario(TEST_REQUEST);
        let mut trainer = Trainer::new(&game_state.board);
        trainer.run();
    }
}
