use std::rc::Rc;

use crate::{models::Board, montecarlo::tree::SnakeTracker};

// Evaluates who will win from a given board state.
pub trait Evaluator {
    // Given a board state and id of player about to move
    // predicts the most likely winner of the game.
    fn predict_winner(&self, board: &Board, current_snake: &str) -> String;
}

// An evaulator which makes random moves to decide the winner.
pub struct RREvaulator {
    // Too lazy to do ownership stuff for just a helper object.
    snake_tracker: Rc<SnakeTracker>,
}

impl RREvaulator {
    pub fn new(snake_tracker: Rc<SnakeTracker>) -> Self {
        Self {
            snake_tracker: snake_tracker.clone(),
        }
    }
}

impl Evaluator for RREvaulator {
    fn predict_winner(&self, board: &Board, current_snake: &str) -> String {
        let mut board_copy = board.clone();
        let mut end_state = board_copy.get_endstate();
        let mut current_snake = current_snake;
        while !end_state.is_terminal() {
            board_copy.execute_random_move(&current_snake);
            end_state = board_copy.get_endstate();
            current_snake = self.snake_tracker.get_next_snake(current_snake)
        }
        match end_state {
            crate::board::EndState::Winner(winner) => return winner,
            crate::board::EndState::Tie => return "Tie".to_string(),
            crate::board::EndState::Playing => {
                panic!("somehow the end state ended with playing")
            }
        }
    }
}
