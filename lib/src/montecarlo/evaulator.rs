use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::{models::Board, montecarlo::tree::SnakeTracker};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MovePolicy {
    // Direction to be moved.
    pub dir: (i32, i32),
    // The probability that this move should be taken.
    pub p: f64,
}

// Evaluates who will win from a given board state.
pub trait Evaluator {
    // Given a board state and id of player about to move
    // predicts the most likely winner of the game.
    fn predict_winner(&self, board: &Board, current_snake: &str) -> String;

    // Predicts the best moves to be played by the player about to move.
    fn predict_best_moves(
        &self,
        board: &Board,
        current_snake: &str,
    ) -> Vec<MovePolicy>;
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

    fn predict_best_moves(&self, _: &Board, _: &str) -> Vec<MovePolicy> {
        // return an even distribution of moves.
        return vec![
            MovePolicy {
                dir: (1, 0),
                p: 1.0,
            },
            MovePolicy {
                dir: (0, 1),
                p: 1.0,
            },
            MovePolicy {
                dir: (-1, 0),
                p: 1.0,
            },
            MovePolicy {
                dir: (0, -1),
                p: 1.0,
            },
        ];
    }
}
