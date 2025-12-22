use super::tree::{Dir, SnakeTracker};
use crate::models::Board;
use crate::montecarlo::evaulator::{Evaluator, MovePolicy};
use core::f64;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Clone)]
pub(crate) struct NodeState {
    // Back pointer to parent. Necessary evil.
    parent: Option<*mut NodeState>,
    pub(crate) children: Vec<NodeState>,
    pub(crate) board_state: Board,

    // The snake who is about to make a move.
    pub(crate) current_snake: String,

    // The direction just moved in.
    pub(crate) taken_dir: Dir,

    pub(crate) sims: i32,

    // Shared ownership by the nodes.
    // Too lazy to do ownership stuff for just a helper object.
    snake_tracker: Rc<SnakeTracker>,
    // Shared reference to an evaluator used by all nodes.
    evaulator: Arc<Mutex<dyn Evaluator>>,

    policy_pred: f64,

    // Whether this node state is always terminal.
    is_solved: bool,

    // The total accumulated value for the snake that took a move
    // to reach this board state.
    accumulated_v: f64,
}

impl NodeState {
    const C: f64 = 1.141;

    pub fn new(
        board_state: Board,
        current_snake: String,
        snake_tracker: Rc<SnakeTracker>,
        evaulator: Arc<Mutex<dyn Evaluator>>,
    ) -> Self {
        NodeState {
            taken_dir: (1, 0),
            current_snake,
            sims: 0,
            parent: None,
            children: vec![],
            board_state,
            snake_tracker: snake_tracker,
            evaulator: evaulator,
            policy_pred: 1.0,
            is_solved: false,
            accumulated_v: 0.0,
        }
    }

    pub fn new_child(
        board_state: Board,
        current_snake: String,
        snake_tracker: Rc<SnakeTracker>,
        taken_dir: Dir,
        evaulator: Arc<Mutex<dyn Evaluator>>,
        policy_pred: f64,
    ) -> Self {
        NodeState {
            current_snake,
            sims: 0,
            taken_dir,
            parent: None,
            children: vec![],
            board_state,
            snake_tracker: snake_tracker,
            evaulator: evaulator,
            policy_pred,
            is_solved: false,
            accumulated_v: 0.0,
        }
    }

    fn policy_per_move(
        &self,
        policy_vec: &Vec<MovePolicy>,
        dir: (i32, i32),
    ) -> f64 {
        for p in policy_vec {
            if p.dir == dir {
                return p.p;
            }
        }
        return 0.0;
    }

    fn get_policy(&self) -> Vec<MovePolicy> {
        let lock = self.evaulator.lock().unwrap();
        lock.predict_best_moves(&self.board_state, &self.current_snake)
    }

    fn predict_value(&self) -> f32 {
        let lock = self.evaulator.lock().unwrap();
        let rewarded_snake =
            self.snake_tracker.get_prev_snake(&self.current_snake);
        lock.predict_value(
            &self.board_state,
            rewarded_snake,
            &self.current_snake,
        )
    }

    pub fn expand(&mut self) {
        if self.board_state.is_terminal() {
            return;
        }
        let mut children = vec![];
        let policy_prediction = self.get_policy();
        for dir in self.board_state.get_valid_moves(&self.current_snake) {
            let mut new_board = self.board_state.clone();
            new_board.execute_dir(&self.current_snake, dir);
            let snake_tracker = self.snake_tracker.clone();
            let next_snake = snake_tracker.get_next_snake(&self.current_snake);
            let snake_tracker = &self.snake_tracker;
            children.push(NodeState::new_child(
                new_board,
                next_snake.to_string(),
                snake_tracker.clone(),
                dir,
                self.evaulator.clone(),
                self.policy_per_move(&policy_prediction, dir),
            ));
        }
        for child in &mut children {
            child.set_parent(self);
        }
        self.children = children;
    }

    pub fn set_parent(&mut self, parent: &mut NodeState) {
        self.parent = Some(parent as *mut NodeState)
    }

    pub fn play_out(&mut self) {
        // Only use the evaluator if the game has not already finished.
        let rewarded_snake = self
            .snake_tracker
            .get_prev_snake(&self.current_snake)
            .to_string();
        if self.board_state.is_terminal() {
            self.is_solved = true;
            match self.board_state.get_endstate() {
                crate::board::EndState::Winner(winner) => {
                    if winner == rewarded_snake {
                        self.back_prop(&rewarded_snake, 1.0)
                    } else {
                        self.back_prop(&rewarded_snake, -1.0);
                    }
                }
                crate::board::EndState::Playing => {
                    panic!("Board state should be terminal")
                }
                crate::board::EndState::Tie => self.back_prop("tie", 0.0),
            }
        } else {
            self.back_prop(&rewarded_snake, self.predict_value() as f64);
        }
    }

    pub fn back_prop(&mut self, rewarded_snake: &str, value: f64) {
        // The win should be awarded to the player who performed the move that led to this board state.
        // Not the player whose turn it is to move since they haven't actually impacted the game state yet.
        let self_rewarded_snake =
            self.snake_tracker.get_prev_snake(&self.current_snake);
        if self_rewarded_snake == rewarded_snake {
            self.accumulated_v += value
        } else if self_rewarded_snake != "tie" {
            self.accumulated_v -= value;
        }
        self.sims += 1;
        match self.parent {
            Some(parent) => unsafe {
                parent.as_mut().unwrap().back_prop(rewarded_snake, value)
            },
            None => { /* Do nothing */ }
        }
    }

    pub fn select_node(&mut self) -> &mut NodeState {
        if self.children.is_empty() {
            return self;
        }
        let parent_sims = self.sims();
        return self
            .children
            .iter_mut()
            .max_by(|x, y| {
                x.utc_val(parent_sims).total_cmp(&y.utc_val(parent_sims))
            })
            .unwrap()
            .select_node();
    }

    pub fn sims(&self) -> f64 {
        return self.sims as f64;
    }

    pub fn utc_val(&self, parent_sims: f64) -> f64 {
        if self.sims == 0 {
            return f64::INFINITY;
        }

        // If the player who moved into this immediately lost - prune this move entirely.
        // If the playr who moved won - always choose this node while exploring, no other moves could be better.
        if self.is_solved {
            let snake_who_moved =
                self.snake_tracker.get_next_snake(&self.current_snake);
            let winner = self.board_state.get_endstate();
            return match winner {
                crate::board::EndState::Winner(winner) => {
                    if winner != snake_who_moved {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                }
                crate::board::EndState::Playing => panic!("should be terminal"),
                crate::board::EndState::Tie => f64::NEG_INFINITY,
            };
        }

        let discover = ((parent_sims).ln() / (1.0 + self.sims())).sqrt()
            * NodeState::C
            * self.policy_pred;
        let reward = (self.accumulated_v as f64) / self.sims();
        return reward + discover;
    }
}
