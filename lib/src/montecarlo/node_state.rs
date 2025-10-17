use super::tree::{Dir, SnakeTracker};
use crate::models::Board;
use crate::montecarlo::evaulator::Evaluator;
use crate::utils::dir_to_string;
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct NodeState {
    // Back pointer to parent. Necessary evil.
    parent: Option<*mut NodeState>,
    pub(crate) children: Vec<NodeState>,
    pub(crate) board_state: Board,

    // The snake who is about to make a move.
    pub(crate) current_snake: String,

    // The snake who just acted.
    pub(crate) snake_who_moved: String,

    // The direction just moved in.
    pub(crate) taken_dir: Dir,

    pub(crate) sims: i32,
    wins: i32,

    // Shared ownership by the nodes.
    // Too lazy to do ownership stuff for just a helper object.
    snake_tracker: Rc<SnakeTracker>,
    // Shared reference to an evaluator used by all nodes.
    evaulator: Rc<dyn Evaluator>,
}

impl NodeState {
    const C: f64 = 1.141;

    pub fn new(
        board_state: Board,
        current_snake: String,
        snake_tracker: Rc<SnakeTracker>,
        evaulator: Rc<dyn Evaluator>,
    ) -> Self {
        let snake_who_moved = snake_tracker.get_prev_snake(&current_snake);
        NodeState {
            taken_dir: (1, 0),
            current_snake,
            sims: 0,
            wins: 0,
            parent: None,
            children: vec![],
            board_state,
            snake_who_moved: snake_who_moved.to_owned(),
            snake_tracker: snake_tracker,
            evaulator: evaulator,
        }
    }

    pub fn new_child(
        board_state: Board,
        current_snake: String,
        snake_who_moved: String,
        snake_tracker: Rc<SnakeTracker>,
        taken_dir: Dir,
        evaulator: Rc<dyn Evaluator>,
    ) -> Self {
        NodeState {
            current_snake,
            sims: 0,
            wins: 0,
            taken_dir,
            snake_who_moved,
            parent: None,
            children: vec![],
            board_state,
            snake_tracker: snake_tracker,
            evaulator: evaulator,
        }
    }

    pub fn expand(&mut self) {
        if self.board_state.is_terminal() {
            return;
        }
        let mut children = vec![];
        for dir in self.board_state.get_valid_moves(&self.current_snake) {
            let mut new_board = self.board_state.clone();
            new_board.execute_dir(&self.current_snake, dir);
            let snake_tracker = self.snake_tracker.clone();
            let next_snake = snake_tracker.get_next_snake(&self.current_snake);
            let snake_tracker = &self.snake_tracker;
            children.push(NodeState::new_child(
                new_board,
                next_snake.to_string(),
                self.current_snake.clone(),
                snake_tracker.clone(),
                dir,
                self.evaulator.clone(),
            ))
        }
        for child in &mut children {
            child.set_parent(self);
        }
        self.children = children;
    }

    pub fn get_next_snake(&self, snake_id: &str) -> String {
        return self.snake_tracker.get_next_snake(snake_id).to_string();
    }

    pub fn set_parent(&mut self, parent: &mut NodeState) {
        self.parent = Some(parent as *mut NodeState)
    }

    pub fn play_out(&mut self) {
        let mut board_copy = self.board_state.clone();
        let mut end_state = board_copy.get_endstate();
        let mut current_snake = self.current_snake.clone();
        while !end_state.is_terminal() {
            board_copy.execute_random_move(&current_snake);
            end_state = board_copy.get_endstate();
            current_snake = self.get_next_snake(&current_snake);
        }
        match end_state {
            crate::board::EndState::Winner(winner) => self.back_prop(&winner),
            crate::board::EndState::Tie => self.back_prop("tie"),
            crate::board::EndState::Playing => {
                panic!("somehow the end state ended with playing")
            }
        }
    }

    pub fn back_prop(&mut self, winner: &str) {
        if self.snake_who_moved == winner {
            self.wins += 1;
        }
        self.sims += 1;
        match self.parent {
            Some(parent) => unsafe {
                parent.as_mut().unwrap().back_prop(winner)
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

    pub fn wins(&self) -> f64 {
        return self.wins as f64;
    }

    pub fn utc_val(&self, parent_sims: f64) -> f64 {
        if self.sims == 0 {
            return f64::INFINITY;
        }
        let discover =
            ((parent_sims + 1.0).ln() / self.sims()).sqrt() * NodeState::C;
        let reward = self.wins() / self.sims();
        return reward + discover + self.heuristic();
    }

    pub fn heuristic(&self) -> f64 {
        self.board_state.get_snake(&self.snake_who_moved).body.len() as f64
            / (self.sims() + 1.0)
    }
}
