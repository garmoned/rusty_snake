use rand::seq::SliceRandom;
use std::{collections::HashMap, rc::Rc};

use crate::{
    config::MonteCarloConfig,
    models::{Battlesnake, Board},
    utils::{self},
};

type Dir = (i32, i32);

#[derive(Clone)]
struct NodeState {
    // Back pointer to parent. Necessary evil.
    parent: Option<*mut NodeState>,
    children: Vec<NodeState>,
    board_state: Board,

    // The snake who is about to make a move.
    current_snake: String,

    // The snake who just acted.
    snake_who_moved: String,

    // The direction just moved in.
    taken_dir: Dir,

    sims: i32,
    wins: i32,

    // Shared ownership by the nodes.
    // Too lazy to do ownership stuff for just a helper object.
    snake_tracker: Rc<SnakeTracker>,
}

pub struct Tree {
    root: NodeState,
    iterations: i64,
}

pub struct SnakeTracker {
    snake_map: HashMap<String, usize>,
    snake_vec: Vec<String>,
}

impl SnakeTracker {
    pub fn new(board: &Board) -> Self {
        let mut snake_vec = vec![];
        let mut snake_map = HashMap::new();
        for (i, snake) in board.snakes.iter().enumerate() {
            let copy_snake = &snake.id;
            snake_vec.push(copy_snake.clone());
            snake_map.insert(copy_snake.clone(), i);
        }
        Self {
            snake_map,
            snake_vec,
        }
    }

    pub fn get_next_snake(&self, current_snake: &str) -> &str {
        let next_index = self.snake_map[current_snake] + 1;
        return &self.snake_vec[next_index % self.snake_vec.len()];
    }
}

impl Tree {
    pub fn new(
        config: MonteCarloConfig,
        mut starting_board: Board,
        starting_snake: Battlesnake,
    ) -> Self {
        let starting_snake_id = starting_snake.id.clone();
        utils::fix_snake_order(&mut starting_board, starting_snake);
        let snake_tracker = Rc::from(SnakeTracker::new(&starting_board));
        return Self {
            iterations: config.iterations,
            root: NodeState::new(
                starting_board,
                starting_snake_id.clone(),
                snake_tracker,
            ),
        };
    }

    fn expand_tree(&mut self) {
        let promising_node = self.root.select_node();
        promising_node.expand();
        if promising_node.children.len() > 0 {
            promising_node
                .children
                .choose_mut(&mut rand::thread_rng())
                .unwrap()
                .play_out();
            return;
        }
        promising_node.play_out();
    }

    pub fn get_best_move(&mut self) -> (i32, i32) {
        self.root.expand();
        for _ in 0..self.iterations {
            self.expand_tree();
        }
        let best_child =
            self.root.children.iter().max_by(|x, y| x.sims.cmp(&y.sims));
        if best_child.is_none() {
            return (1, 0);
        }
        let best_child = best_child.unwrap();
        return best_child.taken_dir;
    }
}

impl NodeState {
    const C: f64 = 1.141;

    pub fn new(
        board_state: Board,
        current_snake: String,
        snake_tracker: Rc<SnakeTracker>,
    ) -> Self {
        NodeState {
            taken_dir: (1, 0),
            current_snake,
            sims: 0,
            wins: 0,
            parent: None,
            children: vec![],
            board_state,
            snake_who_moved: "none".to_string(),
            snake_tracker: snake_tracker,
        }
    }

    pub fn new_child(
        board_state: Board,
        current_snake: String,
        snake_who_moved: String,
        snake_tracker: Rc<SnakeTracker>,
        taken_dir: Dir,
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
            let snake_tracker = snake_tracker;
            let next_snake = snake_tracker.get_next_snake(&self.current_snake);
            let snake_tracker = &self.snake_tracker;
            children.push(NodeState::new_child(
                new_board,
                next_snake.to_string(),
                self.current_snake.clone(),
                snake_tracker.clone(),
                dir,
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
            crate::simulation::EndState::Winner(winner) => {
                self.back_prop(&winner)
            }
            crate::simulation::EndState::Tie => self.back_prop("tie"),
            crate::simulation::EndState::Playing => {
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
                return x
                    .utc_val(parent_sims)
                    .total_cmp(&y.utc_val(parent_sims));
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
        return reward + discover;
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        test_utils::scenarios::{
            get_board, get_scenario, AVOID_DEATH_ADVANCED,
            AVOID_DEATH_GET_FOOD, AVOID_HEAD_TO_HEAD_DEATH, AVOID_SELF_TRAP,
            DO_NOT_CIRCLE_FOOD, GET_THE_FOOD, MULTI_SNAKE,
        },
        utils::dir_to_string,
    };

    #[test]
    fn test_avoid_wall() {
        let game_state = get_board();
        let mut tree = Tree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!("up", best_move)
    }

    #[test]
    fn test_avoid_death_get_food() {
        let game_state = get_scenario(AVOID_DEATH_GET_FOOD);
        let mut tree = Tree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "right")
    }

    #[test]
    fn test_avoid_self_trap() {
        let game_state = get_scenario(AVOID_SELF_TRAP);
        let mut tree = Tree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "up")
    }
    #[test]
    fn test_get_easy_food() {
        let game_state = get_scenario(GET_THE_FOOD);
        let mut tree = Tree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_eq!(best_move, "down")
    }

    #[test]
    fn test_avoid_death_advanced() {
        let game_state = get_scenario(AVOID_DEATH_ADVANCED);
        let mut tree = Tree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "right")
    }

    #[test]
    fn test_do_not_circle_food() {
        let game_state = get_scenario(DO_NOT_CIRCLE_FOOD);
        let mut tree = Tree::new(
            MonteCarloConfig::default(),
            game_state.board.clone(),
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_eq!(best_move, "up")
    }

    #[test]
    fn test_avoid_head_to_head_death() {
        let game_state = get_scenario(AVOID_HEAD_TO_HEAD_DEATH);
        let mut tree = Tree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "right")
    }

    #[test]
    fn test_can_handle_multiplayer() {
        let game_state = get_scenario(MULTI_SNAKE);
        let mut tree = Tree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        assert_eq!(dir_to_string(tree.get_best_move()).is_empty(), false);
    }
}
