use rand::seq::SliceRandom;
use std::{
    collections::HashMap,
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{
    config::MonteCarloConfig,
    models::{Battlesnake, Board},
    montecarlo::evaulator::RREvaulator,
    utils::{self},
};

use super::node_state::NodeState;

pub type Dir = (i32, i32);

pub struct Tree {
    root: NodeState,
    max_duration: u64,
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

    pub fn get_prev_snake(&self, current_snake: &str) -> &str {
        let current_index = self.snake_map[current_snake];
        let prev_index =
            (current_index + self.snake_vec.len() - 1) % self.snake_vec.len();
        return &self.snake_vec[prev_index];
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
        // This can now be potentially swapped out with a neural network evaluator.
        let evaluator = Rc::from(RREvaulator::new(snake_tracker.clone()));
        return Self {
            max_duration: config.max_duration,
            root: NodeState::new(
                starting_board,
                starting_snake_id.clone(),
                snake_tracker,
                evaluator,
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

    pub fn get_root_scores(&self) -> Vec<(Dir, i32)> {
        let mut dirs = vec![];
        for child in &self.root.children {
            dirs.push((child.taken_dir, child.sims))
        }
        return dirs;
    }

    pub fn get_best_move(&mut self) -> (i32, i32) {
        return self.get_best_move_with_start_time(Instant::now());
    }

    pub fn get_best_move_with_start_time(
        &mut self,
        start: Instant,
    ) -> (i32, i32) {
        let max_duration = Duration::from_millis(self.max_duration);
        self.root.expand();
        loop {
            self.expand_tree();
            let elasped_time = start.elapsed();
            if elasped_time >= max_duration {
                break;
            }
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

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        test_utils::scenarios::{
            get_board, get_scenario, AVOID_DEATH_ADVANCED,
            AVOID_DEATH_GET_FOOD, MULTI_SNAKE,
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
