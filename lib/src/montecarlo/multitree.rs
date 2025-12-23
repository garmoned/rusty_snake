use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::mpsc::channel;
use std::time::Instant;

use crate::{
    config::MonteCarloConfig,
    models::{Battlesnake, Board},
    utils,
};

use super::tree::{Dir, Tree};

pub struct Multitree {
    num_trees: usize,
    config: MonteCarloConfig,
    starting_board: Board,
    starting_snake: Battlesnake,
}

impl Multitree {
    pub fn new(
        config: MonteCarloConfig,
        starting_board: Board,
        starting_snake: Battlesnake,
    ) -> Self {
        Self {
            num_trees: utils::read_env("NUM_TREES", 10),
            config,
            starting_board,
            starting_snake,
        }
    }
    pub fn get_best_move(&mut self) -> (i32, i32) {
        let start_time = Instant::now();
        let mut dir_map = HashMap::<Dir, i32>::new();
        let (sender, reciever) = channel();
        (0..self.num_trees)
            .into_par_iter()
            .for_each_with(sender, |s, _| {
                let mut tree = Tree::new(
                    self.config.clone(),
                    self.starting_board.clone(),
                    self.starting_snake.clone(),
                );
                tree.get_best_move_with_start_time(start_time);
                s.send(tree.get_root_scores()).unwrap()
            });
        for root_results in reciever.into_iter() {
            for dir in &root_results {
                dir_map.insert(
                    dir.0,
                    dir_map.get(&dir.0).unwrap_or(&0).add(dir.1),
                );
            }
        }
        dir_map
            .into_iter()
            .max_by(|x, y| x.1.cmp(&y.1))
            .unwrap_or(((1, 0), 0))
            .0
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        test_utils::scenarios::{
            get_board, get_scenario, AVOID_DEATH_ADVANCED,
            AVOID_HEAD_TO_HEAD_DEATH, MULTI_SNAKE,
        },
        utils::dir_to_string,
    };

    #[test]
    fn test_avoid_wall() {
        let game_state = get_board();
        let mut tree = Multitree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!("up", best_move)
    }

    #[test]
    fn test_avoid_death_advanced() {
        let game_state = get_scenario(AVOID_DEATH_ADVANCED);
        let mut tree = Multitree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "right")
    }

    #[test]
    fn test_avoid_head_to_head_death() {
        let game_state = get_scenario(AVOID_HEAD_TO_HEAD_DEATH);
        let mut tree = Multitree::new(
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
        let mut tree = Multitree::new(
            MonteCarloConfig::default(),
            game_state.board,
            game_state.you,
        );
        assert_eq!(dir_to_string(tree.get_best_move()).is_empty(), false);
    }
}
