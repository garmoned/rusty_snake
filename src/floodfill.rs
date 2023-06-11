use crate::{
    models::{Board, Coord},
    utils,
};
use std::{
    collections::{HashSet, VecDeque},
    convert::TryInto,
};

// TODO: Optimize this to floodfill for each snake
// so we don't have to floodfill multiple times per scoring
pub fn floodfill(board: &Board, snake_id: &str) -> usize {
    let mut barriers = HashSet::<Coord>::new();
    let mut filled_tiles = HashSet::<Coord>::new();

    let mut target_snake = board.snakes.get(0).unwrap();

    for snake in &board.snakes {
        if snake.id == snake_id {
            target_snake = snake;
        }
        for coord in &snake.body {
            barriers.insert(coord.clone());
        }
    }
    let mut q = VecDeque::<Coord>::new();
    let start_head = target_snake.body.get(0).unwrap();
    q.push_back(start_head.clone());
    while q.len() > 0 {
        let expand_from = q.pop_front().unwrap();

        for (x, y) in utils::DIRECTIONS {
            let mut new_explore = Coord::default();
            new_explore.x = expand_from.x + x;
            new_explore.y = expand_from.y + y;
            // If the space is empty, in bounds, and unexplored mark it as accessible
            // and push it to be explored further from
            if new_explore.in_bounds(
                board.width.try_into().unwrap(),
                board.height.try_into().unwrap(),
            ) && !barriers.contains(&new_explore)
                && !filled_tiles.contains(&new_explore)
            {
                q.push_back(new_explore.clone());
                filled_tiles.insert(new_explore);
            }
        }
    }

    return filled_tiles.len();
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::test_utils;

    #[test]
    fn my_test() {
        let game_state = test_utils::get_board();

        test_utils::print_board(&game_state);

        println!("{}", floodfill(&game_state, "gs_cfw6qPQGty6hVKV9mDMPFXP6"));
        println!("{}", floodfill(&game_state, "gs_cGHvRfpVm3cx7Y3kqr4dqMfY"));

        assert_eq!(1, 0)
    }
}
