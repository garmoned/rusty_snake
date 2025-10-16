use std::str::FromStr;

use crate::models::{Battlesnake, Board};

pub const DIRECTIONS: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

pub fn dir_to_string(tup: (i32, i32)) -> String {
    if tup == (-1, 0) {
        return "down".to_owned();
    }
    if tup == (0, 1) {
        return "right".to_owned();
    }
    if tup == (0, -1) {
        return "left".to_owned();
    }
    if tup == (1, 0) {
        return "up".to_owned();
    }
    panic!("invalid direction");
}

pub fn fix_snake_order(board: &mut Board, starting_snake: Battlesnake) {
    let starting_snake_id = starting_snake.id.clone();
    let mut new_snakes = vec![starting_snake];
    for snake in &board.snakes {
        if snake.id == starting_snake_id {
            continue;
        }
        new_snakes.push(snake.clone())
    }
    board.snakes = new_snakes
}

pub fn read_env<T: ToString + FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .unwrap_or(default.to_string())
        .parse()
        .unwrap_or(default)
}
