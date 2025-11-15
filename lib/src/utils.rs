use std::str::FromStr;

use crate::models::{Battlesnake, Board};



pub const RIGHT: (i32, i32) = (0, 1);
pub const UP: (i32, i32) = (1, 0);
pub const LEFT: (i32, i32) = (0, -1);
pub const DOWN: (i32, i32) = (-1, 0);

pub const UP_IDX: usize = 0;
pub const DOWN_IDX: usize = 1;
pub const LEFT_IDX: usize = 2;
pub const RIGHT_IDX: usize = 3;

pub const DIRECTIONS: [(i32, i32); 4] = [UP, DOWN, LEFT, RIGHT];

pub fn dir_to_string(tup: (i32, i32)) -> String {
    if tup == DOWN {
        return "down".to_owned();
    }
    if tup == RIGHT {
        return "right".to_owned();
    }
    if tup == LEFT {
        return "left".to_owned();
    }
    if tup == UP {
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
