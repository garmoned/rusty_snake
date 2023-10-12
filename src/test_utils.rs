const TEST_REQUEST: &str = "./test_request.json";
const GAME_OVER: &str = "./end_game.json";
pub const AVOID_DEATH_GET_FOOD: &str = "./avoid_death_get_food.json";
pub const AVOID_SELF_TRAP: &str = "./avoid_self_trap.json";
pub const GET_THE_FOOD: &str = "./get_the_food.json";
pub const AVOID_KILLING_SELF: &str = "./avoid_killing_self.json";
pub const AVOID_DEATH_ADVANCED: &str = "./avoid_death_advanced.json";

use crate::models::{Board, GameState};
use std::{convert::TryInto, fs, path::PathBuf};

pub fn load_game_state(path: &str) -> GameState {
    let full_path = PathBuf::from(path);
    let data = fs::read_to_string(full_path.as_path()).expect("unable to read request file");
    let game_state: GameState =
        serde_json::from_str(data.as_str()).expect("Failed to parse json game state");
    return game_state;
}

pub fn game_over_board() -> Board {
    load_game_state(GAME_OVER).board
}

pub fn get_board() -> Board {
    load_game_state(TEST_REQUEST).board
}

pub fn get_scenario(path: &str) -> GameState {
    load_game_state(path)
}

pub fn friendly_snake() -> String {
    return "MontyPython".to_owned();
}

pub fn print_board(board: &Board) {
    let mut grid = vec!['.'; (board.width * board.height).try_into().unwrap()].into_boxed_slice();

    let snake_shape = ['X', '#', '@'];

    let mut snake_index = 0;

    for snake in &board.snakes {
        for body in &snake.body {
            let y_offset: usize = (body.y * (board.width as i32)).try_into().unwrap();
            grid[y_offset + (body.x as usize)] = snake_shape[snake_index];
        }
        snake_index += 1;
    }

    for y in 0..board.height {
        print!("|");
        for x in 0..board.width {
            let y_offset: usize = (y * (board.width as u32)).try_into().unwrap();
            print!("{}", grid[y_offset + (x as usize)])
        }
        println!("|")
    }
}
