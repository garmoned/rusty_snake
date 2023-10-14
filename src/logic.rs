use log::info;
use serde_json::{json, Value};

use crate::minimax::Tree;
use crate::models::{Battlesnake, Board, Game};
use crate::utils::dir_to_string;

pub fn info() -> Value {
    info!("INFO");
    json!({
        "apiversion": "1",
        "author": "count_zero",
        "color": "#3b228c",
        "head": "pop-star",
        "tail": "shiny",
    })
}

// start is called when your Battlesnake begins a game
pub fn start(_game: &Game, _turn: &u32, _board: &Board, _you: &Battlesnake) {
    info!("GAME START");
}

// end is called when your Battlesnake finishes a game
pub fn end(_game: &Game, _turn: &u32, _board: &Board, _you: &Battlesnake) {
    info!("GAME OVER");
}

pub fn get_move(board: &Board, you: &Battlesnake) -> Value {
    let tree = Tree::new(board.clone(), you.clone());
    let tup = tree.get_best_move();
    json!({ "move": dir_to_string(tup) })
}
