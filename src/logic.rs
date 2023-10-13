use log::info;
use serde_json::{json, Value};

use crate::minimax::Tree;
use crate::models::{Battlesnake, Board, Game};
use crate::utils::dir_to_string;

pub fn info() -> Value {
    info!("INFO");

    return json!({
        "apiversion": "1",
        "author": "", // TODO: Your Battlesnake Username
        "color": "#888888", // TODO: Choose color
        "head": "default", // TODO: Choose head
        "tail": "default", // TODO: Choose tail
    });
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
    return json!({ "move": dir_to_string(tup) });
}
