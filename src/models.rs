use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Debug)]
pub struct Game {
    pub id: String,
    pub ruleset: HashMap<String, Value>,
    pub timeout: u32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Board {
    pub height: u32,
    pub width: u32,
    pub food: Vec<Coord>,
    pub snakes: Vec<Battlesnake>,
    pub hazards: Vec<Coord>,
    pub winner: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Battlesnake {
    pub id: String,
    pub name: String,
    pub health: u32,
    pub body: Vec<Coord>,
    pub head: Coord,
    pub length: u32,
    pub latency: String,
    pub shout: Option<String>,
    pub eliminated_cause: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Hash, Eq, PartialEq)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

impl Coord {
    pub fn default() -> Self {
        return Self { x: 0, y: 0 };
    }

    pub fn intersect(&self, coord: &Coord) -> bool {
        return self.x == coord.x && self.y == self.y;
    }

    pub fn in_bounds(&self, width: i32, height: i32) -> bool {
        return self.x < width && self.y < height && self.y >= 0 && self.x >= 0;
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct GameState {
    pub game: Game,
    pub turn: u32,
    pub board: Board,
    pub you: Battlesnake,
}
