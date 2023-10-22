use std::{
    collections::{hash_map::RandomState, HashMap},
    hash::Hash,
};

use super::{node_state::NodeState, tree::Dir};
use crate::models::Board;
use std::hash::{BuildHasher, Hasher};

#[derive(Clone)]
struct RaveState {
    wins: i32,
    visits: i32,
}

#[derive(Hash, Clone)]
struct RaveKey {
    board: Board,
    snake_who_moved: String,
    taken_dir: Dir,
}

type RaveHash = u64;

#[derive(Clone)]
pub struct RaveTable {
    random: RandomState,
    hash_map: HashMap<RaveHash, RaveState>,
}

impl RaveState {
    fn wins(&self) -> f64 {
        self.wins.into()
    }

    fn visits(&self) -> f64 {
        self.visits.into()
    }
    pub fn amaf(&self) -> f64 {
        self.wins() / self.visits()
    }
}

impl RaveKey {
    pub fn to_rave_hash(self, random: &RandomState) -> RaveHash {
        let mut hasher = random.build_hasher();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl RaveTable {
    pub(crate) fn new() -> Self {
        Self {
            random: RandomState::new(),
            hash_map: HashMap::<RaveHash, RaveState>::new(),
        }
    }

    pub(crate) fn update_node(&mut self, node: &NodeState, win: bool) {
        let key = self.get_key(node);
        let save_state = self.hash_map.get_mut(&key);
        match save_state {
            Some(save_state) => {
                if win {
                    save_state.wins += 1
                }
                save_state.visits += 1;
            }
            None => {
                self.hash_map.insert(
                    key,
                    RaveState {
                        wins: if win { 1 } else { 0 },
                        visits: 1,
                    },
                );
            }
        };
    }

    pub(crate) fn get_amaf(&mut self, node: &NodeState) -> f64 {
        match self.get_state(node) {
            Some(state) => state.amaf(),
            None => 0.0,
        }
    }

    fn get_state(&mut self, node: &NodeState) -> Option<&RaveState> {
        let key = self.get_key(node);
        self.hash_map.get(&key)
    }

    fn get_key(&mut self, node: &NodeState) -> RaveHash {
        let key = RaveKey {
            board: node.board_state.clone(),
            snake_who_moved: node.snake_who_moved.clone(),
            taken_dir: node.taken_dir,
        };
        key.to_rave_hash(&self.random)
    }
}
