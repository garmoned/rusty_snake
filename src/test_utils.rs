#[cfg(test)]
pub mod scenarios {

    use crate::models::GameState;
    use std::{fs, path::PathBuf};

    const TEST_REQUEST: &str = "./scenarios/test_request.json";
    const GAME_OVER: &str = "./scenarios/end_game.json";
    pub const AVOID_DEATH_GET_FOOD: &str =
        "./scenarios/avoid_death_get_food.json";
    pub const AVOID_SELF_TRAP: &str = "./scenarios/avoid_self_trap.json";
    pub const GET_THE_FOOD: &str = "./scenarios/get_the_food.json";
    pub const AVOID_DEATH_ADVANCED: &str =
        "./scenarios/avoid_death_advanced.json";
    pub const DO_NOT_CIRCLE_FOOD: &str = "./scenarios/do_not_circle_food.json";
    pub const AVOID_HEAD_TO_HEAD_DEATH: &str =
        "./scenarios/avoid_head_to_head_death.json";
    pub const MULTI_SNAKE: &str = "./scenarios/multi_snake.json";

    pub fn load_game_state(path: &str) -> GameState {
        let full_path = PathBuf::from(path);
        let data = fs::read_to_string(full_path.as_path())
            .expect("unable to read request file");
        let game_state: GameState = serde_json::from_str(data.as_str())
            .expect("Failed to parse json game state");
        return game_state;
    }

    pub fn game_over_board() -> GameState {
        load_game_state(GAME_OVER)
    }

    pub fn get_board() -> GameState {
        load_game_state(TEST_REQUEST)
    }

    pub fn get_scenario(path: &str) -> GameState {
        load_game_state(path)
    }
}
