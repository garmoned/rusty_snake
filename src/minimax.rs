use std::{borrow::Borrow, collections::HashMap};



use crate::{
    floodfill::floodfill,
    models::Board,
    simulation::{Action, EndState},
    utils,
};

#[derive(Clone)]
struct NodeState {
    board_state: Board,
}

impl NodeState {
    // Heuristic values
    const FILL_V: f32 = 1.0;
    const LIFE_V: f32 = 1.0;
    const LENGTH_V: f32 = 1.0;

    pub fn generate_score_array(&self) -> Vec<f32> {
        let board = &self.board_state;
        let end_state: EndState = board.get_endstate();
        let mut scores = vec![];
        for snake in &board.snakes {
            scores.push(self.calculate_raw_score_per_snake(&snake.id, &end_state, &board))
        }
        return scores;
    }

    fn calculate_raw_score_per_snake(
        &self,
        snake_id: &str,
        end_state: &EndState,
        board: &Board,
    ) -> f32 {
        if let EndState::Winner(winner) = end_state {
            if winner == snake_id {
                return f32::INFINITY;
            } else {
                return f32::NEG_INFINITY;
            }
        }
        match end_state {
            EndState::Winner(winner) => {
                if winner == snake_id {
                    return f32::INFINITY;
                }
                return f32::NEG_INFINITY;
            }
            EndState::Playing => { /* CONTINUE */ }
            EndState::TIE => return 0.0,
        }
        let fill_score = floodfill(board, snake_id);
        let snake = board.get_snake(snake_id);
        let health_score = snake.health;
        let length_score = snake.body.len();
        let mut final_score = (health_score as f32) * NodeState::FILL_V;
        final_score += (length_score as f32) * NodeState::LENGTH_V;
        final_score += (fill_score as f32) * NodeState::FILL_V;
        return final_score;
    }
}

struct Tree {
    snake_map: HashMap<String, usize>,
    snake_vec: Vec<String>,
    root: NodeState,
}

impl Tree {
    const MAX_DEPTH: usize = 4;

    pub fn get_next_snake(&self, current_snake: &str) -> &str {
        let cur_index = self.snake_map[current_snake];
        return &self.snake_vec[cur_index % self.snake_vec.len()];
    }

    pub fn is_last_nake(&self, current_snake: &str) -> bool {
        let cur_index = self.snake_map[current_snake];
        return cur_index + 1 == current_snake.len();
    }

    pub fn new(starting_board: Board) -> Self {
        let mut snake_vec = vec![];
        let mut snake_map = HashMap::new();

        for (i, snake) in starting_board.borrow().snakes.iter().enumerate() {
            let copy_snake = snake.id.clone();
            snake_vec.push(copy_snake.clone());
            snake_map.insert(copy_snake.clone(), i);
        }

        let root_node_state = NodeState {
            board_state: starting_board,
        };

        return Self {
            snake_map,
            snake_vec,
            root: root_node_state,
        };
    }

    pub fn get_best_move(&self) -> (i32, i32) {
        let mut move_buffer = [false; 4];
        self.root
            .board_state
            .get_valid_actions(self.snake_vec.get(0).unwrap(), &mut move_buffer);

        let board_state = &self.root.board_state;
        let current_snake = self.snake_vec.get(0).unwrap();

        let mut max_score = vec![];
        let mut best_move = (1, 0);

        for (i, can_move) in move_buffer.iter().enumerate() {
            if !can_move {
                continue;
            }
            let dir = utils::DIRECTIONS[i];
            let mut board_copy = board_state.clone();
            let action = Action {
                snake_id: current_snake.to_string(),
                dir,
            };
            board_copy.execute_action(action, self.is_last_nake(current_snake));
            let new_node = NodeState {
                board_state: board_copy,
            };
            let new_score = self.get_score(0 + 1, &new_node, &self.get_next_snake(current_snake));

            if max_score.len() == 0 {
                max_score = new_score;
            } else if new_score[self.snake_map[current_snake]]
                > max_score[self.snake_map[current_snake]]
            {
                max_score = new_score;
                best_move = dir;
            }
        }
        return best_move;
    }

    //TODO Add A/B pruning
    fn get_score(&self, depth: usize, node_state: &NodeState, current_snake: &str) -> Vec<f32> {
        if depth == Tree::MAX_DEPTH {
            return node_state.generate_score_array();
        }

        let board_state = &node_state.board_state;
        let mut move_buffer = [false; 4];

        board_state.get_valid_actions(current_snake, &mut move_buffer);

        let mut max_score = vec![];

        for (i, can_move) in move_buffer.iter().enumerate() {
            if !can_move {
                continue;
            }
            let dir = utils::DIRECTIONS[i];
            let mut board_copy = board_state.clone();
            let action = Action {
                snake_id: current_snake.to_owned(),
                dir,
            };
            board_copy.execute_action(action, self.is_last_nake(current_snake));
            let new_node = NodeState {
                board_state: board_copy,
            };
            let new_score =
                self.get_score(depth + 1, &new_node, &self.get_next_snake(current_snake));

            if max_score.len() == 0 {
                max_score = new_score;
            } else if new_score[self.snake_map[current_snake]]
                > max_score[self.snake_map[current_snake]]
            {
                max_score = new_score
            }
        }

        if max_score.len() == 0 {
            return node_state.generate_score_array();
        }

        return max_score;
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::test_utils;

    #[test]
    fn my_test() {
        let game_state = test_utils::get_board();
        let tree = Tree::new(game_state);
        tree.get_best_move();
    }
}
