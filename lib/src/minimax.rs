use std::{collections::HashMap, thread};

use crate::{
    board::{Action, EndState},
    config::MiniMaxConfig,
    floodfill::floodfill,
    models::{Battlesnake, Board},
    utils::{self, dir_to_string},
};

#[derive(Clone)]
struct NodeState {
    board_state: Board,
}

impl NodeState {
    const MAX_SCORE: f32 = 1000.0;

    // Heuristic values
    const FILL_V: f32 = 4.5;
    const LIFE_V: f32 = 0.0;
    const LENGTH_V: f32 = 10.0;

    pub fn generate_score_array(&self) -> Vec<f32> {
        let board = &self.board_state;
        let end_state: EndState = board.get_endstate();
        let mut scores = vec![];
        for (_, snake) in board.snakes.iter().enumerate() {
            scores.push(
                self.calculate_raw_score_per_snake(
                    &snake.id, &end_state, &board,
                ),
            )
        }
        let mut total_score = scores.iter().fold(0.0, |acc, x| acc + x);
        if total_score == 0.0 {
            total_score = NodeState::MAX_SCORE
        }
        return scores
            .iter()
            .map(|x| (x / total_score) * NodeState::MAX_SCORE)
            .collect();
    }

    fn calculate_raw_score_per_snake(
        &self,
        snake_id: &str,
        end_state: &EndState,
        board: &Board,
    ) -> f32 {
        match end_state {
            EndState::Winner(winner) => {
                if winner == snake_id {
                    return NodeState::MAX_SCORE;
                }
                return -NodeState::MAX_SCORE;
            }
            EndState::Playing => { /* CONTINUE */ }
            EndState::Tie => return -NodeState::MAX_SCORE,
        }
        let fill_score = floodfill(board, snake_id);
        let snake = board.get_snake(snake_id);
        let health_score = snake.health;
        let length_score = snake.body.len();
        let mut final_score = (health_score as f32) * NodeState::LIFE_V;
        final_score += (length_score as f32) * NodeState::LENGTH_V;
        final_score += (fill_score as f32) * NodeState::FILL_V;
        return final_score;
    }
}

pub struct Tree {
    snake_map: HashMap<String, usize>,
    snake_vec: Vec<String>,
    root: NodeState,
    target_snake_id: String,
    max_depth: usize,
}

impl Tree {
    pub const PARALLEL_DEPTH: usize = 6;
    pub fn get_next_snake(&self, current_snake: &str) -> &str {
        let next_index = self.snake_map[current_snake] + 1;
        return &self.snake_vec[next_index % self.snake_vec.len()];
    }

    pub fn is_last_nake(&self, current_snake: &str) -> bool {
        let cur_index = self.snake_map[current_snake];
        return cur_index + 1 == self.snake_vec.len();
    }

    pub fn new(
        config: MiniMaxConfig,
        mut starting_board: Board,
        starting_snake: Battlesnake,
    ) -> Self {
        let starting_snake_id = starting_snake.id.clone();
        utils::fix_snake_order(&mut starting_board, starting_snake);
        let mut snake_vec = vec![];
        let mut snake_map = HashMap::new();
        for (i, snake) in starting_board.snakes.iter().enumerate() {
            let copy_snake = &snake.id;
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
            target_snake_id: starting_snake_id,
            max_depth: config.depth,
        };
    }

    pub fn get_best_move(&self) -> (i32, i32) {
        let board_state = &self.root.board_state;
        let current_snake = &self.target_snake_id;

        let alphas = vec![NodeState::MAX_SCORE; board_state.snakes.len()];
        let (score, best_move) = self.get_score_parallel(
            0,
            self.root.clone(),
            alphas,
            current_snake.clone(),
        );

        println!("board state:\n{}", board_state.to_string());
        println!(
            "found best move {} with score {:?}",
            dir_to_string(best_move),
            score
        );

        return best_move;
    }

    fn get_score_parallel(
        &self,
        depth: usize,
        node_state: NodeState,
        alphas: Vec<f32>,
        current_snake: String,
    ) -> (Vec<f32>, (i32, i32)) {
        let mut best_dir = (1, 0);

        if depth == self.max_depth || node_state.board_state.is_terminal() {
            return (node_state.generate_score_array(), best_dir);
        }

        // If eliminated just skip the turn.
        if node_state
            .board_state
            .get_snake(current_snake.as_str())
            .eliminated_cause
            .is_some()
        {
            return self.get_score_parallel(
                depth,
                node_state,
                alphas,
                self.get_next_snake(&current_snake).to_owned(),
            );
        }
        let mut new_alphas = alphas.clone();
        let board_state = &node_state.board_state;
        let mut max_score = vec![];

        thread::scope(|s| {
            let mut handles = vec![];
            for dir in board_state.get_valid_moves(&current_snake) {
                // Perform alpha pruning.
                // If we found a move better than what is above us we can stop looking.
                if max_score.len() > 0
                    && max_score[self.snake_map[&current_snake]]
                        > alphas[self.snake_map[&current_snake]]
                {
                    break;
                }

                let mut board_copy = board_state.clone();
                let action = Action {
                    snake_id: current_snake.to_owned(),
                    dir,
                };
                board_copy
                    .execute_action(action, self.is_last_nake(&current_snake));

                let new_node = NodeState {
                    board_state: board_copy,
                };

                let passed_alphas = new_alphas.clone();
                let next_snake =
                    self.get_next_snake(&current_snake).to_string();
                let handle = s.spawn(|| {
                    if depth >= Tree::PARALLEL_DEPTH {
                        return self.get_score_parallel(
                            depth + 1,
                            new_node,
                            passed_alphas,
                            next_snake,
                        );
                    }

                    return self.get_score(
                        depth + 1,
                        new_node,
                        passed_alphas,
                        next_snake,
                    );
                });

                handles.push((dir, handle));
            }
            loop {
                let num_left =
                    handles.iter().filter(|th| !th.1.is_finished()).count();
                if num_left == 0 {
                    break;
                }
            }

            for handle in handles {
                let dir = handle.0;
                match handle.1.join() {
                    Ok((new_score, _)) => {
                        if max_score.len() == 0
                            || new_score[self.snake_map[&current_snake]]
                                > max_score[self.snake_map[&current_snake]]
                        {
                            best_dir = dir;
                            max_score = new_score;
                            for index in 0..new_alphas.len() {
                                if index == self.snake_map[&current_snake] {
                                    new_alphas[index] = max_score
                                        [self.snake_map[&current_snake]]
                                } else {
                                    new_alphas[index] = NodeState::MAX_SCORE
                                        - max_score
                                            [self.snake_map[&current_snake]]
                                }
                            }
                        }
                    }
                    Err(_) => panic!("panicked on thread"),
                }
            }
        });

        return (max_score, best_dir);
    }

    fn get_score(
        &self,
        depth: usize,
        node_state: NodeState,
        alphas: Vec<f32>,
        current_snake: String,
    ) -> (Vec<f32>, (i32, i32)) {
        let mut best_dir = (1, 0);

        if depth == self.max_depth || node_state.board_state.is_terminal() {
            return (node_state.generate_score_array(), best_dir);
        }

        // If eliminated just skip the turn.
        if node_state
            .board_state
            .get_snake(&current_snake)
            .eliminated_cause
            .is_some()
        {
            return self.get_score(
                depth,
                node_state,
                alphas,
                self.get_next_snake(&current_snake).to_owned(),
            );
        }

        let mut new_alphas = alphas.clone();
        let board_state = &node_state.board_state;
        let mut max_score = vec![];

        for dir in board_state.get_valid_moves(&current_snake) {
            // Perform alpha pruning.
            // If we found a move better than what is above us we can stop looking.
            if max_score.len() > 0
                && max_score[self.snake_map[&current_snake]]
                    > alphas[self.snake_map[&current_snake]]
            {
                break;
            }

            let mut board_copy = board_state.clone();
            let action = Action {
                snake_id: current_snake.to_owned(),
                dir,
            };
            board_copy
                .execute_action(action, self.is_last_nake(&current_snake));

            let new_node = NodeState {
                board_state: board_copy,
            };
            let (new_score, _) = self.get_score(
                depth + 1,
                new_node,
                new_alphas.clone(),
                self.get_next_snake(&current_snake).to_owned(),
            );

            if max_score.len() == 0
                || new_score[self.snake_map[&current_snake]]
                    > max_score[self.snake_map[&current_snake]]
            {
                best_dir = dir;
                max_score = new_score;
                for index in 0..new_alphas.len() {
                    if index == self.snake_map[&current_snake] {
                        new_alphas[index] =
                            max_score[self.snake_map[&current_snake]]
                    } else {
                        new_alphas[index] = NodeState::MAX_SCORE
                            - max_score[self.snake_map[&current_snake]]
                    }
                }
            }
        }
        return (max_score, best_dir);
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::test_utils::scenarios::{
        get_board, get_scenario, AVOID_DEATH_ADVANCED, AVOID_DEATH_GET_FOOD,
        AVOID_HEAD_TO_HEAD_DEATH, AVOID_SELF_TRAP, DO_NOT_CIRCLE_FOOD,
        GET_THE_FOOD,
    };

    #[test]
    fn test_avoid_wall() {
        let game_state = get_board();
        let tree = Tree::new(
            MiniMaxConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!("up", best_move)
    }

    #[test]
    fn test_avoid_death_get_food() {
        let game_state = get_scenario(AVOID_DEATH_GET_FOOD);
        let tree = Tree::new(
            MiniMaxConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "right")
    }

    #[test]
    fn test_avoid_self_trap() {
        let game_state = get_scenario(AVOID_SELF_TRAP);
        let tree = Tree::new(
            MiniMaxConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "up")
    }
    #[test]
    fn test_get_easy_food() {
        let game_state = get_scenario(GET_THE_FOOD);
        let tree = Tree::new(
            MiniMaxConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_eq!(best_move, "down")
    }

    #[test]
    fn test_avoid_death_advanced() {
        let game_state = get_scenario(AVOID_DEATH_ADVANCED);
        let tree = Tree::new(
            MiniMaxConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "right")
    }

    #[test]
    fn test_do_not_circle_food() {
        let game_state = get_scenario(DO_NOT_CIRCLE_FOOD);
        let tree = Tree::new(
            MiniMaxConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_eq!(best_move, "up")
    }

    #[test]
    fn test_avoid_head_to_head_death() {
        let game_state = get_scenario(AVOID_HEAD_TO_HEAD_DEATH);
        let tree = Tree::new(
            MiniMaxConfig::default(),
            game_state.board,
            game_state.you,
        );
        let best_move = dir_to_string(tree.get_best_move());
        assert_ne!(best_move, "left")
    }
}
