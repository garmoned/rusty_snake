use rand::seq::SliceRandom;

use crate::{
    models::{Battlesnake, Board, Coord},
    utils,
};
use std::{collections::HashMap, convert::TryInto, vec};

#[derive(Clone)]
pub struct Action {
    pub snake_id: String,
    pub dir: (i32, i32),
}

const GENERIC_ELIMINATION: &str = "DED";
const SELF_ELIMINATE: &str = "eliminated itself";
const SNAKE_MAX_HEALTH: u32 = 100;

#[derive(Eq, PartialEq, Debug)]
pub enum EndState {
    Winner(String),
    Playing,
    Tie,
}

impl EndState {
    pub fn is_terminal(&self) -> bool {
        match &self {
            EndState::Winner(_) => return true,
            EndState::Playing => return false,
            EndState::Tie => return true,
        }
    }
}

impl Board {
    pub fn execute_random_move(&mut self, snake_id: &str) {
        let valid_moves = self.get_valid_moves(snake_id);
        let dir = valid_moves.choose(&mut rand::thread_rng()).unwrap();
        self.execute_dir(snake_id, dir.clone());
    }

    pub fn get_valid_moves(&self, snake_id: &str) -> Vec<(i32, i32)> {
        let mut dirs = vec![];
        let snake = self.get_snake(snake_id);

        // A dead snake will stay where it is.
        if snake.is_eliminated() {
            return vec![(0, 0)];
        }

        let head = &snake.head;
        for dir in utils::DIRECTIONS {
            let mut coord = Coord::default();
            coord.x = head.x + dir.1;
            coord.y = head.y + dir.0;
            if !coord.in_bounds(self.width(), self.height()) {
                continue;
            }
            if self.intersect_any_snake_body(&coord, snake_id) {
                continue;
            }
            dirs.push(dir)
        }
        if dirs.is_empty() {
            dirs.push((1, 0))
        }
        return dirs;
    }

    pub fn intersect_any_snake_body(
        &self,
        coord: &Coord,
        snake_id: &str,
    ) -> bool {
        for other_snake in &self.snakes {
            for (index, bod) in other_snake.body.iter().enumerate() {
                // Do not check for collisions with own head.
                if index == 0 {
                    continue;
                }
                // It is valid to move into your own tail.
                if snake_id == other_snake.id
                    && index == other_snake.body.len() - 1
                {
                    continue;
                }
                if bod.intersect(coord) {
                    return true;
                }
            }
        }
        return false;
    }

    pub fn to_string(&self) -> String {
        return self.to_string_with_depth(0);
    }

    pub fn width(&self) -> i32 {
        self.width.try_into().unwrap()
    }

    pub fn height(&self) -> i32 {
        self.height.try_into().unwrap()
    }

    pub fn to_string_with_depth(&self, depth: usize) -> String {
        let mut grid = vec![];
        for _ in 0..self.height {
            let mut row = vec![];
            for _ in 0..self.width {
                row.push(".")
            }
            grid.push(row)
        }
        let mut string = "".to_string();

        let mut depth_string = "".to_string();
        for _ in 0..depth {
            depth_string += "---";
        }

        for snake in &self.snakes {
            for bod in &snake.body {
                let mut icon = "#";
                if bod.intersect(&snake.head) {
                    icon = "@"
                }
                if bod.in_bounds(
                    self.width.try_into().unwrap(),
                    self.height.try_into().unwrap(),
                ) {
                    let row = grid.get_mut(bod.y as usize).unwrap();
                    row[bod.x as usize] = icon
                }
            }
        }

        for food in &self.food {
            grid[food.y as usize][food.x as usize] = "O";
        }

        for row in grid.iter().rev() {
            let mut str_row = "".to_string();
            for ch in row {
                str_row += ch;
            }
            string += depth_string.as_ref();
            string += &str_row;
            string += "\n";
        }
        return string;
    }
}

// Returns true if game is over
// Board is modified directly.
impl Board {
    pub fn execute_action(
        &mut self,
        action: Action,
        last_snake: bool,
    ) -> EndState {
        return self.execute(&action.snake_id, action.dir, last_snake);
    }

    pub fn execute_dir(&mut self, snake_id: &str, dir: (i32, i32)) -> EndState {
        return self.execute(snake_id, dir, self.is_last_snake(&snake_id));
    }

    pub fn is_last_snake(&self, snake_id: &str) -> bool {
        return self.snakes.get(self.snakes.len() - 1).unwrap().id == snake_id;
    }

    pub fn execute(
        &mut self,
        snake_id: &str,
        dir: (i32, i32),
        last_snake: bool,
    ) -> EndState {
        let end_state = self.get_endstate();

        match end_state {
            EndState::Winner(_) => return end_state,
            EndState::Playing => { /*continue */ }
            EndState::Tie => return end_state,
        }

        self.move_snake(snake_id, dir);
        self.eliminate_snakes();

        // If it is not the last snake only do the moving.
        if !last_snake {
            return EndState::Playing;
        }
        self.reduce_snake_health();
        self.feed_snakes();
        self.eliminate_snakes();
        self.eliminate_via_collisions();

        return self.get_endstate();
    }

    fn feed_snakes(&mut self) {
        let mut new_food: Vec<Coord> = vec![];
        for food in &self.food {
            let mut food_has_been_eaten = false;
            for snake in &mut self.snakes {
                if snake.body.is_empty() {
                    continue;
                }
                if snake.eliminated_cause.is_some() {
                    continue;
                }
                if snake.head.intersect(food) {
                    snake.feed_snake();
                    food_has_been_eaten = true
                }
            }
            if !food_has_been_eaten {
                new_food.push(food.clone());
            }
        }
        self.food = new_food
    }

    fn eliminate_snakes(&mut self) {
        for snake in &mut self.snakes {
            if snake.is_eliminated() {
                continue;
            }
            if snake.body.len() == 0 {
                panic!("Zero length snake")
            }

            if snake.out_of_health() {
                snake.eliminate();
                continue;
            }
            if snake.snake_is_out_of_bounds(
                self.height.try_into().unwrap(),
                self.width.try_into().unwrap(),
            ) {
                snake.eliminate();
                continue;
            }
            if snake.self_collision() {
                snake.self_eliminate();
                continue;
            }
        }
    }

    fn eliminate_via_collisions(&mut self) {
        let mut is_eliminated = HashMap::<String, bool>::new();
        for snake in &self.snakes {
            is_eliminated.insert(
                snake.id.to_string(),
                snake.collides_with(&self.snakes),
            );
        }

        for snake in &mut self.snakes {
            if *is_eliminated.get(&snake.id).unwrap() {
                snake.eliminate();
            }
        }
    }

    fn reduce_snake_health(&mut self) {
        for snake in &mut self.snakes {
            snake.reduce_health()
        }
    }

    fn move_snake(&mut self, snake_id: &str, dir: (i32, i32)) {
        for snake in &mut self.snakes {
            if snake.body.is_empty() {
                panic!("Trying to move snakes with zero length body")
            }

            // Is this hacky? I don't think so.
            if dir == (0, 0) {
                return;
            }

            if snake_id == snake.id {
                let last_index = snake.body.len() - 1;
                let mut new_head = Coord::default();
                new_head.x = snake.body.get(0).unwrap().x + dir.1;
                new_head.y = snake.body.get(0).unwrap().y + dir.0;
                snake.body.rotate_left(last_index);
                snake.body.get_mut(0).unwrap().x = new_head.x;
                snake.body.get_mut(0).unwrap().y = new_head.y;
                snake.head = new_head;
            }
        }
    }

    pub fn is_terminal(&self) -> bool {
        match self.get_endstate() {
            EndState::Winner(_) => return true,
            EndState::Playing => return false,
            EndState::Tie => return true,
        }
    }

    pub fn get_endstate(&self) -> EndState {
        let mut snakes_remaining = 0;
        let mut alive_snake_id = "".to_string();
        for snake in &self.snakes {
            if snake.eliminated_cause.is_none() {
                snakes_remaining += 1;
                alive_snake_id = snake.id.clone();
                continue;
            }
        }

        if snakes_remaining == 1 {
            return EndState::Winner(alive_snake_id);
        }

        if snakes_remaining == 0 {
            return EndState::Tie;
        }

        return EndState::Playing;
    }

    pub fn get_snake(&self, snake_id: &str) -> &Battlesnake {
        for snake in &self.snakes {
            if snake.id == snake_id {
                return snake;
            }
        }
        panic!("Snake not found")
    }
}

impl Battlesnake {
    fn eliminate(&mut self) {
        self.eliminated_cause = Some(GENERIC_ELIMINATION.to_string())
    }

    fn feed_snake(&mut self) {
        self.health = SNAKE_MAX_HEALTH;
        self.body.push(self.body.last().unwrap().clone())
    }

    fn reduce_health(&mut self) {
        if self.health > 0 {
            self.health -= 1
        }
    }

    fn self_collision(&self) -> bool {
        let head_collide =
            Battlesnake::head_collide_body(&self.head, &self.body);
        return head_collide;
    }

    fn self_eliminate(&mut self) {
        self.eliminated_cause = Some(SELF_ELIMINATE.to_string())
    }

    fn collides_with(&self, snakes: &Vec<Battlesnake>) -> bool {
        for other_snake in snakes {
            if other_snake.is_eliminated() {
                continue;
            }

            if self.dies_head_to_head(other_snake) {
                return true;
            }
            if self.body_collision(other_snake) {
                return true;
            }
        }
        return false;
    }

    fn body_collision(&self, other_snake: &Battlesnake) -> bool {
        Battlesnake::head_collide_body(&self.head, &other_snake.body)
    }

    fn dies_head_to_head(&self, other_snake: &Battlesnake) -> bool {
        return self.head.intersect(&other_snake.head)
            && other_snake.body.len() > self.body.len();
    }

    fn head_collide_body(head: &Coord, body: &Vec<Coord>) -> bool {
        for (i, bod) in body.iter().enumerate() {
            if i == 0 {
                continue;
            }
            if head.intersect(bod) {
                return true;
            }
        }
        return false;
    }

    fn snake_is_out_of_bounds(&self, height: i32, width: i32) -> bool {
        let head = self.body.get(0).unwrap();
        return head.x >= width || head.x < 0 || head.y >= height || head.y < 0;
    }

    fn out_of_health(&self) -> bool {
        return self.health == 0;
    }

    fn is_eliminated(&self) -> bool {
        return self.eliminated_cause.is_some();
    }
}

#[cfg(test)]
mod test {
    use crate::{
        simulation::EndState,
        test_utils::scenarios::{
            game_over_board, get_board, get_scenario, AVOID_DEATH_GET_FOOD,
            GET_THE_FOOD,
        },
    };

    #[test]
    fn test_game_over() {
        let game_state = game_over_board().board;
        assert_eq!(
            game_state.get_endstate(),
            EndState::Winner("gs_cGHvRfpVm3cx7Y3kqr4dqMfY".to_string())
        )
    }

    #[test]
    fn basic_move() {
        let mut board = get_board().board;
        board.move_snake("short_snake", (1, 0));
    }

    #[test]
    fn dies_to_neck() {
        let mut board = get_board().board;
        board.execute("long_snake", (-1, 0), false);
        assert!(board.is_terminal());
    }

    #[test]
    fn dies_to_out_of_bounds() {
        let mut board = get_board().board;
        board.execute("long_snake", (1, 0), false);
        let winner = board.get_endstate();
        let _short_name = "short_snake".to_string();
        matches!(winner, EndState::Winner(_short_name));
    }

    #[test]
    fn survives_move() {
        let mut board = get_board().board;
        board.execute("long_snake", (0, -1), false);
        assert_ne!(board.is_terminal(), true);
    }

    #[test]
    fn test_dies_head_to_head() {
        let mut game = get_scenario(AVOID_DEATH_GET_FOOD);
        let id1 = game.board.snakes[0].id.clone();
        let id2 = game.board.snakes[1].id.clone();
        game.board.execute(&id1, (0, 1), false);
        game.board.execute(&id2, (0, -1), true);
        assert!(game.board.is_terminal());
    }

    #[test]
    fn board_deep_clones() {
        let mut board = get_board().board;
        let board_2 = board.clone();
        board.execute("long_snake", (-1, 0), false);
        assert!(board.is_terminal());
        assert_ne!(board_2.is_terminal(), true);
    }

    #[test]
    fn test_get_easy_food() {
        let mut game = get_scenario(GET_THE_FOOD);
        let id1 = game.board.snakes[0].id.clone();
        let id2 = game.board.snakes[1].id.clone();
        assert_eq!(game.board.snakes.get(0).unwrap().body.len(), 4);
        game.board.execute(&id1, (-1, 0), false);
        game.board.execute(&id2, (0, -1), true);
        assert_eq!(game.board.snakes.get(0).unwrap().body.len(), 5);
    }
}
