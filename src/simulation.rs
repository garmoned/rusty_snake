use std::{
    convert::TryInto,
    vec,
};

use crate::{
    models::{Battlesnake, Board, Coord},
    utils,
};

#[derive(Clone, Copy)]
pub enum Move {
    UP,
    DOWN,
    LEFT,
    RIGHT,
}
#[derive(Clone)]
pub struct Action {
    pub snake_id: String,
    pub dir: (i32, i32),
}

const GENERIC_ELIMINATION: &str = "DED";

const SNAKE_MAX_HEALTH: u32 = 100;

#[derive(Eq, PartialEq)]
pub enum EndState {
    Winner(String),
    Playing,
    TIE,
}

// Returns true if game is over
// Board is modified directly
impl Board {
    pub fn get_valid_actions(&self, snake_id: &str, move_buffer: &mut [bool; 4]) {
        let snake = self.get_snake(snake_id);
        let mut i = 0;
        for dir in utils::DIRECTIONS {
            // There will always be a head
            let mut new_head = snake.body.get(0).unwrap().clone();
            new_head.x += dir.0;
            new_head.y += dir.1;
            let is_neck = new_head == *snake.body.get(1).unwrap();
            move_buffer[i] = new_head.in_bounds(
                self.width.try_into().unwrap(),
                self.height.try_into().unwrap(),
            ) && !is_neck;
            i += 1;
        }
    }

    pub fn execute_action(&mut self, action: Action, last_snake: bool) -> EndState {
        return self.execute(action.snake_id, action.dir, last_snake);
    }

    pub fn execute(&mut self, snake_id: String, dir: (i32, i32), last_snake: bool) -> EndState {
        let end_state = self.get_endstate();

        match end_state {
            EndState::Winner(_) => return end_state,
            EndState::Playing => { /*continue */ }
            EndState::TIE => return end_state,
        }

        self.move_snake(snake_id, dir);

        //If it is not the last snake only do the moving
        if !last_snake {
            return EndState::Playing;
        }

        self.reduce_snake_health();
        self.feed_snakes();
        self.eliminate_snakes();

        return self.get_endstate();
    }

    fn feed_snakes(&mut self) {
        let mut new_food: Vec<Coord> = vec![];
        for food in &self.food {
            let mut food_has_been_eaten = false;
            for snake in &mut self.snakes {
                if let None = snake.eliminated_cause {
                    if snake.body.len() == 0 {
                        continue;
                    }
                }
                if snake.body.get(0).unwrap().intersect(&food) {
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
            if snake.body.len() <= 0 {
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
        }
    }

    fn reduce_snake_health(&mut self) {
        for snake in &mut self.snakes {
            snake.reduce_health()
        }
    }

    fn move_snake(&mut self, snake_id: String, dir: (i32, i32)) {
        for snake in &mut self.snakes {
            if snake.body.len() == 0 {
                panic!("trying to move snakes with zero length body")
            }

            // continue if the snake is eliminated
            if let None = snake.eliminated_cause {
                continue;
            }

            if snake_id == snake.id {
                let mut new_head = Coord::default();
                new_head.x = snake.body.get(0).unwrap().x + dir.0;
                new_head.y = snake.body.get(0).unwrap().y + dir.1;
                snake.body.rotate_right(0);
                snake.body.get_mut(0).unwrap().x = new_head.x;
                snake.body.get_mut(0).unwrap().y = new_head.y;
            }
        }
    }

    pub fn get_endstate(&self) -> EndState {
        let mut snakes_remaining = 0;
        let mut alive_snake_id = "";
        for snake in &self.snakes {
            if let None = snake.eliminated_cause {
                snakes_remaining += 1;
                alive_snake_id = &snake.id;
            }
        }

        if snakes_remaining == 1 {
            return EndState::Winner(alive_snake_id.to_string());
        }

        if snakes_remaining == 0 {
            return EndState::TIE;
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
        self.health -= 1
    }

    fn self_collision(&self) -> bool {
        Battlesnake::head_collide_body(&self.head, &self.body)
    }

    fn body_collision(&self, other_snake: &Battlesnake) -> bool {
        Battlesnake::head_collide_body(&self.head, &other_snake.body)
    }

    fn dies_head_to_head(&self, other_snake: &Battlesnake) -> (bool, bool) {
        (
            self.body
                .get(0)
                .unwrap()
                .intersect(other_snake.body.get(0).unwrap()),
            other_snake.body.len() > self.body.len(),
        )
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
