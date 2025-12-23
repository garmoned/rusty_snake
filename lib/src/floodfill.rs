use crate::{
    models::{Board, Coord},
    utils,
};
use std::{collections::VecDeque, convert::TryInto};

// TODO: Optimize this to floodfill for each snake
// so we don't have to floodfill multiple times per scoring
pub fn floodfill(board: &Board, snake_id: &str) -> usize {
    let mut filled_count = 0;
    let mut barriers = vec![
        vec![false; board.width.try_into().unwrap()];
        board.height.try_into().unwrap()
    ];
    let mut filled_tiles = vec![
        vec![false; board.width.try_into().unwrap()];
        board.height.try_into().unwrap()
    ];
    let mut target_snake = board.snakes.first().unwrap();
    for snake in &board.snakes {
        if snake.id == snake_id {
            target_snake = snake;
        }
        for coord in &snake.body {
            barriers[coord.x()][coord.y()] = true
        }
    }
    let mut q = VecDeque::<Coord>::new();
    let start_head = target_snake.body.first().unwrap();
    q.push_back(start_head.clone());
    while !q.is_empty() {
        let expand_from = q.pop_front().unwrap();
        for (x, y) in utils::DIRECTIONS {
            let mut new_explore = Coord::default();
            new_explore.x = expand_from.x + x;
            new_explore.y = expand_from.y + y;
            // If the space is empty, in bounds, and unexplored mark it as accessible
            // and push it to be explored further from
            if new_explore.in_bounds(board.width(), board.height())
                && !barriers[new_explore.x()][new_explore.y()]
                && !filled_tiles[new_explore.x()][new_explore.y()]
            {
                q.push_back(new_explore.clone());
                filled_tiles[new_explore.x()][new_explore.y()] = true;
                filled_count += 1;
            }
        }
    }

    // Body in the fill to avoid getting punished for getting bigger.
    filled_count + target_snake.body.len()
}

#[cfg(test)]
mod test {

    use super::floodfill;
    use crate::test_utils::scenarios::get_board;

    #[test]
    fn test_flood_fill() {
        let game_state = get_board().board;
        assert_eq!(floodfill(&game_state, "long_snake"), 118);
        assert_eq!(floodfill(&game_state, "short_snake"), 118);
    }
}
