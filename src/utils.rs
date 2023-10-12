pub const DIRECTIONS: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

pub fn dir_to_string(tup: (i32, i32)) -> String {
    if tup == (-1, 0) {
        return "down".to_owned();
    }
    if tup == (0, 1) {
        return "right".to_owned();
    }
    if tup == (0, -1) {
        return "left".to_owned();
    }
    if tup == (1, 0) {
        return "up".to_owned();
    }
    panic!("invalid direction");
}
