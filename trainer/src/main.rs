use snake_lib::{
    learning::simulation::Trainer, test_utils::scenarios::get_board,
};

fn main() {
    let board = get_board();
    let mut trainer = Trainer::new(&board.board);
    trainer.run();
}
