
use snake_lib::{
    learning::simulation::{RunMode, Trainer, TrainerConfig},
    test_utils::scenarios::get_board,
};

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() != 3 {
        eprintln!("Usage: {} <batch_size> <batches> <run_mode>", args[0]);
        std::process::exit(1);
    }

    let batch_size = args[1].parse::<u64>().unwrap();
    let batches = args[2].parse::<u64>().unwrap();
    let run_mode = args[3].parse::<RunMode>().unwrap();

    let config = TrainerConfig {
        batch_size,
        batches,
        run_mode,
    };

    let board = get_board();
    let mut trainer = Trainer::new(&board.board, config);
    trainer.run();
}
