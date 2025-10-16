#[macro_use]
extern crate rocket;
use rocket::info;
use rocket::fairing::AdHoc;
use rocket::http::Status;
use rocket::serde::json::Json;
use serde_json::Value;
use snake_lib::logic;
use snake_lib::models::GameState;
use std::env;

#[get("/")]
fn handle_index() -> Json<Value> {
    Json(logic::info())
}

#[post("/start", format = "json", data = "<start_req>")]
fn handle_start(start_req: Json<GameState>) -> Status {
    logic::start(
        &start_req.game,
        &start_req.turn,
        &start_req.board,
        &start_req.you,
    );

    Status::Ok
}

#[post("/move", format = "json", data = "<move_req>")]
fn handle_move(move_req: Json<GameState>) -> Json<Value> {
    let response = logic::get_move(&move_req.board, &move_req.you);
    Json(response)
}

#[post("/end", format = "json", data = "<end_req>")]
fn handle_end(end_req: Json<GameState>) -> Status {
    logic::end(&end_req.game, &end_req.turn, &end_req.board, &end_req.you);

    Status::Ok
}

#[launch]
fn rocket() -> _ {
    dotenv::dotenv().ok();
    if let Ok(port) = env::var("PORT") {
        env::set_var("ROCKET_PORT", &port);
    }

    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }

    env_logger::init();

    info!("Starting Battlesnake Server...");

    rocket::build()
        .attach(AdHoc::on_response("Server ID Middleware", |_, res| {
            Box::pin(async move {
                res.set_raw_header("Server", "Rusty_Snake");
            })
        }))
        .mount(
            "/",
            routes![handle_index, handle_start, handle_move, handle_end],
        )
}
