use crate::stats::StatsPanel;
use std::collections::HashMap;
use snake_lib::models::{Coord, GameState};
use yew::{function_component, html, Callback, Html, Properties};

#[derive(Properties, PartialEq)]
pub struct Props {
    pub state: GameState,
    pub snake_urls: HashMap<String, String>,
    pub on_url_change: Callback<(String, String)>,
}

#[function_component(Board)]
pub fn board(props: &Props) -> Html {
    let state = &props.state;
    let board = &state.board;
    let width = board.width;
    let height = board.height;

    // Keep dynamic grid layout inline as it depends on state dimensions
    let grid_style = format!(
        "grid-template-columns: repeat({}, 40px); \
         grid-template-rows: repeat({}, 40px);",
        width, height
    );

    let mut cells = Vec::new();
    // Render rows from top (height-1) to bottom (0)
    for y in (0..height as i32).rev() {
        for x in 0..width as i32 {
            let coord = Coord { x, y };
            
            let mut content = Html::default();
            let mut classes = vec!["cell"];

            // Check for entities at this coordinate
            if board.food.contains(&coord) {
                content = html! { "🍎" };
                classes.push("food");
            } else if board.hazards.contains(&coord) {
                classes.push("hazard");
            }

            for snake in &board.snakes {
                if snake.body.contains(&coord) {
                    let is_you = snake.id == state.you.id;
                    let is_head = snake.head == coord;
                    
                    if is_you {
                        if is_head {
                            classes.push("snake-you-head");
                            content = html! { "👀" };
                        } else {
                            classes.push("snake-you-body");
                        }
                    } else {
                         if is_head {
                            classes.push("snake-enemy-head");
                            content = html! { "👀" };
                        } else {
                            classes.push("snake-enemy-body");
                        }
                    }
                    
                    // Optimization: Break after finding the first snake segment for this cell
                    break;
                }
            }

            cells.push(html! {
                <div class={classes} title={format!("({}, {})", x, y)}>
                    { content }
                </div>
            });
        }
    }

    // Filter enemies
    let enemies: Vec<_> = board.snakes.iter()
        .filter(|s| s.id != state.you.id)
        .cloned()
        .collect();

    html! {
        <div class="main-layout">
            <StatsPanel 
                title="You" 
                snakes={vec![state.you.clone()]} 
                snake_urls={props.snake_urls.clone()}
                on_url_change={props.on_url_change.clone()}
            />

            <div class="center-stage">
                <div class="game-info">
                    <h2>{ format!("Turn: {}", state.turn) }</h2>
                </div>
                
                <div class="grid-container" style={grid_style}>
                    { for cells }
                </div>
            </div>

            <StatsPanel 
                title="Enemies" 
                snakes={enemies} 
                is_enemy={true} 
                snake_urls={props.snake_urls.clone()}
                on_url_change={props.on_url_change.clone()}
            />
        </div>
    }
}
