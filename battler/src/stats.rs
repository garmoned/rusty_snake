use std::collections::HashMap;
use snake_lib::models::Battlesnake;
use web_sys::HtmlInputElement;
use yew::{function_component, html, Callback, Html, InputEvent, Properties, TargetCast};

#[derive(Properties, PartialEq)]
pub struct SnakeCardProps {
    pub snake: Battlesnake,
    pub is_enemy: bool,
    pub url: String,
    pub on_url_change: Callback<String>,
}

#[function_component(SnakeCard)]
pub fn snake_card(props: &SnakeCardProps) -> Html {
    let class_name = if props.is_enemy { "snake-card enemy" } else { "snake-card you" };
    
    let oninput = {
        let on_url_change = props.on_url_change.clone();
        Callback::from(move |e: InputEvent| {
            let input: HtmlInputElement = e.target_unchecked_into();
            on_url_change.emit(input.value());
        })
    };

    html! {
        <div class={class_name}>
            <div class="snake-name">{ &props.snake.name }</div>
            <div class="snake-stat">{ format!("Health: {}", props.snake.health) }</div>
            <div class="snake-stat">{ format!("Length: {}", props.snake.length) }</div>
            <div class="snake-url">
                <label>{ "URL:" }</label>
                <input 
                    type="text" 
                    value={props.url.clone()} 
                    {oninput}
                    placeholder="http://localhost:8000"
                />
            </div>
        </div>
    }
}

#[derive(Properties, PartialEq)]
pub struct StatsPanelProps {
    pub title: String,
    pub snakes: Vec<Battlesnake>,
    #[prop_or_default]
    pub is_enemy: bool,
    pub snake_urls: HashMap<String, String>,
    pub on_url_change: Callback<(String, String)>,
}

#[function_component(StatsPanel)]
pub fn stats_panel(props: &StatsPanelProps) -> Html {
    let class_name = if props.is_enemy { "stats-panel right" } else { "stats-panel left" };
    
    html! {
        <div class={class_name}>
            <h3>{ &props.title }</h3>
            {
                for props.snakes.iter().map(|snake| {
                    let snake_id = snake.id.clone();
                    let url = props.snake_urls.get(&snake_id).cloned().unwrap_or_default();
                    let on_url_change = props.on_url_change.clone();
                    
                    let on_card_url_change = Callback::from(move |new_url| {
                        on_url_change.emit((snake_id.clone(), new_url));
                    });

                    html! {
                        <SnakeCard 
                            snake={snake.clone()} 
                            is_enemy={props.is_enemy}
                            url={url}
                            on_url_change={on_card_url_change}
                        />
                    }
                })
            }
        </div>
    }
}
