mod board;
mod stats;

use board::Board;
use snake_lib::test_utils;
use std::collections::HashMap;
use yew::prelude::*;

#[function_component(App)]
fn app() -> Html {
    let mock_state = test_utils::scenarios::get_scenario(
        test_utils::scenarios::AVOID_SELF_TRAP,
    );
    
    let snake_urls = use_state(|| HashMap::new());
    
    let on_url_change = {
        let snake_urls = snake_urls.clone();
        Callback::from(move |(id, url): (String, String)| {
            let mut new_urls = (*snake_urls).clone();
            new_urls.insert(id, url);
            snake_urls.set(new_urls);
        })
    };

    html! {
        <Board 
            state={mock_state} 
            snake_urls={(*snake_urls).clone()}
            on_url_change={on_url_change}
        />
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
