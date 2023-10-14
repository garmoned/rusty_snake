FROM rust:1.67

WORKDIR /usr/src/rusty_snake
COPY . .

RUN cargo install --path .

CMD ["rusty_snake"]