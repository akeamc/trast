FROM clux/muslrust:1.67.0 as build-env
RUN ln -s /usr/bin/g++ /usr/bin/musl-g++
WORKDIR /app
COPY . .
RUN cargo build -p trast --release

FROM gcr.io/distroless/static
COPY --from=build-env /app/target/x86_64-unknown-linux-musl/release/trast /trast
EXPOSE 8000
CMD ["/trast"]
