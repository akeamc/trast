ARG RUST_VERSION=1.67.0

FROM clux/muslrust:$RUST_VERSION AS planner
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM clux/muslrust:$RUST_VERSION AS cacher
RUN cargo install cargo-chef
COPY --from=planner /volume/recipe.json recipe.json
RUN cargo chef cook --release --target x86_64-unknown-linux-musl --recipe-path recipe.json

FROM clux/muslrust:$RUST_VERSION AS builder
RUN GRPC_HEALTH_PROBE_VERSION=v0.4.13 && \
    curl -sLo /bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe
RUN ln -s /usr/bin/g++ /usr/bin/musl-g++
COPY . .
COPY --from=cacher /volume/target target
COPY --from=cacher /root/.cargo /root/.cargo
RUN cargo build -p trast --release

FROM gcr.io/distroless/static
COPY --from=builder /bin/grpc_health_probe /bin/grpc_health_probe
COPY --from=builder /volume/target/x86_64-unknown-linux-musl/release/trast /trast
EXPOSE 8000
CMD ["/trast"]
