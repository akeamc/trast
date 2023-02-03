FROM clux/muslrust:1.67.0 as build-env
RUN GRPC_HEALTH_PROBE_VERSION=v0.4.13 && \
    curl -sLo /bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64
RUN ln -s /usr/bin/g++ /usr/bin/musl-g++
WORKDIR /app
COPY . .
RUN cargo build -p trast --release

FROM gcr.io/distroless/static
COPY --from=build-env --chmod=+x /bin/grpc_health_probe /bin/grpc_health_probe
COPY --from=build-env /app/target/x86_64-unknown-linux-musl/release/trast /trast
EXPOSE 8000
CMD ["/trast"]
