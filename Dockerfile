FROM rust:1.66.1 AS builder
WORKDIR /app

RUN curl -sSL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-static-with-deps-1.13.0%2Bcpu.zip -o libtorch.zip
RUN unzip libtorch.zip
ENV LIBTORCH=/app/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

COPY . .
RUN cargo build --release

FROM gcr.io/distroless/cc
COPY --from=builder /app/libtorch /libtorch
ENV LD_LIBRARY_PATH=/libtorch/lib:$LD_LIBRARY_PATH
COPY --from=builder /app/target/release/trast /app
CMD ["/app"]
