use std::task::{Context, Poll};

use hyper::{Body, HeaderMap};
use opentelemetry::propagation::Extractor;
use tonic::body::BoxBody;
use tower::{Layer, Service};
use tracing::{field, info_span, Instrument, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[derive(Debug, Clone, Default)]
pub struct TraceLayer;

impl<S> Layer<S> for TraceLayer {
    type Service = TraceMiddleware<S>;

    fn layer(&self, service: S) -> Self::Service {
        TraceMiddleware { inner: service }
    }
}

#[derive(Debug, Clone)]
pub struct TraceMiddleware<S> {
    inner: S,
}

impl<S> Service<hyper::Request<Body>> for TraceMiddleware<S>
where
    S: Service<hyper::Request<Body>, Response = hyper::Response<BoxBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = futures::future::BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: hyper::Request<Body>) -> Self::Future {
        // This is necessary because tonic internally uses `tower::buffer::Buffer`.
        // See https://github.com/tower-rs/tower/issues/547#issuecomment-767629149
        // for details on why this is necessary
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);

        let path = req.uri().path().trim_start_matches('/');
        let (service, method) = path.split_once('/').unwrap();

        let parent_context = opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.extract(&RequestHeaderCarrier::new(req.headers()))
        });

        let span = if service.starts_with("grpc.health") {
            Span::none()
        } else {
            info_span!(
                "request",
                "otel.name" = path,
                "otel.service" = service,
                "rpc.method" = method,
                "rpc.system" = "grpc",
                "otel.kind" = "server",
                "rpc.grpc.status_code" = field::Empty,
                "otel.status_code" = field::Empty,
            )
        };

        span.set_parent(parent_context);

        Box::pin(
            async move {
                let response = inner.call(req).await?;

                let grpc_status = response
                    .headers()
                    .get("grpc-status")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("0");

                let span = Span::current();
                span.record("rpc.grpc.status_code", grpc_status);
                if grpc_status != "0" {
                    span.record("otel.status_code", "error");
                }

                Ok(response)
            }
            .instrument(span),
        )
    }
}

struct RequestHeaderCarrier<'a> {
    headers: &'a HeaderMap,
}

impl<'a> RequestHeaderCarrier<'a> {
    fn new(headers: &'a HeaderMap) -> Self {
        Self { headers }
    }
}

impl<'a> Extractor for RequestHeaderCarrier<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.headers.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.headers.keys().map(|header| header.as_str()).collect()
    }
}
