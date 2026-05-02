//! Client implementation for the OpenAI Chat Completions API.

pub(crate) mod types;

pub(crate) use types::ChatCompletionsOptions;

use crate::core::capabilities::ModelName;
use crate::core::client::LanguageModelClient;
use crate::error::Error;
use crate::providers::openai_chat_completions::OpenAIChatCompletions;
use reqwest::header::CONTENT_TYPE;
use reqwest_eventsource::Event;
use types::*;

impl<M: ModelName> LanguageModelClient for OpenAIChatCompletions<M> {
    type Response = ChatCompletionsResponse;
    type StreamEvent = ChatCompletionsStreamEvent;

    fn path(&self) -> String {
        self.settings
            .path
            .clone()
            .unwrap_or_else(|| "chat/completions".to_string())
    }

    fn method(&self) -> reqwest::Method {
        reqwest::Method::POST
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.settings.api_key).parse().unwrap(),
        );
        headers
    }

    fn query_params(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn body(&self) -> reqwest::Body {
        let body = serde_json::to_string(&self.options).unwrap();
        // Diagnostic: when DEBUG_OPENAI_CHAT_BODY=1 is set, dump the
        // serialised request body to a tmp file so the operator can inspect
        // exactly what was sent on a 4xx. The file is overwritten on every
        // request — only useful for one-shot reproductions.
        if std::env::var("DEBUG_OPENAI_CHAT_BODY").as_deref() == Ok("1") {
            let _ = std::fs::write("/tmp/openai-chat-last-body.json", &body);
        }
        reqwest::Body::from(body)
    }

    fn parse_stream_sse(
        event: std::result::Result<Event, reqwest_eventsource::Error>,
    ) -> crate::error::Result<Self::StreamEvent> {
        match event {
            Ok(event) => match event {
                Event::Open => Ok(ChatCompletionsStreamEvent::Open),
                Event::Message(msg) => {
                    if msg.data.trim() == "[DONE]" || msg.data.is_empty() {
                        return Ok(ChatCompletionsStreamEvent::Done);
                    }

                    let chunk: ChatCompletionsStreamChunk = serde_json::from_str(&msg.data)
                        .map_err(|e| Error::ApiError {
                            status_code: None,
                            details: format!("Invalid JSON in SSE: {e}"),
                        })?;

                    Ok(ChatCompletionsStreamEvent::Chunk(chunk))
                }
            },
            Err(e) => {
                let status_code = match &e {
                    reqwest_eventsource::Error::InvalidStatusCode(status, _) => Some(*status),
                    _ => None,
                };
                Err(Error::ApiError {
                    status_code,
                    details: e.to_string(),
                })
            }
        }
    }

    fn end_stream(event: &Self::StreamEvent) -> bool {
        matches!(event, ChatCompletionsStreamEvent::Done)
    }
}
