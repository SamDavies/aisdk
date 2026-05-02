//! Language model implementation for the OpenAI provider.

use crate::core::capabilities::ModelName;
use crate::core::client::LanguageModelClient;
use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream, Usage,
};
use crate::core::messages::AssistantMessage;
use crate::providers::openai::client::{OpenAILanguageModelOptions, types};
use crate::providers::openai::{OpenAI, client};
use crate::{
    core::{language_model::LanguageModel, tools::ToolCallInfo},
    error::Result,
};
use async_trait::async_trait;
use futures::StreamExt;

#[async_trait]
impl<M: ModelName> LanguageModel for OpenAI<M> {
    /// Returns the name of the model.
    fn name(&self) -> String {
        self.lm_options.model.clone()
    }

    /// Generates text using the OpenAI provider.
    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let mut options: OpenAILanguageModelOptions = options.into();

        options.model = self.lm_options.model.clone();

        self.lm_options = options;

        let response: client::OpenAIResponse = self.send(&self.settings.base_url).await?;

        let mut collected: Vec<LanguageModelResponseContentType> = Vec::new();

        for out in response.output.unwrap_or_default() {
            match out {
                types::MessageItem::OutputMessage { content, .. } => {
                    for c in content {
                        if let types::OutputContent::OutputText { text, .. } = c {
                            collected.push(LanguageModelResponseContentType::new(text))
                        }
                    }
                }
                types::MessageItem::FunctionCall {
                    arguments,
                    name,
                    call_id,
                    ..
                } => {
                    let mut tool_info = ToolCallInfo::new(name);
                    tool_info.id(call_id);
                    tool_info.input(serde_json::from_str(&arguments).unwrap_or_default());
                    collected.push(LanguageModelResponseContentType::ToolCall(tool_info));
                }
                _ => (),
            }
        }

        Ok(LanguageModelResponse {
            contents: collected,
            usage: response.usage.map(|usage| usage.into()),
        })
    }

    /// Streams text using the OpenAI provider.
    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let mut options: OpenAILanguageModelOptions = options.into();

        options.model = self.lm_options.model.to_string();
        options.stream = Some(true);

        self.lm_options = options;

        // Retry logic for rate limiting
        let max_retries = 5;
        let mut retry_count = 0;
        let mut wait_time = std::time::Duration::from_secs(1);

        let openai_stream = loop {
            match self.send_and_stream(&self.settings.base_url).await {
                Ok(stream) => break stream,
                Err(crate::error::Error::ApiError {
                    status_code: Some(status),
                    ..
                }) if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    && retry_count < max_retries =>
                {
                    retry_count += 1;
                    tokio::time::sleep(wait_time).await;
                    wait_time *= 2; // Exponential backoff
                    continue;
                }
                Err(e) => return Err(e),
            }
        };

        let stream = openai_stream.map(|evt_res| match evt_res {
            Ok(client::OpenAiStreamEvent::ResponseOutputTextDelta { delta, .. }) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Text(delta),
                )])
            }
            Ok(client::OpenAiStreamEvent::ResponseReasoningSummaryTextDelta { delta, .. }) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Reasoning(delta),
                )])
            }
            Ok(client::OpenAiStreamEvent::ResponseCompleted { response, .. }) => {
                let mut result: Vec<LanguageModelStreamChunk> = Vec::new();

                let usage: Usage = response.usage.unwrap_or_default().into();
                let output = response.output.unwrap_or_default();

                for msg in output {
                    match &msg {
                        // ---- Final OutputMessage ----
                        types::MessageItem::OutputMessage { content, .. } => {
                            if let Some(types::OutputContent::OutputText {
                                text,
                                annotations,
                                ..
                            }) = content.first()
                            {
                                let combined =
                                    append_sources_footer(text.clone(), annotations.as_slice());
                                if combined.len() > text.len() {
                                    let footer = combined[text.len()..].to_string();
                                    result.push(LanguageModelStreamChunk::Delta(
                                        LanguageModelStreamChunkType::Text(footer),
                                    ));
                                }
                                result.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::new(combined),
                                    usage: Some(usage.clone()),
                                }));
                            }
                        }

                        // ---- Reasoning ----
                        types::MessageItem::Reasoning { summary, .. } => {
                            if let Some(types::ReasoningSummary { text, .. }) = summary.first() {
                                result.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Reasoning {
                                        content: text.to_owned(),
                                        extensions: crate::extensions::Extensions::default(),
                                    },
                                    usage: Some(usage.clone()),
                                }));
                            }
                        }

                        // ---- FunctionCall ----
                        types::MessageItem::FunctionCall {
                            call_id,
                            name,
                            arguments,
                            ..
                        } => {
                            let mut tool_info = ToolCallInfo::new(name.clone());
                            tool_info.id(call_id.clone());
                            tool_info.input(serde_json::from_str(arguments).unwrap_or_default());

                            result.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                content: LanguageModelResponseContentType::ToolCall(tool_info),
                                usage: Some(usage.clone()),
                            }));
                        }

                        _ => {}
                    }
                }

                Ok(result)
            }
            Ok(client::OpenAiStreamEvent::ResponseIncomplete { response, .. }) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Incomplete(
                        response
                            .incomplete_details
                            .map(|d| d.reason)
                            .unwrap_or("Unknown".to_string()),
                    ),
                )])
            }
            Ok(client::OpenAiStreamEvent::ResponseError { code, message, .. }) => {
                let reason = format!("{}: {}", code.unwrap_or("unknown".to_string()), message);
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Failed(reason),
                )])
            }
            Ok(evt) => Ok(vec![LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::NotSupported(format!("{evt:?}")),
            )]),
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}

/// Format a Sources footer (deduplicated by URL, preserving order) from
/// Responses-API output annotations.
fn format_sources_footer(urls: &[String]) -> String {
    if urls.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n\nSources:");
    for url in urls {
        out.push_str("\n- ");
        out.push_str(url);
    }
    out
}

/// Append a Sources footer to a body when annotations contain url citations.
fn append_sources_footer(text: String, annotations: &[types::OutputTextAnnotation]) -> String {
    let mut seen: Vec<String> = Vec::new();
    for ann in annotations {
        if let types::OutputTextAnnotation::UrlCitation { url, .. } = ann
            && !seen.contains(url)
        {
            seen.push(url.clone());
        }
    }
    if seen.is_empty() {
        return text;
    }
    let mut combined = text;
    combined.push_str(&format_sources_footer(&seen));
    combined
}

#[cfg(test)]
mod sources_tests {
    use super::*;

    #[test]
    fn empty_returns_empty() {
        assert_eq!(format_sources_footer(&[]), "");
    }

    #[test]
    fn single_url() {
        let out = format_sources_footer(&["https://example.com".into()]);
        assert_eq!(out, "\n\nSources:\n- https://example.com");
    }

    #[test]
    fn dedupes_and_preserves_order() {
        let anns = vec![
            types::OutputTextAnnotation::UrlCitation {
                start_index: 0,
                end_index: 1,
                url: "https://a.com".into(),
                title: "A".into(),
            },
            types::OutputTextAnnotation::UrlCitation {
                start_index: 2,
                end_index: 3,
                url: "https://b.com".into(),
                title: "B".into(),
            },
            types::OutputTextAnnotation::UrlCitation {
                start_index: 4,
                end_index: 5,
                url: "https://a.com".into(),
                title: "A2".into(),
            },
        ];
        let out = append_sources_footer("body".into(), &anns);
        assert_eq!(out, "body\n\nSources:\n- https://a.com\n- https://b.com");
    }

    #[test]
    fn no_annotations_returns_text_unchanged() {
        assert_eq!(append_sources_footer("body".into(), &[]), "body");
    }

    #[test]
    fn ignores_non_url_annotations() {
        let anns = vec![types::OutputTextAnnotation::FileCitation {
            file_id: "f".into(),
            filename: "f.txt".into(),
            index: 0,
        }];
        assert_eq!(append_sources_footer("body".into(), &anns), "body");
    }
}
