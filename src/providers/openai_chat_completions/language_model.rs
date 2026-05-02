//! Language model implementation for the OpenAI Chat Completions provider.

use crate::core::capabilities::ModelName;
use crate::core::client::LanguageModelClient;
use crate::core::language_model::{
    LanguageModel, LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream,
};
use crate::core::messages::AssistantMessage;
use crate::core::tools::ToolCallInfo;
use crate::error::Result;
use crate::providers::openai_chat_completions::OpenAIChatCompletions;
use crate::providers::openai_chat_completions::client::{self, types};
use async_trait::async_trait;
use futures::StreamExt;

#[async_trait]
impl<M: ModelName> LanguageModel for OpenAIChatCompletions<M> {
    fn name(&self) -> String {
        self.options.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let mut options: client::ChatCompletionsOptions = options.into();
        options.model = self.options.model.clone();
        self.options = options;

        let response: types::ChatCompletionsResponse = self.send(&self.settings.base_url).await?;

        // Convert choices to LanguageModelResponse
        let mut contents = Vec::new();

        for choice in response.choices {
            // Handle text content (with optional Sources footer when the
            // model surfaced web-search citations).
            if let Some(text) = choice.message.content {
                let combined = append_sources_footer(text, choice.message.annotations.as_deref());
                if !combined.is_empty() {
                    contents.push(LanguageModelResponseContentType::Text(combined));
                }
            }

            // Handle tool calls
            if let Some(tool_calls) = choice.message.tool_calls {
                for tool_call in tool_calls {
                    let mut tool_info = ToolCallInfo::new(tool_call.function.name);
                    tool_info.id(tool_call.id);
                    tool_info.input(
                        serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                    );
                    contents.push(LanguageModelResponseContentType::ToolCall(tool_info));
                }
            }
        }

        Ok(LanguageModelResponse {
            contents,
            usage: response.usage.map(|u| u.into()),
        })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let mut options: client::ChatCompletionsOptions = options.into();
        options.model = self.options.model.clone();
        options.stream = Some(true);
        // Enable streamed usage reporting. Without this, OpenAI/OpenRouter do
        // not emit the final usage chunk on streams. OpenAI ignores the
        // top-level `usage` field; OpenRouter uses it to include exact `cost`.
        options.stream_options = Some(types::StreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        });
        self.options = options;

        let stream = self.send_and_stream(&self.settings.base_url).await?;

        // State for accumulating tool calls across chunks
        use std::collections::HashMap;
        let mut accumulated_tool_calls: HashMap<u32, (String, String, String)> = HashMap::new();
        // Web-search citations arrive as separate annotation deltas; collect
        // them and emit a "Sources:" footer as a final text delta before the
        // terminating Done chunk.
        let mut accumulated_citations: Vec<String> = Vec::new();

        // Map stream events to SDK stream chunks
        let stream = stream.map(move |evt_res| match evt_res {
            Ok(types::ChatCompletionsStreamEvent::Chunk(chunk)) => {
                let mut results = Vec::new();
                let mut emitted_done_with_usage = false;

                for choice in chunk.choices {
                    // Reasoning delta (for reasoning models like o1, DeepSeek R1)
                    if let Some(reasoning) = choice.delta.reasoning_content
                        && !reasoning.is_empty()
                    {
                        results.push(LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Reasoning(reasoning),
                        ));
                    }

                    // Text delta
                    if let Some(content) = choice.delta.content
                        && !content.is_empty()
                    {
                        results.push(LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Text(content),
                        ));
                    }

                    // Web-search citation deltas. Deduplicate by URL — OpenAI
                    // sometimes repeats the same annotation across consecutive
                    // chunks while extending the cited text range.
                    if let Some(annotations) = choice.delta.annotations {
                        for annotation in annotations {
                            if let types::Annotation::UrlCitation { url_citation } = annotation
                                && !accumulated_citations.contains(&url_citation.url)
                            {
                                accumulated_citations.push(url_citation.url);
                            }
                        }
                    }

                    // Accumulate tool call deltas
                    if let Some(tool_calls) = choice.delta.tool_calls {
                        for tool_call in tool_calls {
                            let entry = accumulated_tool_calls.entry(tool_call.index).or_insert((
                                String::new(),
                                String::new(),
                                String::new(),
                            ));

                            // Accumulate ID
                            if let Some(id) = tool_call.id {
                                entry.0 = id;
                            }

                            // Accumulate name and arguments
                            if let Some(function) = tool_call.function {
                                if let Some(name) = function.name {
                                    entry.1 = name;
                                }
                                if let Some(args) = function.arguments {
                                    entry.2.push_str(&args);
                                    results.push(LanguageModelStreamChunk::Delta(
                                        LanguageModelStreamChunkType::ToolCall(args),
                                    ));
                                }
                            }
                        }
                    }

                    if let Some(finish_reason) = choice.finish_reason {
                        let usage = chunk.usage.clone().map(|u| u.into());
                        if usage.is_some() {
                            emitted_done_with_usage = true;
                        }

                        // Emit Sources footer as a final text delta, just
                        // before the Done that terminates this turn.
                        if matches!(finish_reason.as_str(), "stop" | "length")
                            && !accumulated_citations.is_empty()
                        {
                            let footer = format_sources_footer(&accumulated_citations);
                            accumulated_citations.clear();
                            results.push(LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::Text(footer),
                            ));
                        }

                        match finish_reason.as_str() {
                            "stop" | "length" => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                            }
                            "tool_calls" | "function_call" => {
                                // Send accumulated tool calls
                                for (id, name, args) in accumulated_tool_calls.values() {
                                    let mut tool_info = ToolCallInfo::new(name.clone());
                                    tool_info.id(id.clone());
                                    tool_info.input(serde_json::from_str(args).unwrap_or_else(
                                        |_| serde_json::Value::Object(serde_json::Map::new()),
                                    ));
                                    results.push(LanguageModelStreamChunk::Done(
                                        AssistantMessage {
                                            content: LanguageModelResponseContentType::ToolCall(
                                                tool_info,
                                            ),
                                            usage: usage.clone(),
                                        },
                                    ));
                                }
                            }
                            "content_filter" => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                                results.push(LanguageModelStreamChunk::Delta(
                                    LanguageModelStreamChunkType::Failed(
                                        "Content filtered".to_string(),
                                    ),
                                ));
                            }
                            // For any unknown finish reason, treat as normal completion
                            _ => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                            }
                        }
                    }
                }

                // OpenAI/OpenRouter streams (with stream_options.include_usage = true)
                // emit the final usage in a trailing chunk that has `choices: []`
                // and `usage: Some(...)`. The per-choice loop above never runs for
                // such a chunk, so without this the exact prompt/completion/cached
                // token counts and OpenRouter's `cost` field are silently dropped.
                // Emit a synthetic Done carrying just the usage so downstream
                // accumulation (`StreamTextResponse::usage()`) sees it.
                if !emitted_done_with_usage
                    && let Some(usage) = chunk.usage.clone().map(|u| u.into())
                {
                    results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                        content: LanguageModelResponseContentType::Text(String::new()),
                        usage: Some(usage),
                    }));
                }

                Ok(results)
            }
            Ok(types::ChatCompletionsStreamEvent::Open) => Ok(vec![]),
            Ok(types::ChatCompletionsStreamEvent::Done) => Ok(vec![]),
            Ok(types::ChatCompletionsStreamEvent::Error(e)) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Failed(e),
                )])
            }
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}

/// Format a list of cited URLs as a markdown footer prefixed by a blank line.
/// Returns empty string when the slice is empty.
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

/// Append a Sources footer (deduplicated by URL) to a text body when the
/// supplied annotations contain url citations. No-op when none are present.
fn append_sources_footer(text: String, annotations: Option<&[types::Annotation]>) -> String {
    let urls: Vec<String> = annotations
        .map(|anns| {
            let mut seen: Vec<String> = Vec::new();
            for ann in anns {
                if let types::Annotation::UrlCitation { url_citation } = ann
                    && !seen.contains(&url_citation.url)
                {
                    seen.push(url_citation.url.clone());
                }
            }
            seen
        })
        .unwrap_or_default();

    if urls.is_empty() {
        return text;
    }
    let mut combined = text;
    combined.push_str(&format_sources_footer(&urls));
    combined
}

#[cfg(test)]
mod sources_tests {
    use super::*;

    #[test]
    fn footer_empty_when_no_urls() {
        assert_eq!(format_sources_footer(&[]), "");
    }

    #[test]
    fn footer_single_url() {
        let urls = vec!["https://example.com".to_string()];
        assert_eq!(
            format_sources_footer(&urls),
            "\n\nSources:\n- https://example.com"
        );
    }

    #[test]
    fn footer_multiple_urls() {
        let urls = vec!["https://a.com".to_string(), "https://b.com".to_string()];
        assert_eq!(
            format_sources_footer(&urls),
            "\n\nSources:\n- https://a.com\n- https://b.com"
        );
    }

    #[test]
    fn append_passes_through_when_no_annotations() {
        assert_eq!(append_sources_footer("body".into(), None), "body");
    }

    #[test]
    fn append_dedups_by_url() {
        let anns = vec![
            types::Annotation::UrlCitation {
                url_citation: types::UrlCitation {
                    url: "https://a.com".into(),
                    title: None,
                    start_index: None,
                    end_index: None,
                },
            },
            types::Annotation::UrlCitation {
                url_citation: types::UrlCitation {
                    url: "https://a.com".into(),
                    title: Some("A".into()),
                    start_index: Some(0),
                    end_index: Some(5),
                },
            },
            types::Annotation::UrlCitation {
                url_citation: types::UrlCitation {
                    url: "https://b.com".into(),
                    title: None,
                    start_index: None,
                    end_index: None,
                },
            },
        ];
        assert_eq!(
            append_sources_footer("body".into(), Some(&anns)),
            "body\n\nSources:\n- https://a.com\n- https://b.com"
        );
    }
}
