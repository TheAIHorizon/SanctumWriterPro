import { LLMMessage, LLMProviderType, StreamCallbacks, isCloudProvider } from '@/types';
import { documentTools, formatToolsForOllama, formatToolsForLMStudio, formatToolsForOpenAI } from './tools';

const OLLAMA_URL = 'http://localhost:11434';
const LMSTUDIO_URL = 'http://localhost:1234';

// Cloud provider base URLs
const CLOUD_URLS = {
  openrouter: 'https://openrouter.ai/api/v1',
  openai: 'https://api.openai.com/v1',
  anthropic: 'https://api.anthropic.com/v1',
  google: 'https://generativelanguage.googleapis.com/v1beta',
  xai: 'https://api.x.ai/v1',
};

// Default cap on generated output tokens. This is intentionally decoupled
// from contextLength: contextLength describes how much input the model can
// see (used for num_ctx / input budgeting), while max_tokens caps how much
// the model is allowed to *generate*. Conflating the two breaks providers
// with very large context windows (e.g. Anthropic models with 200K context
// would otherwise be asked to generate 200K output tokens, which the API
// rejects).
const DEFAULT_MAX_OUTPUT_TOKENS = 4096;

export interface LLMOptions {
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  contextLength?: number;
  /** Max tokens the model is allowed to generate in its response. */
  maxTokens?: number;
}

// Rough token estimation (4 chars per token on average)
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export function estimateMessagesTokens(messages: LLMMessage[]): number {
  return messages.reduce((total, msg) => total + estimateTokens(msg.content), 0);
}

export async function getAvailableModels(provider: LLMProviderType): Promise<Array<{ id: string; name: string }>> {
  try {
    if (provider === 'ollama') {
      const response = await fetch(`${OLLAMA_URL}/api/tags`);
      if (!response.ok) throw new Error('Failed to fetch Ollama models');
      const data = await response.json();
      return (data.models || []).map((m: { name: string }) => ({
        id: m.name,
        name: m.name,
      }));
    } else {
      const response = await fetch(`${LMSTUDIO_URL}/v1/models`);
      if (!response.ok) throw new Error('Failed to fetch LM Studio models');
      const data = await response.json();
      return (data.data || []).map((m: { id: string }) => ({
        id: m.id,
        name: m.id,
      }));
    }
  } catch (error) {
    console.error('Error fetching models:', error);
    return [];
  }
}

export async function checkProviderHealth(provider: LLMProviderType): Promise<boolean> {
  try {
    const url = provider === 'ollama' ? `${OLLAMA_URL}/api/tags` : `${LMSTUDIO_URL}/v1/models`;
    const response = await fetch(url, { method: 'GET' });
    return response.ok;
  } catch {
    return false;
  }
}

export async function streamChat(
  provider: LLMProviderType,
  model: string,
  messages: LLMMessage[],
  callbacks: StreamCallbacks,
  useTools: boolean = true,
  options: LLMOptions = {},
  apiKey?: string
): Promise<void> {
  try {
    if (provider === 'ollama') {
      await streamOllamaChat(model, messages, callbacks, useTools, options);
    } else if (provider === 'lmstudio') {
      await streamLMStudioChat(model, messages, callbacks, useTools, options);
    } else if (isCloudProvider(provider)) {
      // Cloud providers require API key
      if (!apiKey) {
        throw new Error(`API key required for ${provider}`);
      }
      await streamCloudChat(provider, model, messages, callbacks, useTools, options, apiKey);
    } else {
      throw new Error(`Unknown provider: ${provider}`);
    }
  } catch (error) {
    callbacks.onError(error instanceof Error ? error : new Error('Unknown error'));
  }
}

async function streamOllamaChat(
  model: string,
  messages: LLMMessage[],
  callbacks: StreamCallbacks,
  useTools: boolean,
  options: LLMOptions = {}
): Promise<void> {
  // Try with tools first, fall back to without if model doesn't support it
  let response = await tryOllamaRequest(model, messages, useTools, options);
  
  // If 400 error and we were using tools, retry without tools
  if (!response.ok && response.status === 400 && useTools) {
    console.log('Model does not support tools, retrying without...');
    response = await tryOllamaRequest(model, messages, false, options);
  }

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ollama error: ${response.status} - ${errorText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      
      try {
        const data = JSON.parse(line);
        
        if (data.message?.content) {
          callbacks.onToken(data.message.content);
        }
        
        if (data.message?.tool_calls) {
          for (const toolCall of data.message.tool_calls) {
            callbacks.onToolCall({
              name: toolCall.function.name,
              arguments: typeof toolCall.function.arguments === 'string'
                ? JSON.parse(toolCall.function.arguments)
                : toolCall.function.arguments,
            });
          }
        }
        
        if (data.done) {
          callbacks.onComplete();
        }
      } catch (e) {
        console.error('Error parsing Ollama response:', e);
      }
    }
  }
}

async function tryOllamaRequest(
  model: string,
  messages: LLMMessage[],
  useTools: boolean,
  options: LLMOptions = {}
): Promise<Response> {
  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    options: {
      temperature: options.temperature ?? 0.7,
      top_p: options.topP ?? 0.9,
      top_k: options.topK ?? 40,
      repeat_penalty: options.repeatPenalty ?? 1.1,
      num_ctx: options.contextLength ?? 4096,
    },
  };

  if (useTools) {
    body.tools = formatToolsForOllama(documentTools);
  }

  return fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function streamLMStudioChat(
  model: string,
  messages: LLMMessage[],
  callbacks: StreamCallbacks,
  useTools: boolean,
  options: LLMOptions = {}
): Promise<void> {
  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    temperature: options.temperature ?? 0.7,
    top_p: options.topP ?? 0.9,
    max_tokens: options.maxTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
    frequency_penalty: (options.repeatPenalty ?? 1.1) - 1, // Convert to OpenAI format
  };

  // LM Studio may not support tools, so we'll try without them by default
  // and only add them if explicitly requested and the model supports it
  if (useTools) {
    body.tools = formatToolsForLMStudio(documentTools);
    body.tool_choice = 'auto';
  }

  let response = await fetch(`${LMSTUDIO_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  // Retry without tools if error
  if (!response.ok && useTools) {
    const retryBody = { model, messages, stream: true };
    response = await fetch(`${LMSTUDIO_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(retryBody),
    });
  }

  if (!response.ok) {
    throw new Error(`LM Studio error: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';
  let toolCallBuffer: { name: string; arguments: string } | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim() || line.startsWith(':')) continue;
      if (line === 'data: [DONE]') {
        if (toolCallBuffer && toolCallBuffer.arguments) {
          try {
            callbacks.onToolCall({
              name: toolCallBuffer.name,
              arguments: JSON.parse(toolCallBuffer.arguments),
            });
          } catch (e) {
            console.error('Error parsing tool arguments:', e);
          }
        }
        callbacks.onComplete();
        return;
      }

      const data = line.replace(/^data: /, '');
      if (!data) continue;

      try {
        const parsed = JSON.parse(data);
        const delta = parsed.choices?.[0]?.delta;

        if (delta?.content) {
          callbacks.onToken(delta.content);
        }

        if (delta?.tool_calls) {
          for (const tc of delta.tool_calls) {
            if (tc.function?.name) {
              toolCallBuffer = { name: tc.function.name, arguments: '' };
            }
            if (tc.function?.arguments && toolCallBuffer) {
              toolCallBuffer.arguments += tc.function.arguments;
            }
          }
        }

        if (parsed.choices?.[0]?.finish_reason === 'stop') {
          callbacks.onComplete();
        }
      } catch {
        // Skip invalid JSON chunks
      }
    }
  }
}

// Cloud provider streaming (OpenRouter, OpenAI, Anthropic, xAI, Google)
async function streamCloudChat(
  provider: 'openrouter' | 'openai' | 'anthropic' | 'google' | 'xai',
  model: string,
  messages: LLMMessage[],
  callbacks: StreamCallbacks,
  useTools: boolean,
  options: LLMOptions = {},
  apiKey: string
): Promise<void> {
  // Anthropic has a different API format
  if (provider === 'anthropic') {
    await streamAnthropicChat(model, messages, callbacks, options, apiKey);
    return;
  }
  
  // Google Gemini has its own format
  if (provider === 'google') {
    await streamGoogleChat(model, messages, callbacks, options, apiKey);
    return;
  }
  
  // OpenRouter, OpenAI, and xAI all use OpenAI-compatible format
  const baseUrl = CLOUD_URLS[provider];
  
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`,
  };
  
  // OpenRouter specific headers
  if (provider === 'openrouter') {
    headers['HTTP-Referer'] = 'https://sanctumwriter.app';
    headers['X-Title'] = 'SanctumWriter Pro';
  }
  
  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    temperature: options.temperature ?? 0.7,
    top_p: options.topP ?? 0.9,
    max_tokens: options.maxTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
  };

  if (useTools) {
    body.tools = formatToolsForOpenAI(documentTools);
    body.tool_choice = 'auto';
  }
  
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`${provider} error: ${response.status} - ${errorText}`);
  }
  
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');
  
  const decoder = new TextDecoder();
  let buffer = '';
  let toolCallBuffer: { name: string; arguments: string } | null = null;
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (!line.trim() || line.startsWith(':')) continue;
      if (line === 'data: [DONE]') {
        if (toolCallBuffer && toolCallBuffer.arguments) {
          try {
            callbacks.onToolCall({
              name: toolCallBuffer.name,
              arguments: JSON.parse(toolCallBuffer.arguments),
            });
          } catch (e) {
            console.error('Error parsing tool arguments:', e);
          }
        }
        callbacks.onComplete();
        return;
      }
      
      const data = line.replace(/^data: /, '');
      if (!data) continue;
      
      try {
        const parsed = JSON.parse(data);
        const delta = parsed.choices?.[0]?.delta;
        
        if (delta?.content) {
          callbacks.onToken(delta.content);
        }
        
        if (delta?.tool_calls) {
          for (const tc of delta.tool_calls) {
            if (tc.function?.name) {
              toolCallBuffer = { name: tc.function.name, arguments: '' };
            }
            if (tc.function?.arguments && toolCallBuffer) {
              toolCallBuffer.arguments += tc.function.arguments;
            }
          }
        }
        
        if (parsed.choices?.[0]?.finish_reason === 'stop' || parsed.choices?.[0]?.finish_reason === 'tool_calls') {
          callbacks.onComplete();
        }
      } catch {
        // Skip invalid JSON chunks
      }
    }
  }
}

// Anthropic Claude streaming
async function streamAnthropicChat(
  model: string,
  messages: LLMMessage[],
  callbacks: StreamCallbacks,
  options: LLMOptions = {},
  apiKey: string
): Promise<void> {
  // Convert messages to Anthropic format (system message is separate)
  const systemMessage = messages.find(m => m.role === 'system');
  const chatMessages = messages
    .filter(m => m.role !== 'system')
    .map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content,
    }));
  
  const body: Record<string, unknown> = {
    model,
    messages: chatMessages,
    max_tokens: options.maxTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
    stream: true,
  };
  
  if (systemMessage) {
    body.system = systemMessage.content;
  }
  
  const response = await fetch(`${CLOUD_URLS.anthropic}/messages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify(body),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Anthropic error: ${response.status} - ${errorText}`);
  }
  
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');
  
  const decoder = new TextDecoder();
  let buffer = '';
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (!line.trim() || !line.startsWith('data: ')) continue;
      
      const data = line.replace(/^data: /, '');
      if (!data) continue;
      
      try {
        const parsed = JSON.parse(data);
        
        if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
          callbacks.onToken(parsed.delta.text);
        }
        
        if (parsed.type === 'message_stop') {
          callbacks.onComplete();
          return;
        }
      } catch {
        // Skip invalid JSON
      }
    }
  }
}

// Google Gemini streaming
async function streamGoogleChat(
  model: string,
  messages: LLMMessage[],
  callbacks: StreamCallbacks,
  options: LLMOptions = {},
  apiKey: string
): Promise<void> {
  // Convert messages to Gemini format
  const contents = messages
    .filter(m => m.role !== 'system')
    .map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }));
  
  const systemInstruction = messages.find(m => m.role === 'system');
  
  const body: Record<string, unknown> = {
    contents,
    generationConfig: {
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      maxOutputTokens: options.maxTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
    },
  };
  
  if (systemInstruction) {
    body.systemInstruction = { parts: [{ text: systemInstruction.content }] };
  }
  
  const response = await fetch(
    `${CLOUD_URLS.google}/models/${model}:streamGenerateContent?alt=sse&key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }
  );
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Google error: ${response.status} - ${errorText}`);
  }
  
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');
  
  const decoder = new TextDecoder();
  let buffer = '';
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (!line.trim() || !line.startsWith('data: ')) continue;
      
      const data = line.replace(/^data: /, '');
      if (!data) continue;
      
      try {
        const parsed = JSON.parse(data);
        const text = parsed.candidates?.[0]?.content?.parts?.[0]?.text;
        
        if (text) {
          callbacks.onToken(text);
        }
        
        if (parsed.candidates?.[0]?.finishReason === 'STOP') {
          callbacks.onComplete();
          return;
        }
      } catch {
        // Skip invalid JSON
      }
    }
  }
  
  callbacks.onComplete();
}

export async function nonStreamingChat(
  provider: LLMProviderType,
  model: string,
  messages: LLMMessage[],
  useTools: boolean = false
): Promise<{ content: string; toolCalls?: Array<{ name: string; arguments: Record<string, unknown> }> }> {
  if (provider === 'ollama') {
    const body: Record<string, unknown> = {
      model,
      messages,
      stream: false,
    };
    if (useTools) {
      body.tools = formatToolsForOllama(documentTools);
    }

    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) throw new Error(`Ollama error: ${response.statusText}`);
    const data = await response.json();
    
    return {
      content: data.message?.content || '',
      toolCalls: data.message?.tool_calls?.map((tc: { function: { name: string; arguments: string | Record<string, unknown> } }) => ({
        name: tc.function.name,
        arguments: typeof tc.function.arguments === 'string'
          ? JSON.parse(tc.function.arguments)
          : tc.function.arguments,
      })),
    };
  } else {
    const body: Record<string, unknown> = {
      model,
      messages,
      stream: false,
    };
    if (useTools) {
      body.tools = formatToolsForLMStudio(documentTools);
      body.tool_choice = 'auto';
    }

    const response = await fetch(`${LMSTUDIO_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) throw new Error(`LM Studio error: ${response.statusText}`);
    const data = await response.json();
    const choice = data.choices?.[0];

    return {
      content: choice?.message?.content || '',
      toolCalls: choice?.message?.tool_calls?.map((tc: { function: { name: string; arguments: string | Record<string, unknown> } }) => ({
        name: tc.function.name,
        arguments: typeof tc.function.arguments === 'string'
          ? JSON.parse(tc.function.arguments)
          : tc.function.arguments,
      })),
    };
  }
}
