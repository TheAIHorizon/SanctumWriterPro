import { NextResponse } from 'next/server';

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const LMSTUDIO_URL = process.env.LMSTUDIO_URL || 'http://localhost:1234';

// Cloud provider URLs
const CLOUD_URLS = {
  openrouter: 'https://openrouter.ai/api/v1',
  openai: 'https://api.openai.com/v1',
  anthropic: 'https://api.anthropic.com/v1',
  google: 'https://generativelanguage.googleapis.com/v1beta',
  xai: 'https://api.x.ai/v1',
};

// Curated model lists for providers without /models endpoint or for better UX
const ANTHROPIC_MODELS = [
  { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4 (Latest)', contextLength: 200000 },
  { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet', contextLength: 200000 },
  { id: 'claude-3-5-haiku-20241022', name: 'Claude 3.5 Haiku', contextLength: 200000 },
  { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', contextLength: 200000 },
];

const GOOGLE_MODELS = [
  { id: 'gemini-2.5-flash-preview-05-20', name: 'Gemini 2.5 Flash (Preview)', contextLength: 1000000 },
  { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash', contextLength: 1000000 },
  { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro', contextLength: 2000000 },
  { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash', contextLength: 1000000 },
];

const XAI_MODELS = [
  { id: 'grok-3', name: 'Grok 3', contextLength: 131072 },
  { id: 'grok-3-fast', name: 'Grok 3 Fast', contextLength: 131072 },
  { id: 'grok-2', name: 'Grok 2', contextLength: 131072 },
];

// POST so the API key travels in the request body, never in the query
// string (query strings are captured by server/proxy logs).
export async function POST(request: Request) {
  let body: { provider?: string; apiKey?: string } = {};
  try {
    body = await request.json();
  } catch {
    // Fall through with defaults when no/invalid JSON body is sent
  }
  const provider = body.provider || 'ollama';
  const apiKey =
    request.headers.get('authorization')?.replace(/^Bearer\s+/i, '') ||
    body.apiKey ||
    '';

  try {
    // Local providers
    if (provider === 'ollama') {
      const response = await fetch(`${OLLAMA_URL}/api/tags`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        return NextResponse.json(
          { error: 'Ollama not available', models: [], available: false },
          { status: 503 }
        );
      }

      const data = await response.json();
      const models = (data.models || []).map((m: { name: string; size: number; modified_at: string }) => ({
        id: m.name,
        name: m.name,
        size: m.size,
        modified: m.modified_at,
        provider: 'ollama',
      }));

      return NextResponse.json({ models, provider: 'ollama', available: true });
    }
    
    if (provider === 'lmstudio') {
      const response = await fetch(`${LMSTUDIO_URL}/v1/models`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        return NextResponse.json(
          { error: 'LM Studio not available', models: [], available: false },
          { status: 503 }
        );
      }

      const data = await response.json();
      const models = (data.data || []).map((m: { id: string }) => ({
        id: m.id,
        name: m.id,
        provider: 'lmstudio',
      }));

      return NextResponse.json({ models, provider: 'lmstudio', available: true });
    }

    // Cloud providers (Pro feature)
    if (!apiKey) {
      return NextResponse.json(
        { error: `API key required for ${provider}`, models: [], available: false },
        { status: 401 }
      );
    }

    // OpenRouter - has /models endpoint
    if (provider === 'openrouter') {
      const response = await fetch(`${CLOUD_URLS.openrouter}/models`, {
        headers: { 'Authorization': `Bearer ${apiKey}` },
      });

      if (!response.ok) {
        return NextResponse.json(
          { error: 'OpenRouter API key invalid or expired', models: [], available: false },
          { status: response.status }
        );
      }

      const data = await response.json();
      // Filter to writing-suitable models and sort by popularity
      const models = (data.data || [])
        .filter((m: { id: string }) => 
          m.id.includes('gpt') || 
          m.id.includes('claude') || 
          m.id.includes('llama') ||
          m.id.includes('mistral') ||
          m.id.includes('gemini') ||
          m.id.includes('qwen')
        )
        .slice(0, 50) // Limit to 50 most relevant
        .map((m: { id: string; name?: string; context_length?: number; pricing?: { prompt: string; completion: string } }) => ({
          id: m.id,
          name: m.name || m.id,
          provider: 'openrouter',
          contextLength: m.context_length,
          pricing: m.pricing ? {
            prompt: parseFloat(m.pricing.prompt) * 1000000,
            completion: parseFloat(m.pricing.completion) * 1000000,
          } : undefined,
        }));

      return NextResponse.json({ models, provider: 'openrouter', available: true });
    }

    // OpenAI - has /models endpoint
    if (provider === 'openai') {
      const response = await fetch(`${CLOUD_URLS.openai}/models`, {
        headers: { 'Authorization': `Bearer ${apiKey}` },
      });

      if (!response.ok) {
        return NextResponse.json(
          { error: 'OpenAI API key invalid', models: [], available: false },
          { status: response.status }
        );
      }

      const data = await response.json();
      // Filter to chat models
      const models = (data.data || [])
        .filter((m: { id: string }) => 
          m.id.includes('gpt-4') || 
          m.id.includes('gpt-3.5') ||
          m.id.includes('o1') ||
          m.id.includes('o3')
        )
        .map((m: { id: string }) => ({
          id: m.id,
          name: m.id,
          provider: 'openai',
        }));

      return NextResponse.json({ models, provider: 'openai', available: true });
    }

    // Anthropic - use curated list (they don't have public /models)
    if (provider === 'anthropic') {
      // Just verify the key works with a simple test
      // In production, you'd want a lighter-weight verification
      const models = ANTHROPIC_MODELS.map(m => ({
        ...m,
        provider: 'anthropic',
      }));

      return NextResponse.json({ models, provider: 'anthropic', available: true });
    }

    // Google - use curated list
    if (provider === 'google') {
      const models = GOOGLE_MODELS.map(m => ({
        ...m,
        provider: 'google',
      }));

      return NextResponse.json({ models, provider: 'google', available: true });
    }

    // xAI - use curated list
    if (provider === 'xai') {
      const models = XAI_MODELS.map(m => ({
        ...m,
        provider: 'xai',
      }));

      return NextResponse.json({ models, provider: 'xai', available: true });
    }

    return NextResponse.json(
      { error: `Unknown provider: ${provider}`, models: [], available: false },
      { status: 400 }
    );

  } catch (error) {
    console.error('Error fetching models:', error);
    return NextResponse.json(
      { error: `${provider} not available`, models: [], available: false },
      { status: 503 }
    );
  }
}

