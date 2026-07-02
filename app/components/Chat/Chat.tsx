'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Bot, User, Loader2, CheckCircle, XCircle, Wand2, Database, Brain } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/lib/store/useAppStore';
import { useChatStore } from '@/lib/store/useChatStore';
import { useSettingsStore } from '@/lib/store/useSettingsStore';
import { useWorkflowStore, STAGE_LABELS } from '@/lib/store/useWorkflowStore';
import { useRAGStore } from '@/lib/store/useRAGStore';
import { streamChat, estimateMessagesTokens, LLMOptions } from '@/lib/llm/client';
import { getSystemPrompt, WorkflowContext } from '@/lib/llm/tools';
import { parseToolCallToOperation, executeOperation } from '@/lib/editor/operations';
import { retrieveContext, buildMemoryContext, autoSaveSession } from '@/lib/rag';
import { ChatMessage, LLMMessage, isCloudProvider } from '@/types';
import { EditorView } from '@codemirror/view';
import { v4 as uuidv4 } from 'uuid';

interface ChatProps {
  editorView: EditorView | null;
}

function MessageContent({ message }: { message: ChatMessage }) {
  return (
    <div
      className={cn(
        'chat-message rounded-lg px-3 py-2 max-w-full',
        message.role === 'user' ? 'bg-user-message' : 'bg-ai-message'
      )}
    >
      <div className="flex items-start gap-2">
        <div className={cn(
          'w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5',
          message.role === 'user' ? 'bg-border' : 'bg-accent/20'
        )}>
          {message.role === 'user' ? (
            <User className="w-3.5 h-3.5 text-text-secondary" />
          ) : (
            <Bot className="w-3.5 h-3.5 text-accent" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm text-text-primary whitespace-pre-wrap break-words">
            {message.content}
            {message.status === 'streaming' && (
              <span className="inline-block w-2 h-4 bg-accent ml-0.5 animate-pulse" />
            )}
          </div>
          
          {/* Tool calls */}
          {message.toolCalls && message.toolCalls.length > 0 && (
            <div className="mt-2 space-y-1">
              {message.toolCalls.map((tc) => (
                <div
                  key={tc.id}
                  className="flex items-center gap-2 text-xs bg-sidebar-bg rounded px-2 py-1"
                >
                  <Wand2 className="w-3 h-3 text-accent" />
                  <span className="text-text-secondary">{tc.name}</span>
                  {tc.result && (
                    tc.result.success ? (
                      <CheckCircle className="w-3 h-3 text-green-500" />
                    ) : (
                      <XCircle className="w-3 h-3 text-red-500" />
                    )
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function Chat({ editorView }: ChatProps) {
  const [input, setInput] = useState('');
  const [ragStatus, setRagStatus] = useState<'idle' | 'retrieving' | 'ready'>('idle');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  const { currentDocument, selection, provider, model, showToast } = useAppStore();
  const {
    messages,
    isStreaming,
    addMessage,
    updateMessage,
    appendToMessage,
    addToolCallToMessage,
    updateToolCallResult,
    setIsStreaming,
  } = useChatStore();
  const {
    temperature,
    topP,
    topK,
    repeatPenalty,
    contextLength,
    setContextUsed,
    serviceURLs,
    getAPIKey,
  } = useSettingsStore();
  
  const { getWorkflow, getProgress } = useWorkflowStore();
  const { ragSettings, sessionMemorySettings } = useRAGStore();
  
  // Listen for prompt library insertions
  useEffect(() => {
    const handlePromptInsert = (e: CustomEvent<{ content: string }>) => {
      setInput(e.detail.content);
      inputRef.current?.focus();
    };
    
    window.addEventListener('sanctum-insert-prompt', handlePromptInsert as EventListener);
    return () => {
      window.removeEventListener('sanctum-insert-prompt', handlePromptInsert as EventListener);
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!input.trim() || isStreaming) return;
    if (!currentDocument) {
      showToast('Open a document first', 'info');
      return;
    }

    const userMessage = input.trim();
    setInput('');

    // Add user message
    addMessage({ role: 'user', content: userMessage, status: 'complete' });

    // Create assistant message placeholder
    const assistantId = addMessage({
      role: 'assistant',
      content: '',
      status: 'streaming',
    });

    setIsStreaming(true);

    // Get workflow context if available
    let workflowContext: WorkflowContext | undefined;
    const workflow = getWorkflow(currentDocument.path);
    if (workflow) {
      const progress = getProgress(currentDocument.path);
      const pendingItems = workflow.checklist
        .filter(item => item.stage === workflow.currentStage && !item.completed)
        .map(item => item.label);
      
      workflowContext = {
        currentStage: workflow.currentStage,
        stageLabel: STAGE_LABELS[workflow.currentStage],
        progress,
        pendingItems,
        notes: workflow.notes,
      };
    }

    // Build base system prompt
    let systemPrompt = getSystemPrompt(
      currentDocument.content,
      currentDocument.path,
      selection ? {
        text: selection.text,
        fromLine: selection.fromLine,
        toLine: selection.toLine,
      } : undefined,
      workflowContext
    );

    // RAG: Retrieve relevant context from knowledge base
    let ragContext = '';
    if (ragSettings.enabled) {
      try {
        setRagStatus('retrieving');
        
        // Retrieve from knowledge base
        const retrieval = await retrieveContext(userMessage, {
          collections: ragSettings.collections,
          maxResults: ragSettings.maxRetrievedChunks,
          minScore: ragSettings.minSimilarityScore,
          maxTokens: ragSettings.maxTokensForContext,
        }, ragSettings.embeddingModel);

        if (retrieval.results.length > 0) {
          ragContext = `

## Retrieved Context from Knowledge Base

The following information was retrieved from the user's knowledge base and may be relevant:

${retrieval.context}

Use this context to inform your response when relevant. Cite sources when referencing specific information.
`;
        }

        // Retrieve session memory if enabled
        if (sessionMemorySettings.enabled) {
          const memoryContext = await buildMemoryContext(
            currentDocument.path,
            userMessage,
            ragSettings.embeddingModel
          );
          if (memoryContext) {
            ragContext += '\n' + memoryContext;
          }
        }

        setRagStatus('ready');
      } catch (error) {
        console.warn('RAG retrieval failed, continuing without context:', error);
        setRagStatus('idle');
      }
    }

    // Append RAG context to system prompt
    systemPrompt += ragContext;

    const llmMessages: LLMMessage[] = [
      { role: 'system', content: systemPrompt },
      // Include recent conversation history
      ...messages.slice(-6).map((m) => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })),
      { role: 'user', content: userMessage },
    ];

    // Track context usage
    const contextUsed = estimateMessagesTokens(llmMessages);
    setContextUsed(contextUsed);

    // Build options from settings
    const llmOptions: LLMOptions = {
      temperature,
      topP,
      topK,
      repeatPenalty,
      contextLength,
    };

    // Cloud providers require an API key entered in Settings; local
    // providers (Ollama/LM Studio) don't use one.
    const apiKey = isCloudProvider(provider) ? getAPIKey(provider) : undefined;

    if (isCloudProvider(provider) && !apiKey) {
      updateMessage(assistantId, {
        status: 'error',
        content: `Error: No API key configured for ${provider}. Add one in Settings.`,
      });
      setIsStreaming(false);
      showToast(`No API key configured for ${provider}`, 'error');
      return;
    }

    try {
      await streamChat(
        provider,
        model,
        llmMessages,
        {
          onToken: (token) => {
            appendToMessage(assistantId, token);
          },
          onToolCall: (toolCall) => {
            const toolCallId = uuidv4();
            addToolCallToMessage(assistantId, {
              id: toolCallId,
              name: toolCall.name,
              arguments: toolCall.arguments,
            });

            // Execute the tool call
            if (editorView) {
              const operation = parseToolCallToOperation(toolCall.name, toolCall.arguments);
              const result = executeOperation(editorView, operation);
              
              updateToolCallResult(assistantId, toolCallId, {
                success: result.success,
                message: result.message,
              });

              if (result.success) {
                showToast(result.message, 'success');
              } else {
                showToast(result.message, 'error');
              }
            }
          },
          onComplete: async () => {
            updateMessage(assistantId, { status: 'complete' });
            setIsStreaming(false);
            setRagStatus('idle');

            // Auto-save session memory if enabled
            if (sessionMemorySettings.enabled && sessionMemorySettings.autoSave && currentDocument) {
              const allMessages = [...messages, { role: 'user', content: userMessage }];
              if (allMessages.length >= sessionMemorySettings.autoSaveThreshold) {
                try {
                  await autoSaveSession(
                    currentDocument.path,
                    allMessages.map(m => ({ role: m.role, content: m.content })),
                    serviceURLs?.ollama || 'http://localhost:11434',
                    model,
                    ragSettings.embeddingModel,
                    sessionMemorySettings.autoSaveThreshold
                  );
                } catch (error) {
                  console.warn('Auto-save session failed:', error);
                }
              }
            }
          },
          onError: (error) => {
            updateMessage(assistantId, {
              status: 'error',
              content: `Error: ${error.message}`,
            });
            setIsStreaming(false);
            setRagStatus('idle');
            showToast(`Chat error: ${error.message}`, 'error');
          },
        },
        true, // Use tools
        llmOptions,
        apiKey
      );
    } catch (error) {
      updateMessage(assistantId, {
        status: 'error',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
      setIsStreaming(false);
    }
  }, [
    input,
    isStreaming,
    currentDocument,
    selection,
    provider,
    model,
    messages,
    editorView,
    addMessage,
    appendToMessage,
    addToolCallToMessage,
    updateToolCallResult,
    updateMessage,
    setIsStreaming,
    showToast,
    temperature,
    topP,
    topK,
    repeatPenalty,
    contextLength,
    setContextUsed,
    getAPIKey,
    ragSettings,
    sessionMemorySettings,
    serviceURLs,
    getWorkflow,
    getProgress,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="h-full flex flex-col bg-chat-bg">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
        <Bot className="w-4 h-4 text-accent" />
        <span className="text-sm font-medium text-text-primary">Assistant</span>
        {selection && (
          <span className="text-xs text-accent bg-accent/10 px-2 py-0.5 rounded">
            Selection active
          </span>
        )}
        {ragSettings.enabled && (
          <span className={cn(
            "text-xs px-2 py-0.5 rounded flex items-center gap-1",
            ragStatus === 'retrieving' 
              ? "bg-yellow-500/10 text-yellow-500" 
              : "bg-green-500/10 text-green-500"
          )}>
            <Database className="w-3 h-3" />
            {ragStatus === 'retrieving' ? 'Retrieving...' : 'RAG'}
          </span>
        )}
        {sessionMemorySettings.enabled && (
          <span className="text-xs bg-purple-500/10 text-purple-500 px-2 py-0.5 rounded flex items-center gap-1">
            <Brain className="w-3 h-3" />
            Memory
          </span>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto p-3 space-y-3">
        {messages.length === 0 ? (
          <div className="text-center text-text-secondary text-sm py-8">
            <Bot className="w-10 h-10 mx-auto mb-3 text-accent opacity-50" />
            <p>Ask me to help with your document</p>
            <p className="text-xs mt-2">
              Select text and ask me to rewrite it,<br />
              or ask for any writing assistance
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <MessageContent key={message.id} message={message} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-border">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            data-chat-input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selection
                ? 'Ask about the selected text...'
                : 'Ask for writing help...'
            }
            className="flex-1 px-3 py-2 bg-sidebar-bg border border-border rounded-lg text-sm text-text-primary placeholder-text-secondary resize-none focus:outline-none focus:border-accent min-h-[40px] max-h-[120px]"
            rows={1}
            disabled={isStreaming}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isStreaming}
            className={cn(
              'p-2 rounded-lg transition-colors',
              input.trim() && !isStreaming
                ? 'bg-accent text-white hover:bg-accent-hover'
                : 'bg-border text-text-secondary cursor-not-allowed'
            )}
          >
            {isStreaming ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

