const { useState, useEffect, useRef } = React;

// Send icon component
const SendIcon = () => (
  <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

// Knowledge base icon component
const KnowledgeIcon = () => (
  <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
  </svg>
);

// Close icon component
const CloseIcon = () => (
  <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

// User avatar component
const UserAvatar = () => (
  <div className="avatar user-avatar">
    You
  </div>
);

// AI avatar component
const AIAvatar = () => (
  <div className="avatar ai-avatar">
    AI
  </div>
);

// New chat icon component
const NewChatIcon = () => (
  <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
    <line x1="12" y1="11" x2="12" y2="15"></line>
    <line x1="10" y1="13" x2="14" y2="13"></line>
  </svg>
);

// Knowledge modal component
const KnowledgeModal = ({ isOpen, onClose, knowledgeFiles, setKnowledgeFiles, setKnowledgeOpen }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const fileInputRef = useRef(null);
  
  // Fetch file content
  const fetchFileContent = async (filename) => {
    try {
      setLoading(true);
      const response = await fetch(`/api/knowledge/${filename}`);
      if (!response.ok) {
        throw new Error('Failed to get file content');
      }
      const data = await response.json();
      setFileContent(data.content);
    } catch (error) {
      console.error('Error fetching file content:', error);
      setFileContent('Error loading file content, please try again');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle file click to fetch content
  const handleFileClick = (file) => {
    setSelectedFile(file);
    fetchFileContent(file.name);
  };
  
  // Reset state when closing the dialog
  const handleClose = () => {
    setSelectedFile(null);
    setFileContent('');
    setUploadStatus('');
    onClose();
  };

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Check if file is markdown format
    if (!file.name.endsWith('.md')) {
      setUploadStatus('Error: Only .md files are allowed');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploadStatus('Uploading...');
      const response = await fetch('/api/knowledge/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const data = await response.json();
      setUploadStatus(`File ${data.filename} uploaded successfully`);
      
      // Refresh file list
      const filesResponse = await fetch('/api/knowledge');
      if (filesResponse.ok) {
        const filesData = await filesResponse.json();
        // Update parent component's file list
        onClose();
        setTimeout(() => {
          setKnowledgeFiles(filesData.files);
          setKnowledgeOpen(true);
        }, 100);
      }
    } catch (error) {
      console.error('File upload error:', error);
      setUploadStatus(`Error: ${error.message}`);
    }
  };

  // Trigger file selection dialog
  const handleAddFileClick = () => {
    fileInputRef.current.click();
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="knowledge-modal-overlay">
      <div className="knowledge-modal">
        <div className="knowledge-modal-header">
          <h2>Knowledge Base Files</h2>
          <button className="close-button" onClick={handleClose}>
            <CloseIcon />
          </button>
        </div>
        <div className="knowledge-modal-content">
          <div className="knowledge-file-list">
            <div className="file-list-header">
              <h3>File List</h3>
              <button className="add-file-button" onClick={handleAddFileClick} title="Add File">
                <svg viewBox="0 0 24 24" width="24" height="24">
                  <path fill="currentColor" d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"></path>
                </svg>
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".md"
                style={{ display: 'none' }}
              />
            </div>
            {uploadStatus && (
              <div className={`upload-status ${uploadStatus.includes('Error') ? 'error' : (uploadStatus.includes('success') ? 'success' : '')}`}>
                {uploadStatus}
              </div>
            )}
            <ul>
              {knowledgeFiles.map((file, index) => (
                <li 
                  key={index} 
                  className={`knowledge-file ${selectedFile && selectedFile.name === file.name ? 'selected' : ''}`}
                  onClick={() => handleFileClick(file)}
                >
                  {file.name}
                </li>
              ))}
            </ul>
          </div>
          <div className="knowledge-file-content">
            {selectedFile ? (
              loading ? (
                <div className="loading-message">Loading...</div>
              ) : (
                <div className="file-content-container">
                  <h3>{selectedFile.name}</h3>
                  <div className="file-content">
                    <MarkdownRenderer content={fileContent} />
                  </div>
                </div>
              )
            ) : (
              <div className="no-file-selected">
                Please select a file from the list to view its content
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Markdown renderer component
const MarkdownRenderer = ({ content }) => {
  // Use useEffect to avoid calling window object during server-side rendering
  const [renderedContent, setRenderedContent] = useState("");
  
  useEffect(() => {
    if (content && window.marked && window.DOMPurify) {
      // Configure marked options
      marked.setOptions({
        breaks: true,     // Support line breaks conversion to <br>
        gfm: true,        // Support GitHub-style Markdown
        headerIds: false, // Don't automatically add IDs to header tags
      });
      
      // Parse Markdown and sanitize HTML to prevent XSS attacks
      const rawHtml = marked.parse(content);
      const cleanHtml = DOMPurify.sanitize(rawHtml);
      setRenderedContent(cleanHtml);
    } else {
      setRenderedContent(content);
    }
  }, [content]);
  
  return (
    <div 
      className="markdown-content" 
      dangerouslySetInnerHTML={{ __html: renderedContent }}
    />
  );
};

/**
 * 引用校验结果面板。
 *
 * 这是整个系统唯一一处把"模型的引用到底站不站得住"暴露给用户的地方：
 * 一个编造条号的回答读起来和真回答完全一样，只有这里能区分。
 * 全通过时只给一行，不占视觉重量；有问题时列出每一条和它的失败原因——
 * 「条号不存在」和「条号存在但没检索到」的修法不同，所以不能合并成一句"引用可能不准"。
 */
/**
 * 拒答徽章的文案。
 *
 * 旧文案是「以下内容基于模型的领域常识，非本法条库检索结果」——它与
 * `ABSTENTION_INSTRUCTION` 第 2 条（`src/index.ts`：「**不得**凭记忆或常识编造任何
 * 法条条号、条文内容」）**语义正好相反**：系统明令禁止模型靠常识回答，
 * 徽章却告诉用户这段就是常识。两句话必须有一句是错的，而这句在用户眼前。
 *
 * 加了判官之后有两种拒答，必须分开说——第二种才是这一段真正有意思的地方，
 * 把它说成"没检索到"会让整个判官机制在界面上消失：
 *
 *   - 相似度/无命中：库里没有相关条文
 *   - 判官判定不可答：**检索到了**看似相关的条文，但那些条文回答不了这个问题
 *     （典型例子：问刑事责任，命中的却是分配民事责任的那一条）
 */
function abstainBadgeText(judge) {
  const judgedUnanswerable = judge && judge.outcome === 'judged' && judge.answerable === false;
  if (judgedUnanswerable) {
    return '检索到了看似相关的条文，但判定它回答不了这个问题（例如问题涉及《刑法》，不在本库范围内）。以下为系统说明，不含法条引用。';
  }
  return '本地知识库（民法典 + 公司法）中没有检索到相关条文。以下为系统说明，不含法条引用。';
}

function CitationCheckBlock({ report }) {
  if (!report || !report.total) return null;

  if (!report.hasProblem) {
    return (
      <div className="citation-check ok">
        ✓ {report.total} 处引用条号已逐条核对：均存在于本地法条库，且出现在本次检索结果中
      </div>
    );
  }

  const problems = report.checks.filter(c => c.status !== 'verified');
  return (
    <div className="citation-check warn">
      <div className="citation-check-title">
        ⚠ {report.total} 处引用中有 {problems.length} 处未能核实
        {report.fabricated > 0 && `，其中 ${report.fabricated} 处条号在本地法条库中不存在`}
      </div>
      <ul className="citation-check-list">
        {problems.map((c, i) => (
          <li key={i}>
            <code>{c.label}</code>
            {c.status === 'fabricated'
              ? ' —— 本地法条库中不存在此条号，请勿直接采信'
              : ` —— 条号确实存在（${c.sources.join('、')}），但未出现在本次检索结果中，属于凭记忆引用`}
          </li>
        ))}
      </ul>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [knowledgeFiles, setKnowledgeFiles] = useState([]);
  const [showWelcomeBanner, setShowWelcomeBanner] = useState(true);
  const [statusLine, setStatusLine] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  // 服务端会话 id：多轮历史由服务端按它维护，前端不再自己回传 history
  const sessionIdRef = useRef(null);
  // 流式渲染要按下标就地更新那条 assistant 消息，这里记录当前长度
  const messagesLengthRef = useRef(0);

  // Fetch knowledge base file list
  const fetchKnowledgeFiles = async () => {
    try {
      const response = await fetch('/api/knowledge');
      if (!response.ok) {
        throw new Error('Failed to get knowledge base file list');
      }
      const data = await response.json();
      setKnowledgeFiles(data.files);
    } catch (error) {
      console.error('Error fetching knowledge files:', error);
      // Could add error notification
    }
  };

  // Open knowledge dialog
  const handleOpenKnowledge = () => {
    fetchKnowledgeFiles();
    setKnowledgeOpen(true);
  };

  // Auto-adjust textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '50px';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [input]);

  // Scroll to latest message
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    messagesLengthRef.current = messages.length;
    scrollToBottom();
  }, [messages]);

  // Handle sending message
  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Hide welcome banner
    setShowWelcomeBanner(false);

    // Add user message
    const userMessage = { role: 'user', content: input, time: new Date().toLocaleTimeString() };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // 会话历史由服务端按 sessionId 维护（Redis / 内存兜底），前端只需带上 sessionId
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          sessionId: sessionIdRef.current || undefined
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error('Server error');
      }

      // 先插入一条空的 assistant 消息，随 token 增量更新它
      const assistantIndex = messagesLengthRef.current + 1;
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '',
        time: new Date().toLocaleTimeString(),
        format: 'markdown',
        streaming: true
      }]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let accumulated = '';

      const applyContent = (text) => {
        setMessages(prev => {
          const next = [...prev];
          if (next[assistantIndex]) {
            next[assistantIndex] = { ...next[assistantIndex], content: text };
          }
          return next;
        });
      };

      // SSE 解析：事件之间以空行分隔，每个事件形如 "event: x\ndata: {...}"
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sep;
        while ((sep = buffer.indexOf('\n\n')) !== -1) {
          const rawEvent = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);

          let eventName = 'message';
          let dataLine = '';
          for (const line of rawEvent.split('\n')) {
            if (line.startsWith('event: ')) eventName = line.slice(7).trim();
            else if (line.startsWith('data: ')) dataLine += line.slice(6);
          }
          if (!dataLine) continue;

          let payload;
          try { payload = JSON.parse(dataLine); } catch { continue; }

          if (eventName === 'meta') {
            sessionIdRef.current = payload.sessionId;
          } else if (eventName === 'token') {
            accumulated += payload.token;
            applyContent(accumulated);
          } else if (eventName === 'status') {
            setStatusLine(payload.message || '');
          } else if (eventName === 'done') {
            setStatusLine('');
            setMessages(prev => {
              const next = [...prev];
              if (next[assistantIndex]) {
                next[assistantIndex] = {
                  ...next[assistantIndex],
                  content: accumulated || '（未生成内容）',
                  streaming: false,
                  abstained: payload.abstained,
                  // 判官的判定结果（不含 reason —— 未经审阅的模型理由不进界面）。
                  // 徽章要靠 judge.outcome/answerable 区分"没检索到"和"检索到了但答不了"。
                  judge: payload.judge || null,
                  citationCheck: payload.citationCheck || null,
                  citations: payload.citations || []
                };
              }
              return next;
            });
          } else if (eventName === 'error') {
            throw new Error(payload.error || '服务处理失败');
          }
        }
      }
    } catch (error) {
      setStatusLine('');
      console.error('Error:', error);
      
      // Add error message
      setMessages(prevMessages => [
        ...prevMessages, 
        { 
          role: 'system', 
          content: 'An error occurred. Please try again later. ' + error.message,
          time: new Date().toLocaleTimeString()
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle key events (Enter to send message)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Reset chat
  const handleNewChat = () => {
    setMessages([]);
    setInput('');
    setShowWelcomeBanner(true); // Show welcome banner again
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Legal Assistant</h1>
        <div className="header-buttons">
          <button className="new-chat-button" onClick={handleNewChat} title="New Chat">
            <NewChatIcon /> New Chat
          </button>
          <button className="knowledge-button" onClick={handleOpenKnowledge} title="View Knowledge Base">
            <KnowledgeIcon /> Knowledge Base
          </button>
        </div>
      </div>
      
      {showWelcomeBanner && (
        <div className="welcome-banner">
          Welcome to the Legal Assistant. Please enter your legal question.
        </div>
      )}
      
      <div className="chat-messages">
        {messages.map((message, index) => (
          message.role === 'system' ? (
            <div key={index} className="system-message">
              {message.content}
              {message.time && <div className="message-time">{message.time}</div>}
            </div>
          ) : (
            <div key={index} className={`message-wrapper ${message.role === 'user' ? 'user-wrapper' : 'assistant-wrapper'}`}>
              <div className={`message-bubble ${message.role === 'user' ? 'user-bubble' : 'assistant-bubble'}`}>
                <div className="message">
                  <div className="message-header">
                    {message.role === 'user' ? <UserAvatar /> : <AIAvatar />}
                  </div>
                  <div className="message-body">
                    <div className="message-content">
                      {message.role === 'assistant' ? (
                        <>
                          <MarkdownRenderer content={message.content} />
                          {message.abstained && (
                            <div className="abstained-badge">
                              {abstainBadgeText(message.judge)}
                            </div>
                          )}
                          <CitationCheckBlock report={message.citationCheck} />
                          {message.citations && message.citations.length > 0 && (
                            <div className="citation-list">
                              <span className="citation-list-label">本次检索到的条文：</span>
                              {message.citations.map((c, i) => (
                                <span key={i} className="citation-item" title={c.chapter || ''}>
                                  {c.source.replace(/\.md$/, '')} · {c.articleNo || '章节标题'} · {c.score}
                                </span>
                              ))}
                            </div>
                          )}
                          {message.fromKnowledgeBase !== undefined && (
                            <div className="source-attribution">
                              {message.fromKnowledgeBase ? "This answer is from the local knowledge base" : "This answer is not from the local knowledge base"}
                            </div>
                          )}
                        </>
                      ) : (
                        message.content
                      )}
                    </div>
                    {message.time && <div className="message-time">{message.time}</div>}
                  </div>
                </div>
              </div>
            </div>
          )
        ))}
        
        {/* 流式状态条：检索进度、工具调用等服务端事件实时可见。
            回答本体已经在上面那条 assistant 消息里增量渲染，所以这里不再放整个气泡。 */}
        {loading && (
          <div className="stream-status">
            <span className="loading-dots">{statusLine || 'Thinking'}</span>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input-container">
        <div className="chat-input">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter your question..."
            disabled={loading}
          />
          <button onClick={handleSendMessage} disabled={loading || !input.trim()}>
            <SendIcon />
          </button>
        </div>
      </div>
      
      <KnowledgeModal 
        isOpen={knowledgeOpen} 
        onClose={() => setKnowledgeOpen(false)} 
        knowledgeFiles={knowledgeFiles} 
        setKnowledgeFiles={setKnowledgeFiles}
        setKnowledgeOpen={setKnowledgeOpen}
      />
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root')); 