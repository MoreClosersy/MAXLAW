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

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [knowledgeFiles, setKnowledgeFiles] = useState([]);
  const [showWelcomeBanner, setShowWelcomeBanner] = useState(true);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

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
      // Create chat history, only including role and content
      const chatHistory = messages
        .filter(msg => msg.role !== 'system') // Exclude system messages
        .map(msg => ({ role: msg.role, content: msg.content }));
      
      // Send request to backend API with current message and history
      const response = await fetch('http://localhost:3001/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: input,
          history: chatHistory
        }),
      });

      if (!response.ok) {
        throw new Error('Server error');
      }

      const data = await response.json();
      
      // Add assistant reply
      const assistantMessage = {
        role: 'assistant',
        content: data.response || 'Sorry, I cannot answer this question.',
        time: new Date().toLocaleTimeString(),
        format: data.format || 'text',
        fromKnowledgeBase: data.fromKnowledgeBase
      };
      
      setMessages(prevMessages => [...prevMessages, assistantMessage]);
    } catch (error) {
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
        
        {loading && (
          <div className="message-wrapper assistant-wrapper">
            <div className="message-bubble assistant-bubble">
              <div className="message">
                <div className="message-header">
                  <AIAvatar />
                </div>
                <div className="message-body">
                  <div className="message-content">
                    <span className="loading-dots">Thinking</span>
                  </div>
                </div>
              </div>
            </div>
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