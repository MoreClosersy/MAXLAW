import express, { Request, Response } from 'express';
import cors from 'cors';
import path from 'path';
import { spawn } from 'child_process';
import fs from 'fs';
import { fileURLToPath } from 'url';
import multer from 'multer';

// Get current file's directory path
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Define interfaces
interface Message {
  role: string;
  content: string;
}

interface Conversation {
  [sessionId: string]: Message[];
}

const app = express();
const PORT = 3001;

// Configure multer storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const knowledgeDir = path.join(__dirname, 'knowledge');
    // Ensure knowledge directory exists
    if (!fs.existsSync(knowledgeDir)) {
      fs.mkdirSync(knowledgeDir, { recursive: true });
    }
    cb(null, knowledgeDir);
  },
  filename: function (req, file, cb) {
    // Use original filename
    cb(null, file.originalname);
  }
});

// Configure file filter
const fileFilter = (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  // Only accept markdown files
  if (file.mimetype === 'text/markdown' || file.originalname.endsWith('.md')) {
    cb(null, true);
  } else {
    cb(new Error('Only markdown (.md) files are accepted'));
  }
};

// Configure upload
const upload = multer({ 
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024 // Limit to 5MB
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'frontend')));

// Temporary storage for conversation history
const conversations: Conversation = {};

// Generate simple session ID
function generateSessionId(): string {
  return Math.random().toString(36).substring(2, 15);
}

// Chat API endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message, sessionId = generateSessionId(), history = [] } = req.body;
    
    // Create response file path
    const responseDir = path.join(__dirname, 'output');
    fs.mkdirSync(responseDir, { recursive: true });
    const responseFile = path.join(responseDir, `response_${sessionId}.md`);
    
    // Use simple script to generate response
    const childProcess = spawn('pnpm', ['tsx', 'src/chat-api.ts'], {
      env: {
        ...process.env,
        USER_QUERY: message,
        RESPONSE_FILE: responseFile,
      }
    });
    
    let stdoutData = '';
    let stderrData = '';
    
    childProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
      console.log(`Output: ${data}`);
    });
    
    childProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
      console.error(`Error: ${data}`);
    });
    
    // Wait for process to complete
    await new Promise((resolve, reject) => {
      childProcess.on('close', (code) => {
        console.log(`Process exit code: ${code}`);
        if (code === 0) {
          resolve(null);
        } else {
          reject(new Error(`Process exit code: ${code}`));
        }
      });
      
      // Set timeout
      setTimeout(() => {
        childProcess.kill();
        reject(new Error('Request timeout'));
      }, 180000); // 180 seconds timeout (3 minutes)
    });
    
    // Read generated answer
    let response = '';
    
    if (fs.existsSync(responseFile)) {
      // Read file content in original format, preserving all line breaks and formatting
      response = fs.readFileSync(responseFile, 'utf8');
      
      // Optional: Delete temporary file
      fs.unlinkSync(responseFile);
    } else {
      response = 'Unable to generate an answer, please try again later.';
    }
    
    // Save conversation history
    if (!conversations[sessionId]) {
      conversations[sessionId] = [];
    }
    
    conversations[sessionId].push({ role: 'user', content: message });
    conversations[sessionId].push({ role: 'assistant', content: response });
    
    // Specify this is a Markdown response
    res.json({ 
      response, 
      sessionId,
      format: 'markdown'
    });
  } catch (error) {
    console.error('API error:', error);
    res.status(500).json({ error: 'Server internal error' + ((error as Error).message ? ': ' + (error as Error).message : '') });
  }
});

// Get knowledge base file list
app.get('/api/knowledge', (req: Request, res: Response) => {
  try {
    const knowledgeDir = path.join(__dirname, 'knowledge');
    const files = fs.readdirSync(knowledgeDir).filter(file => file.endsWith('.md'));
    
    const knowledgeFiles = files.map(file => {
      const filePath = path.join(knowledgeDir, file);
      const stats = fs.statSync(filePath);
      return {
        name: file,
        size: stats.size,
        lastModified: stats.mtime
      };
    });
    
    res.json({ files: knowledgeFiles });
  } catch (error) {
    console.error('Error getting knowledge base files:', error);
    res.status(500).json({ error: `Failed to get knowledge base files: ${(error as Error).message}` });
  }
});

// Get knowledge base file content
app.get('/api/knowledge/:filename', (req: Request, res: Response) => {
  try {
    const filename = req.params.filename;
    // Security check: ensure filename doesn't contain path operators
    if (filename.includes('..') || filename.includes('/') || !filename.endsWith('.md')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }
    
    const filePath = path.join(__dirname, 'knowledge', filename);
    
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    const content = fs.readFileSync(filePath, 'utf8');
    res.json({ 
      filename,
      content,
      format: 'markdown'
    });
  } catch (error) {
    console.error('Error getting knowledge base file content:', error);
    res.status(500).json({ error: `Failed to get file content: ${(error as Error).message}` });
  }
});

// Upload knowledge base file
app.post('/api/knowledge/upload', upload.single('file'), (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file received' });
    }

    // Check if file already exists
    const fileName = req.file.originalname;

    // Reindex knowledge base (if you have such functionality)
    // Here you can call script to update vector database or regenerate embeddings
    // For example:
    const childProcess = spawn('pnpm', ['tsx', 'src/index.ts', 'embed'], {
      stdio: 'inherit'
    });

    childProcess.on('close', (code) => {
      console.log(`Knowledge base reindex process exit code: ${code}`);
    });

    res.status(200).json({ 
      message: 'File uploaded successfully',
      filename: fileName
    });
  } catch (error) {
    console.error('File upload error:', error);
    res.status(500).json({ error: `File upload failed: ${(error as Error).message}` });
  }
});

// Serve frontend
app.get('/', (req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
}); 