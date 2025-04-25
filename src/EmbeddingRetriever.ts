import { logTitle } from "./utils.js";
import VectorStore from "./VectorStore.js";
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

export default class EmbeddingRetriever {
    private embeddingModel: string;
    private vectorStore: VectorStore;
    private documentMap: Map<string, {id: number, source: string}> = new Map();
    private documentCounter: number = 0;
    private conversationHistory: string[] = [];
    private embeddingCache: Map<string, number[]> = new Map();
    private cacheDir: string;
    private cacheFile: string;

    constructor(embeddingModel: string) {
        this.embeddingModel = embeddingModel;
        this.vectorStore = new VectorStore();
        this.cacheDir = path.join(process.cwd(), 'cache');
        this.cacheFile = path.join(this.cacheDir, `${embeddingModel.replace(/\//g, '_')}_cache.json`);
        this.initializeCache();
    }

    private initializeCache() {
        // Ensure cache directory exists
        if (!fs.existsSync(this.cacheDir)) {
            fs.mkdirSync(this.cacheDir, { recursive: true });
            console.log(`Created cache directory: ${this.cacheDir}`);
        }

        // Load existing cache
        if (fs.existsSync(this.cacheFile)) {
            try {
                const cacheData = JSON.parse(fs.readFileSync(this.cacheFile, 'utf-8'));
                this.embeddingCache = new Map(Object.entries(cacheData));
                console.log(`Loaded ${this.embeddingCache.size} embedding cache items`);
            } catch (error) {
                console.error('Failed to load embedding cache file:', error);
                this.embeddingCache = new Map();
            }
        }
    }

    private saveCache() {
        try {
            const cacheObj = Object.fromEntries(this.embeddingCache);
            fs.writeFileSync(this.cacheFile, JSON.stringify(cacheObj), 'utf-8');
            console.log(`Saved ${this.embeddingCache.size} embedding cache items to ${this.cacheFile}`);
        } catch (error) {
            console.error('Failed to save embedding cache:', error);
        }
    }

    private getCacheKey(text: string): string {
        // Use hash of text content as cache key to ensure same content has same key
        return crypto.createHash('md5').update(text).digest('hex');
    }

    public saveToHistory(query: string, response: string) {
        this.conversationHistory.push(query, response);
        if (this.conversationHistory.length > 10) {
            this.conversationHistory.shift();
        }
    }

    private generateDeterministicEmbedding(text: string, dimension: number = 1536): number[] {
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        
        const rng = (n: number) => {
            const x = Math.sin(n + hash) * 10000;
            return x - Math.floor(x);
        };
        
        const embedding = Array(dimension).fill(0)
            .map((_, i) => rng(i));
        
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        return embedding.map(val => val / magnitude);
    }

    async embedDocument(document: string, source: string = "unknown") {
        logTitle('EMBEDDING DOCUMENT');
        const docId = this.documentCounter++;
        this.documentMap.set(document, {id: docId, source: source});
        
        const paragraphs = this.splitIntoParagraphs(document);
        
        for (const paragraph of paragraphs) {
            if (paragraph.trim().length === 0) continue;
        
            let embedding;
            try {
                embedding = await this.embed(paragraph);
            } catch (error) {
                console.error('Failed to get embedding from API, using fallback method');
                embedding = this.generateDeterministicEmbedding(paragraph);
            }
        
            await this.vectorStore.addEmbedding(embedding, paragraph, source);
        }
        
        // Save cache after processing all paragraphs
        this.saveCache();
        
        return true;
    }
    
    private splitIntoParagraphs(document: string): string[] {
        const sections = document.split(/\n\s*\n|\n#{1,6}\s+/);
        
        const headings = document.match(/\n#{1,6}\s+[^\n]+/g) || [];
        
        const paragraphs = [...sections, ...headings]
            .filter(p => p && p.trim().length > 0);
            
        return paragraphs;
    }

    async embedQuery(query: string) {
        logTitle('EMBEDDING QUERY');
        const embedding = await this.embed(query);
        return embedding;
    }

    private async embed(document: string): Promise<number[]> {
        try {
            // Check cache first
            const cacheKey = this.getCacheKey(document);
            if (this.embeddingCache.has(cacheKey)) {
                console.log('Using cached embedding vector');
                return this.embeddingCache.get(cacheKey)!;
            }

            // Check if API key is set to "none", if so use local embedding directly
            if (!process.env.EMBEDDING_API_KEY || process.env.EMBEDDING_API_KEY === 'none') {
                console.log('Using local embedding method (API disabled)');
                const embedding = this.generateDeterministicEmbedding(document);
                // Cache result
                this.embeddingCache.set(cacheKey, embedding);
                return embedding;
            }
            
            const isHuggingFace = process.env.EMBEDDING_API_KEY?.startsWith('hf_');
            let data;
            let embedding: number[];
            
            if (isHuggingFace) {
                const response = await fetch(`${process.env.EMBEDDING_BASE_URL}/${this.embeddingModel}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${process.env.EMBEDDING_API_KEY}`,
                    },
                    body: JSON.stringify({
                        inputs: {
                            source_sentence: document,
                            sentences: [document]
                        }
                    }),
                });
                
                // Check response status code
                if (!response.ok) {
                    console.error(`API response error: ${response.status} ${response.statusText}`);
                    embedding = this.generateDeterministicEmbedding(document);
                }
                
                // Try to get response text for checking
                const responseText = await response.text();
                try {
                    data = JSON.parse(responseText);
                } catch (parseError) {
                    console.error('JSON parsing error, response content:', responseText.substring(0, 100) + '...');
                    embedding = this.generateDeterministicEmbedding(document);
                }
                
                console.log('API Response (HF):', JSON.stringify(data, null, 2));
                
                if (Array.isArray(data) && typeof data[0] === 'number') {
                    embedding = this.generateDeterministicEmbedding(document);
                }
                else if (data && data.embeddings) {
                    embedding = data.embeddings[0];
                } else {
                    console.error('Invalid Hugging Face API response format');
                    embedding = this.generateDeterministicEmbedding(document);
                }
            } else {
                const response = await fetch(`${process.env.EMBEDDING_BASE_URL}/embeddings`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${process.env.EMBEDDING_API_KEY}`,
                    },
                    body: JSON.stringify({
                        model: this.embeddingModel,
                        input: document,
                        encoding_format: 'float',
                    }),
                });
                
                // Check response status code
                if (!response.ok) {
                    console.error(`API response error: ${response.status} ${response.statusText}`);
                    embedding = this.generateDeterministicEmbedding(document);
                }
                
                // Try to get response text for checking
                const responseText = await response.text();
                try {
                    data = JSON.parse(responseText);
                } catch (parseError) {
                    console.error('JSON parsing error, response content:', responseText.substring(0, 100) + '...');
                    embedding = this.generateDeterministicEmbedding(document);
                }
                
                console.log('API Response (OpenAI):', JSON.stringify(data, null, 2));
                
                if (data.data && data.data[0] && data.data[0].embedding) {
                    embedding = data.data[0].embedding;
                } else {
                    console.error('Invalid API response format');
                    embedding = this.generateDeterministicEmbedding(document);
                }
            }
            
            // Cache result
            this.embeddingCache.set(cacheKey, embedding);
            
            // Periodically save cache (can set threshold or timer)
            if (this.embeddingCache.size % 10 === 0) {
                this.saveCache();
            }
            
            return embedding;
        } catch (error) {
            console.error('Error in embed method:', error);
            const embedding = this.generateDeterministicEmbedding(document);
            
            // Cache result even in error cases to avoid repeat errors
            const cacheKey = this.getCacheKey(document);
            this.embeddingCache.set(cacheKey, embedding);
            
            return embedding;
        }
    }

    async retrieve(query: string, topK: number = 3): Promise<string[]> {
        const keywords = this.extractKeywords(query);
        console.log("Extracted keywords:", keywords);
        
        this.saveToHistory(query, "");
        
        const queryEmbedding = await this.embedQuery(query);
        const results = await this.vectorStore.search(queryEmbedding, topK, keywords);
        
        const validResults = results.filter((item: any) => 
            item && item.document && item.document.trim().length > 0);
            
        if (validResults.length === 0) {
            console.log('No valid results found in local knowledge base');
            return [];
        }
        
        const resultsWithSource = validResults.map((item: any) => {
            return `【Source: ${item.source}】\n${item.document}`;
        });
        
        this.saveToHistory(query, resultsWithSource.join("\n\n"));
        
        return resultsWithSource;
    }
    
    private extractKeywords(query: string): string[] {
        const keywords: string[] = [];
        
        const hasReference = query.includes("this") || 
                            query.includes("rephrasing") || 
                            query.includes("first point") ||
                            query.includes("what does it mean") ||
                            query.includes("explain") ||
                            query.includes("clarify");
        
        if (hasReference && this.conversationHistory.length > 0) {
            console.log("Detected referential query, trying to find keywords from history");
            
            for (let i = this.conversationHistory.length - 1; i >= 0; i--) {
                const historyItem = this.conversationHistory[i];
                
                const lawMatches = historyItem.match(/第[一二三四五六七八九十百千万零\d]+[条章节款项]/g) || [];
                if (lawMatches.length > 0) {
                    console.log("Found law articles from history:", lawMatches);
                    keywords.push(...lawMatches);
                }
                
                const lawNames = ['Civil Code', 'Contract Law', 'Company Law', 'Labor Law', 'Intellectual Property'];
                lawNames.forEach(name => {
                    if (historyItem.includes(name) && !keywords.includes(name)) {
                        console.log("Found law name from history:", name);
                        keywords.push(name);
                    }
                });
                
                if (keywords.length > 0) {
                    break;
                }
            }
        }
        
        const lawMatches = query.match(/第[一二三四五六七八九十百千万零\d]+[条章节款项]/g) || [];
        if (lawMatches.length > 0) {
            keywords.push(...lawMatches);
        }
        
        const numberMatches = query.match(/\d+/g) || [];
        if (numberMatches.length > 0) {
            keywords.push(...numberMatches);
        }
        
        const lawNames = ['Civil Code', 'Contract Law', 'Company Law', 'Labor Law', 'Intellectual Property'];
        lawNames.forEach(name => {
            if (query.includes(name) && !keywords.includes(name)) {
                keywords.push(name);
            }
        });
        
        return Array.from(new Set(keywords));
    }
}