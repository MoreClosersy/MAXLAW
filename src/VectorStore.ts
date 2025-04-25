export interface VectorStoreItem {
    embedding: number[];
    document: string;
    source: string;
}

export default class VectorStore {
    private vectorStore: VectorStoreItem[];

    constructor() {
        this.vectorStore = [];
    }

    async addEmbedding(embedding: number[], document: string, source: string = "unknown") {
        this.vectorStore.push({ embedding, document, source });
    }

    async search(queryEmbedding: number[], topK: number = 3, keywords: string[] = []): Promise<any[]> {
        // Calculate cosine similarity between query and all documents
        const scored = this.vectorStore.map((item) => ({
            document: item.document,
            source: item.source,
            score: this.cosineSimilarity(queryEmbedding, item.embedding),
            // Check if document contains any keywords
            containsKeywords: keywords.length > 0 ? 
                keywords.some(kw => item.document.includes(kw)) : false
        }));
        
        // Sort results:
        // 1. Documents containing keywords first
        // 2. Then by cosine similarity
        const sorted = scored.sort((a, b) => {
            // If one contains keywords and the other doesn't, prioritize the one with keywords
            if (a.containsKeywords && !b.containsKeywords) return -1;
            if (!a.containsKeywords && b.containsKeywords) return 1;
            
            // Otherwise sort by cosine similarity
            return b.score - a.score;
        });
        
        // Get top K results
        const topKDocuments = sorted.slice(0, topK);
        
        return topKDocuments;
    }

    private cosineSimilarity(vecA: number[], vecB: number[]): number {
        const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
        const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (normA * normB);
    }
}