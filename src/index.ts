import MCPClient from "./MCPClient.js";
import Agent from "./Agent.js";
import path from "path";
import EmbeddingRetriever from "./EmbeddingRetriever.js";
import fs from "fs";
import { logTitle } from "./utils.js";

// Legal Assistant System Prompt
const LEGAL_SYSTEM_PROMPT = `You are a professional legal advisor, proficient in international and domestic legal regulations, and skilled at answering users' legal questions based on the local legal knowledge base.

## Response Requirements:
1. Legal Text Citations:
   - Accurately cite relevant legal provisions from the retrieved local legal knowledge base
   - Always mark the source for each citation, format as: "Document Name" Article X/Section/Clause
   - Directly quote legal text using quotation marks, maintaining the original wording
   - Must cite multiple relevant legal bases to enhance the authority of your answer
   - **Particularly important: Prioritize provisions from the local knowledge base**

2. Case Analysis:
   - Provide at least one specific case illustrating the application of the law
   - Case analysis must include: case facts, legal application, reasoning for judgment
   - Explain how the case relates to specific legal provisions in the knowledge base
   - If there are relevant precedents in the knowledge base, prioritize those; otherwise construct reasonable examples

3. Reasoning Analysis:
   - Explain how legal provisions apply to the user's specific question
   - Provide legal reasoning process, from legal principles to specific application
   - When there are multiple legal interpretations, explain different viewpoints and their bases

4. Format and Structure:
   - Use Markdown format to organize content, including heading levels (#, ##, ###)
   - Use **bold** to mark important concepts or conclusions
   - Use > quote blocks to quote legal text
   - Use ordered and unordered lists to organize information
   - Use tables appropriately to present comparative information
   - Code blocks can be used to display specific legal clause formats

## Data Sources and Citations:
- Prioritize legal knowledge retrieved from the local knowledge base (identifiable from the "source" tag)
- Note: Retrieved content will be marked with [Source: xxx] at the top, this is an internal tag, do not display this tag in your answer
- If there is no relevant information in the local knowledge base, the system will automatically query the internet, in which case you should:
  1. Clearly mark "The following content is from internet search results"
  2. Provide information sources or reference websites when possible
  3. Remind users that internet information may not be as accurate as official legal provisions, and suggest further verification
- Do not fabricate legal provisions, maintain accurate citation of legal materials
- When user questions involve multiple legal areas, comprehensively cite laws from all relevant areas

## Professional Requirements:
- Always maintain accuracy and professionalism in legal terminology
- Avoid making absolute statements that might be misinterpreted as professional legal opinions
- For highly specialized or controversial legal questions, suggest users consult a professional lawyer
- Answers should be objective and neutral, without personal value judgments

If the user's legal question is not specific enough, proactively guide them to provide more contextual information for more accurate legal analysis.
`;

const TASK = `
Please explain the provisions regarding contract validity, especially what circumstances would make a contract invalid? Can you provide an example of an actual case?
`;

// Define output path
const outputPath = path.join(process.cwd(), 'output');
// Ensure output directory exists
if (!fs.existsSync(outputPath)) {
  fs.mkdirSync(outputPath, { recursive: true });
}

const fetchMCP = new MCPClient("mcp-server-fetch", "uvx", ['mcp-server-fetch']);
const fileMCP = new MCPClient("mcp-server-file", "npx", ['-y', '@modelcontextprotocol/server-filesystem', outputPath]);

async function main() {
    // Get context from local knowledge base
    const context = await retrieveContext();

    // Agent
    const agent = new Agent('gpt-4o-mini', [fetchMCP, fileMCP], LEGAL_SYSTEM_PROMPT, context);
    await agent.init();
    
    // Get user query
    const userQuery = process.env.USER_QUERY || TASK;
    
    // Get response
    const response = await agent.invoke(userQuery);
    
    console.log('Response generated');
    
    await agent.close();
    return response;
}

// If this file is run directly, execute main function
if (typeof require !== 'undefined' && require.main === module) {
    main().catch(console.error);
} 

// Export main function for API calls
export { main };

async function retrieveContext() {
    // RAG
    const embeddingRetriever = new EmbeddingRetriever("text-embedding-3-small");
    const knowledgeDir = path.join(process.cwd(), 'knowledge');
    
    // Load all legal related files
    try {
        const files = fs.readdirSync(knowledgeDir);
        const legalFiles = files.filter(file => file.endsWith('.md'));
        
        console.log(`Found the following knowledge base files: ${legalFiles.join(', ')}`);
    
    for (const file of legalFiles) {
        const filePath = path.join(knowledgeDir, file);
        if (fs.existsSync(filePath)) {
            const content = fs.readFileSync(filePath, 'utf-8');
                await embeddingRetriever.embedDocument(content, file); // Pass filename
                console.log(`Embedded document: ${file}`);
        } else {
                console.error(`File not found: ${file}`);
            }
        }
    } catch (error) {
        console.error('Error loading knowledge base files:', error);
        }
    
    // Get user query
    const userQuery = process.env.USER_QUERY || TASK;
    
    // Retrieve relevant content from local knowledge base
    const retrievedContexts = await embeddingRetriever.retrieve(userQuery, 5); // Increase result count
    const context = retrievedContexts.join('\n\n');
    
    logTitle('CONTEXT');
    console.log(context);
    
    return context;
}