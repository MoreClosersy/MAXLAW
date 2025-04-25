import fs from 'fs';
import { main } from './index.js';

async function handleChatRequest() {
  try {
    // Get user query and response file path from environment variables
    const userQuery = process.env.USER_QUERY;
    const responseFile = process.env.RESPONSE_FILE;
    
    if (!userQuery) {
      throw new Error('Missing user query');
    }
    
    if (!responseFile) {
      throw new Error('Missing response file path');
    }
    
    console.log(`Processing user query: ${userQuery}`);
    console.log(`Response will be written to: ${responseFile}`);
    
    // Call main function to process the query
    const response = await main();
    
    // Write the response to the specified file, preserving all line breaks and formatting
    fs.writeFileSync(responseFile, response);
    
    console.log('Query processing completed');
  } catch (error) {
    console.error('Error processing chat request:', error);
    // If an error occurs, write the error message to the response file
    if (process.env.RESPONSE_FILE) {
      fs.writeFileSync(
        process.env.RESPONSE_FILE,
        `Sorry, an error occurred while processing your request: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
    process.exit(1);
  }
}

// Execute the function immediately
handleChatRequest(); 