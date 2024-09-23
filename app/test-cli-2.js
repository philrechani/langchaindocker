import { queryOllama } from "./app.js";
import { ChromaClient } from 'chromadb';

async function runTest() {
  const chromaClient = new ChromaClient({ 
    path: "http://chroma-db:8000",
    fetchOptions: {
      headers: {
        'Content-Type': 'application/json',
      },
    },
  });
  let chromaCollection;

  async function initializeChromaCollection() {
    try {
      chromaCollection = await chromaClient.getOrCreateCollection("test_queries");
      console.log("Collection created or retrieved successfully");
    } catch (error) {
      console.error("Error creating Chroma collection:", error);
      if (error.response) {
        console.error("Response status:", error.response.status);
        console.error("Response data:", await error.response.text());
      } else {
        console.error("Error details:", error.message);
      }
      throw error; // Re-throw the error to stop execution
    }
  }

  await initializeChromaCollection();

  const test_url = 'https://abc7chicago.com/post/crowdstrike-outage-impacting-flight-status-ohare-midway-airports/15070771/';
  const test_query = 'can you tell me about what happened during the crowdstrike outage in chicago?';

  try {
    const result = await queryOllama(chromaCollection, test_query, 'url', test_url);
    console.log(result);
  } catch (error) {
    console.error("Error in queryOllama:", error);
  }
}

runTest().catch(error => console.error("Test failed:", error));