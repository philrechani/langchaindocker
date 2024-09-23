import { queryOllama } from "./app.js";
import { ChromaClient } from 'chromadb';


async function initializeChromaCollection() {

const client = new ChromaClient({ path: "http://chroma-db:8000" });
  try {
    const collectionName = "test_queries";
    let collection;

    const existingCollections = await client.listCollections();
    const collectionExists = existingCollections.some(c => c.name === collectionName);

    if (collectionExists) {
      console.log(`Collection ${collectionName} already exists. Using the existing collection.`);
      collection = await client.getCollection({ name: collectionName });
    } else {
      console.log(`Creating new collection: ${collectionName}`);
      await client.createCollection({ name: collectionName });
      collection = await client.getCollection({ name: collectionName });
    }
    return collection
  } catch (error) {
    console.error("Error creating Chroma collection:", error);
  }
}

// Call this function before using chromaCollection
const chromaCollection = await initializeChromaCollection();


const test_url = 'https://abc7chicago.com/post/crowdstrike-outage-impacting-flight-status-ohare-midway-airports/15070771/'

const test_query = 'can you tell me about what happened during the crowdstrike outage in chicago?'

const result = await queryOllama(chromaCollection,test_query,'url',test_url)
console.log(result)