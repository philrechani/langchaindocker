import { ChromaClient } from 'chromadb';
import axios from 'axios';

async function generateEmbedding(text) {
  try {
    console.log(`Generating embedding for text: "${text}"`);
    const response = await axios.post('http://llm-server:11434/api/embeddings', {
      model: "nomic-embed-text",
      prompt: text
    });
    //console.log('Full Ollama response:', JSON.stringify(response.data, null, 2));

    if (response.data && response.data.embedding && Array.isArray(response.data.embedding)) {
      console.log(`Embedding generated successfully. Length: ${response.data.embedding.length}`);
      return response.data.embedding;
    } else {
      console.error('Invalid embedding response:', response.data);
      return null;
    }
  } catch (error) {
    console.error('Error generating embedding:', error.message);
    if (error.response) {
      console.error('Error response:', error.response.data);
    }
    return null;
  }
}

async function testEmbedding() {
  const testText = "This is a test.";
  const embedding = await generateEmbedding(testText);
  if (embedding) {
    console.log("Test embedding generated successfully.");
  } else {
    console.log("Failed to generate test embedding.");
  }
}

async function testChromaDB() {
  await testEmbedding(); // Test embedding generation before proceeding

  const client = new ChromaClient({ path: "http://chroma-db:8000" });

  try {
    const collectionName = "test_embeddings_collection";
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

    const documents = [
      "The quick brown fox jumps over the lazy dog",
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "ChatGPT is an AI language model created by OpenAI"
    ];
    const ids = ["doc1", "doc2", "doc3"];

    console.log("Adding documents to collection...");
    for (let i = 0; i < documents.length; i++) {
      console.log(`Processing document ${i + 1}/${documents.length}: "${documents[i]}"`);
      try {
        const embeddingResponse = await generateEmbedding(documents[i]);
        //console.log("Embedding response:", JSON.stringify(embeddingResponse));
        
        if (embeddingResponse && Array.isArray(embeddingResponse.embeddings) && embeddingResponse.embeddings.length > 0) {
          const embedding = embeddingResponse.embeddings[0];
          await collection.upsert({
            ids: [ids[i]],
            embeddings: [embedding],
            documents: [documents[i]]
          });
          //console.log(`Added document: ${ids[i]}`);
        } else {
          //console.error(`Invalid embedding for document: ${ids[i]}`, embeddingResponse);
        }
      } catch (error) {
        console.error(`Error processing document ${ids[i]}:`, error.message);
      }
    }

    console.log("Checking collection count...");
    const count = await collection.count();
    console.log(`Number of documents in collection: ${count}`);

    if (count === 0) {
      console.error("No documents were added to the collection. Stopping execution.");
      return;
    }

    console.log("Querying the collection...");
    const queryTexts = ["quick brown fox", "AI language model"];
    const queryEmbeddings = await Promise.all(queryTexts.map(generateEmbedding));
    const results = await collection.query({
      queryEmbeddings: queryEmbeddings,
      nResults: 2
    });

    console.log("Raw query results:", JSON.stringify(results, null, 2));

    console.log("Query Results:");
    queryTexts.forEach((query, i) => {
      console.log(`\nQuery: ${query}`);
      if (results && results.documents && Array.isArray(results.documents[i])) {
        results.documents[i].forEach((doc, j) => {
          const distance = results.distances && results.distances[i] ? results.distances[i][j] : 'N/A';
          console.log(`  Result ${j + 1}: ${doc} (Distance: ${distance})`);
        });
      } else {
        console.log("  No results found or unexpected result structure");
      }
    });

    console.log("\nFinal Collection Info:");
    console.log(`Number of documents: ${await collection.count()}`);

    console.log("\nAll Collections:");
    console.log(await client.listCollections());

  } catch (error) {
    console.error("An error occurred:", error.message);
  }
}

export { testChromaDB };