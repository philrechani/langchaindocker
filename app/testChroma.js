import "@tensorflow/tfjs-node";
import { ChromaClient } from 'chromadb';
import { TensorFlowEmbeddings } from "@langchain/community/embeddings/tensorflow";

async function runChromaTest() {
    try {
        console.log("Connecting to ChromaDB...");
        const client = new ChromaClient('http://chroma-db:5000');

        console.log("Creating embedding function...");
        const embeddingFunction = new TensorFlowEmbeddings();

        console.log("Creating collection...");
        const collection = await client.getOrCreateCollection(
            {
                name: "testChroma",
                metadata: {},
                embeddingFunction: embeddingFunction
            });

        console.log("Adding documents...");
        await collection.add(
            ["id1", "id2"],
            undefined,
            [{ "source": "my_source" }, { "source": "my_pants" }],
            ['what is the meaning of life?', 'whatever is true']
        );

        console.log("Querying collection...");
        const results = await collection.query(
            undefined,
            2,
            undefined,
            ['What is meaning?']
        );

        console.log("Results:", results);
    } catch (error) {
        console.error("Error details:", error);
        if (error.cause) {
            console.error("Cause:", error.cause);
        }
    }
}

runChromaTest();