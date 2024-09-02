import { Ollama } from "@langchain/community/llms/ollama";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "@tensorflow/tfjs-node";
import { TensorFlowEmbeddings } from "@langchain/community/embeddings/tensorflow";
import { RetrievalQAChain } from "langchain/chains";

async function processDocument(url) {

  const loader = new CheerioWebBaseLoader(url);
  const data = await loader.load();

  // Split the text into 500 character chunks. And overlap each chunk by 20 characters
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 20
  });
  const splitDocs = await textSplitter.splitDocuments(data);

  // Then use the TensorFlow Embedding to store these chunks in the datastore
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new TensorFlowEmbeddings());
  return vectorStore

}
const OLLAMA_SERVICE_URL = 'http://llm-server:11434';

const ollama = new Ollama({
  baseUrl: OLLAMA_SERVICE_URL,
  model: "mistral",
});

//const answer = await ollama.invoke(`why is the sky blue?`);
//console.log(answer);


//update later


const vectorStore = await processDocument(process.argv[3])
const retriever = vectorStore.asRetriever();
const chain = RetrievalQAChain.fromLLM(ollama, retriever);
const result = await chain.call({ query: process.argv[2]});
console.log(result.text)