import { Ollama } from "@langchain/community/llms/ollama";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { Chroma } from "@langchain/community/vectorstores/chroma";
import "@tensorflow/tfjs-node";
import { TensorFlowEmbeddings } from "@langchain/community/embeddings/tensorflow";
import { RetrievalQAChain } from "langchain/chains";
/* import { testChromaDB, testQueryChromaDB, queryChromaDB, generateEmbedding } from './text-axios-3.js' */
import { Document } from '@langchain/core/documents'
import * as fs from 'node:fs';
import axios from 'axios';

async function generateEmbedding(text) {
  try {
    //console.log(`Generating embedding for text: "${text}"`);
    const response = await axios.post('http://llm-server:11434/api/embeddings', {
      model: "nomic-embed-text",
      prompt: text
    });
    console.log('Full Ollama response:', JSON.stringify(response.data, null, 2));

    if (response.data && Array.isArray(response.data.embedding)) {
      
      //console.log(`Embedding generated successfully. Length: ${response.data.embedding.length}`);
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

function getCurrentDate() {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');

  return `${year}-${month}-${day}`;
}

class CustomEmbeddings {
  async embedDocuments(texts) {
    return Promise.all(texts.map(text => generateEmbedding(text)));
  }

  async embedQuery(text) {
    return generateEmbedding(text);
  }
}

async function processDocument(chromaCollection, url = null, text = null) {
  let data;
  const customEmbeddings = new CustomEmbeddings();
  //loads the url
  if (url) {
    const loader = new CheerioWebBaseLoader(url);
    data = await loader.load();
  }

  if (text) {
    fs.readFile(filePath, 'utf8' ,(err,data) => {
      if (err) {
        console.error('Error reading file:', err);
        return;
      }
      data = new Document({
        pageContent: data,
        metadata: { source: url, date: getCurrentDate() }
      })
    })
    
  }

  // Split the text into 500 character chunks. And overlap each chunk by 20 characters
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 20
  });
  const splitDocs = await textSplitter.splitDocuments(data);

  // documents must be of format: new Document({pageContent: 'text here', metadata: {source: 'filesource.txt', date: 'date retrieved'}})
  //doc can only be Document or text
  //this shouldn't trigger given my current set up
  const documents = splitDocs.map(doc =>
    doc instanceof Document ? doc : new Document({
      pageContent: doc,
      metadata: { source: url, date: getCurrentDate() }
    })
  );


  const vectorStore = await Chroma.fromDocuments(
    documents,
    customEmbeddings,
    { collectionName: chromaCollection.name, url: "http://chroma-db:8000" }
  );
  return vectorStore
}

async function queryOllama(chromaCollection, query, flag, url=null,text=null) {
  const OLLAMA_SERVICE_URL = 'http://llm-server:11434';

  const ollama = new Ollama({
    baseUrl: OLLAMA_SERVICE_URL,
    model: "mistral",
  });

  let vectorStore;

  //const answer = await ollama.invoke(`why is the sky blue?`);
  //console.log(answer);
  //update later

  if (flag === 'url') {
    vectorStore = await processDocument(chromaCollection,url=url)
  }

  if (flag === 'text') {
    vectorStore = await processDocument(chromaCollection,text=text)
  }

  if (flag === 'doc') {

  }
  const retriever = vectorStore.asRetriever();
  const chain = RetrievalQAChain.fromLLM(ollama, retriever); //
  const result = await chain.call({ query: query });
  return result.text
}

export { queryOllama }