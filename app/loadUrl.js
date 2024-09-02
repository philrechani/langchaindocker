import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { TensorFlowEmbeddings } from "@langchain/community/embeddings/tensorflow";


export async function processDocument(url,vectra) {
  const loader = new CheerioWebBaseLoader(url);
  const data = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 20
  });
  const splitDocs = await textSplitter.splitDocuments(data);

  const embeddingFunction = new TensorFlowEmbeddings()
  const embeddings = await Promise.all(splitDocs.map(doc => embeddingFunction.embed(doc)));

  //const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new TensorFlowEmbeddings());
  splitDocs.forEach((doc,index)=>{
    vectra.add(`doc_${index}`, embeddings[doc],doc)
  })


  await collection.add({
    ids: splitDocs.map((_,index) => `doc_${index}`),
    documents: splitDocs,
    embeddings: embeddings
  })
}

