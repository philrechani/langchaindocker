import { Ollama } from "@langchain/community/llms/ollama";
import { processDocument } from "./loadUrl.js";
import { RetrievalQAChain } from "langchain/chains";
import Vectra from 'vectra';
import {LocalIndex} from 'vectra'

process.stdin.setEncoding('utf8');

async function initializeVectra() {
    const index = new LocalIndex(path.join(__dirname, '..', 'index'))
    const chromaClient = new Vectra();

    try {
        const collectionName = 'test_collection';
        const collection = await chromaClient.ceateCollection({name: collectionName});
        console.log('Collection initialized:', collection);
        return collection;
    } catch (error) {
        console.error('Error initializing collection:', error);
        throw error;
    }
}


console.log('Type "help" for a list of commands')

let inputBuffer = '';

async function processInput() {

    const collection = await initializeChromaDB()

    const OLLAMA_SERVICE_URL = 'http://llm-server:11434';

    const ollama = new Ollama({
        baseUrl: OLLAMA_SERVICE_URL,
        model: "mistral",
    });

    while (true) {
        const chunk = await new Promise((resolve) => {
            process.stdin.once('data', resolve)
        })

        inputBuffer += chunk

        if (inputBuffer.includes('\n')) {
            const inputs = inputBuffer.split('\n')
            for (let i = 0; i < inputs.length - 1; i++) {
                const input = inputs[i].trim()

                if (input === 'exit') {
                    console.log('Exiting...')
                    process.exit(0)
                } else if (input === 'help') {
                    console.log('we all need help')
                } else if (input === 'load') {
                    console.log('paste your link below')
                    const urlChunk = await new Promise((resolve) => {
                        process.stdin.once('data', resolve);
                    });

                    const url = urlChunk.trim();
                    processDocument(url, collection)
                } else if (input === 'chat') {

                    const llmQuery = await new Promise((resolve) => {
                        process.stdin.once('data', resolve);
                    });



                    const retriever = collection.asRetriever();
                    const chain = RetrievalQAChain.fromLLM(ollama, retriever);
                    const result = await chain.call({ query: llmQuery });
                    console.log(result.text)
                }


            }
            inputBuffer = inputs[inputs.length - 1]
        }
    }
}

processInput()