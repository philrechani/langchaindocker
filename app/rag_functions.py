import pymupdf as fitz
from spacy.lang.en import English 
import os
import requests
from tqdm.auto import tqdm 
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from bs4 import BeautifulSoup

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available 

from huggingface_hub import login
from rank_bm25_local.rank_bm25 import BM25Okapi

nlp = English()
embedding_model_name = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(model_name_or_path=embedding_model_name, 
                                      device="cuda")
nlp.add_pipe("sentencizer")

def iterator(obj, istqdm = False):
    if istqdm:
        return tqdm(obj)
    else:
        return obj
    
def load_embedding(name = embedding_model_name,cuda_enabled = False):
    embedding_model_name = "all-mpnet-base-v2"
    device = 'cpu'
    if cuda_enabled:
        device = 'cuda'
    embedding_model = SentenceTransformer(model_name_or_path=embedding_model_name, 
                                        device=device)
    return embedding_model 

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)
    return cleaned_text

def open_and_read_pdf(pdf_path: str, istqdm = False) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  
    pages_and_texts = []
    for page_number, page in iterator(enumerate(doc), istqdm):  
        text = page.get_text()  
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,  
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,
                                "text": text})
    return pages_and_texts

def load_webpage_requests(url: str) -> str:
    """
    Loads a webpage using the requests library.

    Parameters:
        url (str): The URL of the webpage to be loaded.

    Returns:
        str: The text content of the webpage.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def process_webpage_content(text: str) -> list[dict]:
    """
    Processes the webpage content and collects statistics.

    Parameters:
        text (str): The text content of the webpage.

    Returns:
        List[Dict]: A list containing a dictionary with character count, word count,
        sentence count, token count, and the extracted text.
    """

    return [{
        "page_number": 1,
        "char_count": len(text),
        "word_count": len(text.split()),
        "sentence_count_raw": len(text.split('. ')),
        "token_count": len(text) / 4,
        "text": text
    }]

def apply_spacy_nlp(pages_and_texts: dict, istqdm = False) -> dict:
    for item in iterator(pages_and_texts, istqdm):
        item["sentences"] = list(nlp(item["text"]).sents)
        
        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        
        # Count the sentences 
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts

def split_list(input_list: list, 
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def chunk_sentences(pages_and_texts: dict, num_sentence_chunk_size: int, istqdm = False) -> dict:
    # Loop through pages and texts and split sentences into chunks+
    for item in iterator(pages_and_texts, istqdm):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                            slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts

def restructure_chunks(pages_and_texts: dict, istqdm = False) -> dict:
    pages_and_chunks = []
    
    for item in iterator(pages_and_texts, istqdm):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            
            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
            
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks


def filter_pages_and_texts(pages_and_chunks: dict, 
                           min_token_length: int) -> dict:
    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    return pages_and_chunks_over_min_token_len

def get_embedding(text):
    url = "http://localhost:49152/api/embeddings"
    payload = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    response = requests.post(url, json=payload)
    return response.json()['embedding']

def apply_ollama_embeddings(pages_and_chunks_over_min_token_len: dict, 
                     istqdm = False) -> dict:
    
    for item in iterator(pages_and_chunks_over_min_token_len, istqdm):
        item["embedding"] = get_embedding(item['sentence_chunk'])
    
    return pages_and_chunks_over_min_token_len

def apply_embeddings(pages_and_chunks_over_min_token_len: dict, 
                     embedding_model: SentenceTransformer,
                     istqdm = False,
                     flatten = False) -> dict:
    if not flatten:
        for item in iterator(pages_and_chunks_over_min_token_len, istqdm):
            item["embedding"] = embedding_model.encode(item["sentence_chunk"],
                                                batch_size=32,
                                                convert_to_tensor=True)
    else:
        text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
        embeddings = embedding_model.encode(text_chunks)
        print(embeddings)
        for embedding, item in iterator(zip(embeddings, pages_and_chunks_over_min_token_len), istqdm):
            item["embedding"] = embedding
    
    return pages_and_chunks_over_min_token_len

def connect_to_collection(host='localhost', port=49151, name='testing_python_creation'):
    chroma_client = chromadb.HttpClient(host=host, port=port)
    collection = chroma_client.get_or_create_collection(name=name)
    return chroma_client, collection

def add_to_collection(embedded_pages_and_chunks,
                      collection,
                      embedding_model_name = embedding_model_name,
                      path =None, 
                      url =None,
                      text = None):
    link = ''
    if path:
        # this is where plain text and pdf should be distinguished
        file_name = os.path.basename(path)
        link = path
    if url:
        file_name = url
        link = url
    if text:
        link = 'plain text'
        
    collection.add(
        documents=[item['sentence_chunk'] for item in embedded_pages_and_chunks],
        metadatas=[{
            'page_number': item['page_number'],
            'char_count': item['chunk_char_count'],
            'word_count': item['chunk_word_count'],
            'token_count': item['chunk_token_count'],
            'embedding_model': embedding_model_name,
            'link': link
        } for item in embedded_pages_and_chunks],
        ids=[f"{file_name}_chunk_{i}" for i in range(len(embedded_pages_and_chunks))],
        embeddings=[item['embedding'].tolist() for item in embedded_pages_and_chunks]
    )
    
def query_collection(query, 
                     collection, 
                     embedding_model,
                     NLP = False):
    query_embedding = embedding_model.encode(query)
    
    # Create a list of context items
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=30
    )
    # get the score?
    context_items = [result for result in results['documents'][0]]
    if NLP:
        
        tokenized_context_items = [doc.split(" ") for doc in context_items ]
        bm25 = BM25Okapi(tokenized_context_items)
        tokenized_query = query.split(" ")
        context_items = bm25.get_top_n(tokenized_query,context_items)
        
    return context_items, results

def apply_pipeline(pages_and_texts):
    pages_and_texts = apply_spacy_nlp(pages_and_texts)
    pages_and_texts = chunk_sentences(pages_and_texts,num_sentence_chunk_size = 10 )
    pages_and_chunks = restructure_chunks(pages_and_texts)
    pages_and_chunks = filter_pages_and_texts(pages_and_chunks,30)
    embedded_pages_and_chunks = apply_embeddings(pages_and_chunks,embedding_model,flatten=True)
    return embedded_pages_and_chunks

def load_llm(token):
    
    login(token)
    
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")
    # Note: the following is Gemma focused, however, there are more and more LLMs of the 2B and 7B size appearing for local use.
    if gpu_memory_gb < 5.1:
        print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
    elif gpu_memory_gb < 8.1:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
        use_quantization_config = True 
        model_id = "google/gemma-2b-it"
    elif gpu_memory_gb < 19.0:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
        use_quantization_config = False 
        model_id = "google/gemma-2b-it"
    elif gpu_memory_gb > 19.0:
        print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
        use_quantization_config = False 
        model_id = "google/gemma-7b-it"

    print(f"use_quantization_config set to: {use_quantization_config}")
    print(f"model_id set to: {model_id}")


    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16)

    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    print(f"[INFO] Using attention implementation: {attn_implementation}")

    model_id = model_id 
    print(f"[INFO] Using model_id: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                    torch_dtype=torch.float16,
                                                    quantization_config=quantization_config if use_quantization_config else None,
                                                    low_cpu_mem_usage=False,
                                                    attn_implementation=attn_implementation) 

    if not use_quantization_config:
        llm_model.to("cuda")
        
    return llm_model, tokenizer

def prompt_formatter(query: str, 
                     context_items: list[dict],
                     tokenizer,
                     prompt: str) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join(context_items)

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    

    # Update base prompt with context items and query   
    base_prompt = prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

def ask(query,
        embedding_model,
        collection,
        tokenizer,
        llm_model,
        prompt,
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True, 
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """
    
    context_items, results = query_collection(query,collection, embedding_model)
    context_items = [result for result in results['documents'][0]]

    # Add score to context item
    """ for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU  """
        
    prompt = prompt_formatter(query=query,
                              context_items=context_items,
                              tokenizer=tokenizer,
                              prompt=prompt)
    
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    if return_answer_only:
        return output_text
    
    return output_text, context_items