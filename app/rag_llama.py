from llama_cpp import Llama
import torch
import chromadb
from bs4 import BeautifulSoup
import requests
import pymupdf as fitz
from datetime import datetime
import re
import os
import sys

# Get the current directory
current_dir = os.path.dirname(os.path.abspath('__file__'))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
try:
    from config.CONFIG import PERSIST_DIRECTORY, MODEL_PATH
    print('successfully loaded custom CONFIG')
except Exception as e:
    print(e)
    print('loaded default configuration')
    from config.DEFAULT_CONFIG import PERSIST_DIRECTORY, MODEL_PATH


class RAG(Llama):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        
        self.current_mode = 'chat'
        self.context_size = self.kwargs['n_ctx']
        self.chroma_host = ''
        self.chroma_persist_directory = PERSIST_DIRECTORY
        self.current_collection_name = 'default_peanut_mushroom'
        self.current_client = None
        self.current_collection = None
        
    # LLM Functions  
        
    def _pca_(X, num_components):
        X_mean = torch.mean(X, 0)
        X = X - X_mean.expand_as(X)
        U, S, V = torch.svd(X.t(), some=True, compute_uv=True) #compute_
        return torch.mm(X, U[:, :num_components])
    
    def start_model(self):
        pass
    
    def reinitialize_model(self, mode):
        if mode == self.current_mode:
            return  # No need to reinitialize if already in the correct mode

        #this stuff needs to be flexible. i.e. it needs to go back to the set context size and only 1 for embedding
        if mode == 'embed':
            self.kwargs['embedding'] = True
            self.kwargs['vocab_only'] = False
            self.kwargs['n_ctx'] = 1  # Set context window to 1 for embeddings
        elif mode == 'chat':
            self.kwargs['embedding'] = False
            self.kwargs['vocab_only'] = False
            self.kwargs['n_ctx'] = self.context_size# Set appropriate context window for chat
        elif mode == 'tokenize':
            self.kwargs['embedding'] = False
            self.kwargs['vocab_only'] = True
            self.kwargs['n_ctx'] = self.context_size
            
        super().__init__(*self.args, **self.kwargs)
        self.current_mode = mode                                   
    
    def chat(self, prompt,**kwargs):
        self.reinitialize_model('chat')
        return super().__call__(prompt,max_tokens=None,**kwargs)
    
    def __call__(self, prompt,**kwargs):
        self.reinitialize_model('chat')
        return super().__call__(prompt,**kwargs)
    
    def ask(self,prompt, max_tokens= None, n_results=5, **kwargs):
        if self.current_collection is not None:
            self.reinitialize_model('embed')
            embeddings = super().create_embedding(prompt)['data'][0]['embedding']
            [0][0]
            
            try:
                results = self.current_collection.query(query_embeddings=embeddings,n_results = n_results)['documents']
                context = ' '.join(results[0])
                
            except Exception as e:
                print(e)
                print('No context found. Using no context')
                context = ''
            
            self.reinitialize_model('chat')
            prompt = f'Answer this query: {prompt}\n With the following context: {context}'
            
            return super().__call__(prompt,max_tokens=max_tokens, **kwargs), context
        else:
            print('First initialize the collection to ask with context')
        
    def set_template(self):
        pass
    
    # Knowledge Base Functions
    
    def load_page(self, text: str, page_number: int) -> dict:
        return {"char_count": len(text),
                'date_added': datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),
                "page_number": page_number,  
                "sentence_count": len(text.split(". ")),
                "token_count": len(text) / 4,
                "text": text,
                "word_count": len(text.split(" "))}
    
    def load_webpage_text(self, url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return [self.load_page(text)]
        
    def open_and_read_pdf(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)  
        pages = []
        for page_number, page in enumerate(doc):  
            text = page.get_text()
            if len(text) != 0:
                pages.append(self.load_page(text,page_number))
        return pages
    
    def split_text(self,text):
        # First, split by newlines
        lines = text.split('\n')
        
        # Then, split each line by '. '
        result = []
        for line in lines:
            result.extend(re.split(r'(?<=\.)\s', line))
        
        # Remove any empty strings
        result = [item.strip() for item in result if item.strip()]
        
        return result    
    
    def chunked(self, iterable, chunk_size):
        for i in range(0, len(iterable), chunk_size):
            yield iterable[i:i + chunk_size]
    
    #might consider doing it by paragraphs
    def process_pages(self, pages: list[dict], sentence_number = 5) -> list[dict]:
        self.reinitialize_model('tokenize')
        
        token_limit = self.kwargs['n_ctx']/2
        print('token_limit:',token_limit)
        for page in pages:
            page['sentences'] = []
            total_tokens = 0
            raw_text = page['text']  
            pieces = self.split_text(raw_text)
            
            for chunk in self.chunked(pieces, sentence_number):
                chunk_text = ' '.join(chunk)
                tokens = super().tokenize(chunk_text.encode('utf-8'))
                n_tokens = len(tokens)
                
                if n_tokens > token_limit:
                    n_tokens = token_limit
                    token_texts = [super().detokenize([token]).decode('utf-8').strip() for token in tokens]
                    for new_chunks in self.chunked(token_texts, token_limit):
                        chunk_text = ' '.join(new_chunks)
                total_tokens += n_tokens
                page['sentences'].append({
                            'char_count': len(chunk_text),
                            'date_added': page['date_added'],
                            'page_number': page['page_number'],
                            'text': chunk_text,
                            'token_count': n_tokens,
                            'sentence_count': len(chunk_text.split('. ')),
                            'word_count': len(chunk_text.split(' '))    
                        })
            page['page_token_count'] = total_tokens
        return pages
    
    def embed_pages(self, pages):
        self.reinitialize_model('embed')
        for page in pages:
            for sentence in page['sentences']:
                sentence['embedding'] = super().embed(sentence['text'])[0]
        return pages
    
    # ChromaDB Functions
    
    def initialize_collection(self, type = 'memory', collection_name = None):
        if collection_name is None:
            collection_name = self.current_collection_name
            
        if type == 'persist':
            self.current_client = chromadb.PersistentClient(path = self.chroma_persist_directory)
            
        if type == 'remote':
            self.current_client = chromadb.HttpClient()
            
        if type == 'memory':
            self.current_client = chromadb.Client()
            
        self.current_collection = self.current_client.get_or_create_collection(name = collection_name)
    
    def add_to_collection(self, pages: list[dict],
                      path = None, 
                      url = None,
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
            file_name = ''
            link = 'plain text'
        for j, page in enumerate(pages):
            self.current_collection.add(
                documents=[sentence['text'] for sentence in page['sentences']],
                metadatas=[{
                    'char_count': sentence['char_count'],
                    'page_number': sentence['page_number'],
                    'date_added': sentence['date_added'],
                    'sentence_count': sentence['sentence_count'],
                    'token_count': sentence['token_count'],
                    'word_count': sentence['word_count'],
                    
                    'embedding_model': self.kwargs['model_path'],
                    'link': link
                } for sentence in page['sentences']],
                ids=[f"{file_name}_chunk_{i}_{j}" for i in range(len(page['sentences']))],
                embeddings=[sentence['embedding'] for sentence in page['sentences']]
            )
        print('added successfully...')
    
    def query_collection(self, collection, query_embeddings = None, query_text = None, query_images = None, query_uris = None, n_results = None, where = None, where_document = None, include = None):
        
        params = {
            'query_embeddings': query_embeddings,
            'query_text': query_text,
            'query_images': query_images,
            'query_uris': query_uris,
            'n_results': n_results,
            'where': where,
            'where_document': where_document,
            'include': include
        }
        
        filitered_params = {k: v for k, v in params.items() if v is not None}
        
        return collection.query(**filitered_params)
    
    
    
    
    
    
    