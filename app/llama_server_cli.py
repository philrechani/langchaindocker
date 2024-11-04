import sys
import os
import rag_functions as rag
import subprocess
import time
import signal
import platform

try:
    from config.CONFIG import MODEL_PATH, EXE_PATH, PERSIST_DIRECTORY
    print('successfully loaded custom CONFIG')
except Exception as e:
    print(e)
    print('loaded default configuration')
    from config.DEFAULT_CONFIG import MODEL_PATH, EXE_PATH, PERSIST_DIRECTORY

# Get the current directory
current_dir = os.path.dirname(os.path.abspath('__file__'))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import llama_cpp_api as llapi

API = llapi.LlamaAPI()
API.set_model_path(MODEL_PATH)
API.set_exe_path(EXE_PATH)

#persistent database
client, collection = rag.connect_to_collection(local=True, name='home',persist_directory=PERSIST_DIRECTORY)

#add document to database (check)

#query document database 

#query llm (in chat)

#query llm with automatic context (in ask, should have a separate search feature)

#query llm with specific context

#query llm with websearch context


command = ''
process = API.initialize_server(type='completion')
server = 'completion'
print('Welcome! Type /help for help\n')
while command != '/exit':
    
    command = input('> ')
    func = command.split(' ')[0]
    params = command.split(' ')[1:]
    
    if func == '/load':
        
        if params[0] == '--file':
            path = params[1]
            pages_and_texts = rag.open_and_read_pdf(path)
            
        elif params[0] == '--url':
            url = params[1]
            
            pages_and_texts = pages_and_texts = rag.process_webpage_content(rag.load_webpage_requests(url))
            
        else:
            print('PLEASE PROVIDE A FLAG (--file or --url)')
            continue
        
        if not server or server != 'embed':
            process = API.initialize_server(type='embed')
            server = 'embed'
            
        pages_and_texts_filtered = rag.apply_spacy_nlp_filtered(pages_and_texts)
        pages_and_texts = rag.chunk_sentences(pages_and_texts_filtered, num_sentence_chunk_size=10)
        pages_and_chunks = rag.restructure_chunks(pages_and_texts)
        pages_and_chunks = rag.filter_pages_and_texts(pages_and_chunks,30)
        embedded_pages_and_chunks = rag.apply_embeddings_api(API,pages_and_chunks)
        
        if params[0] == '--file':
            path = params[1]
            rag.add_to_collection(embedded_pages_and_chunks, collection, path = path, embedding_model_name=EXE_PATH)
            
        elif params[0] == '--url':
            url = params[1]
            
            rag.add_to_collection(embedded_pages_and_chunks, collection, url = url, embedding_model_name=EXE_PATH)
        
        
            
    if func == '/store':
        pass
    
    
    if func == '/ask':
        query = ' '.join(params[1:])
        if not server or server != 'embed':
            process = API.initialize_server(type='embed')
            server = 'embed'
        
        context_items, results = rag.query_collection_api(API,query,collection)
        
        if not server or server != 'completion':
            process = API.initialize_server(type='completion')
            server = 'completion'
    
        context = "- " + "\n- ".join(context_items)
        context_prompt =f'With this context: {context}, answer the following prompt: {query}'
        response = API.completion(context_prompt)
        print('---\n')
        print(response['content'])
        print('---\n')
        print(context_items)
        if False:
            for item in context_items:
                print(item)
    
    # talk without context
    if func == '/chat':
        query = ' '.join(params[1:])
        if not server or server != 'completion':
            process = API.initialize_server(type='completion')
            server = 'completion'
        response = API.completion(query)
        print(response['content'])
            
    if func == '/help':
        print("""
              /load\n
              The load command will find the content and apply the embeddings and automatically store it to the ChromaDB
              \tFlags: 
              \t\t--file - for local pdf only at the moment
              \t\t--url  - scrapes the text of a website provided
              /ask
              Prompt the LLM by first searching the database
              """)
        
print('Shutting down previous server')
try:
    # On Windows, we need to kill the entire process group
    if platform.system() == 'Windows':
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    
    # Wait for up to 5 seconds for the process to end
    for _ in range(50):  # 50 * 0.1 seconds = 5 seconds
        if process.poll() is not None:
            break
        time.sleep(0.1)
    else:
        # If it's still running after 5 seconds, force kill
        print("Process didn't terminate, forcing kill...")
        if platform.system() == 'Windows':
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
except Exception as e:
    print(f"Error shutting down previous server: {e}")