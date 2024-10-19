from rag_functions import (
    apply_pipeline,
    load_llm,
    ask,
    load_webpage_requests,
    process_webpage_content,
    open_and_read_pdf,
    connect_to_collection,
    add_to_collection,
    load_embedding
)
from prompts.prompts import MIN_PROMPT
try:
    
    from config.CONFIG import HOST, PORT, NAME, HF_TOKEN, CUDA_ENABLED
    print('successfully loaded custom CONFIG')
except Exception as e:
    print(e)
    print('loaded default configuration')
    from config.DEFAULT_CONFIG import HOST, PORT, NAME, HF_TOKEN, CUDA_ENABLED

command = ''
print('loading llm...')
llm_model, tokenizer = load_llm(HF_TOKEN)
print('loading the embedding model...')
embedding_model = load_embedding(cuda_enabled=CUDA_ENABLED)
print('connecting to chromadb...')
chroma_client, collection = connect_to_collection(host=HOST,port=PORT,name=NAME)
print('ready!\n')


print('Welcome! Type /help for help\n')

while command != '/exit':
    command = input('> ')
    func = command.split(' ')[0]
    params = command.split(' ')[1:]
    if func == '/load':
        
        if params[0] == '--file':
            path = params[1]
            pages_and_texts = open_and_read_pdf(path)
            embedded_pages_and_chunks = apply_pipeline(pages_and_texts,embedding_model)
            add_to_collection(embedded_pages_and_chunks, collection, path = path)
            
        elif params[0] == '--url':
            url = params[1]
            text = load_webpage_requests(url)
            pages_and_texts = process_webpage_content(text)
            embedded_pages_and_chunks = apply_pipeline(pages_and_texts,embedding_model)
            add_to_collection(embedded_pages_and_chunks, collection, url = url)
        else:
            print('PLEASE PROVIDE A FLAG (--file or --url)')
            continue
        # more stuff here
            
    if func == '/store':
        pass
    
    if func == '/ask':
        query = ' '.join(params[1:])
        output_text, context_items = ask(query,
                                         embedding_model,
                                         collection,
                                         tokenizer,
                                         llm_model,
                                         prompt=MIN_PROMPT,
                                         return_answer_only=False)
        print('---\n')
        print(output_text)
        print('---\n')
        for item in context_items:
            print(item)
            
    if func == '/help':
        print("""
              /load\n
              The load command will find the content and apply the embeddings and automatically store it to the ChromaDB
              \tFlags: 
              \t\t--file - for local pdf only at the moment
              \t\t--url  - scrapes the text of a website provided
              
              """)