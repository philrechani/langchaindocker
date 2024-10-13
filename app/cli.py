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

command = ''
print('loading llm...')
llm_model, tokenizer = load_llm()
print('loading the embedding model...')
embedding_model = load_embedding()
print('connecting to chromadb...')
chroma_client, collection = connect_to_collection()
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
            embedded_pages_and_chunks = apply_pipeline(pages_and_texts)
            add_to_collection(embedded_pages_and_chunks, collection, path = path)
            
        elif params[0] == '--url':
            url = params[1]
            text = load_webpage_requests(url)
            pages_and_texts = process_webpage_content(text)
            embedded_pages_and_chunks = apply_pipeline(pages_and_texts)
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