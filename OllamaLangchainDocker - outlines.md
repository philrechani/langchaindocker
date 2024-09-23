collection parameters:
ids: name of the document
embeddings: embedding of the document
documents: actual document text
(link): (if available, link to the document)

cli.js:
needs url
-if url then generate embedding from CheerioBaseLoader
if document text
-generate based directly through text

(a flexible function would determine the kind of input from the value of the name, or a flag) (flag would be most straightforward)

needs name
this will be the document id


url -> embedding, document -> collection['embeddings','document','link']

filepath -> embedding, document -> collection['embeddings','document','link']

string -> embedding, document -> collect['embeddings','document']
