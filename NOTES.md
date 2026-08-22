chunker.py -> performing sectioning of the doucment, relying on PyMuPDF Python library
    - 1600 char limit of chunks based on prose (text in the form of paragraphs) --> improves semantic similarity search through chunks rather than portions of text too large
    - 200 char overlap allows incomplete sentences/phrases/words to be picked up for creating quality chunks
    - tables get chunked as their own
    - raising errors if there is an empty text content on a page
    

retrieval.py -> forming the relevant chunks to the user's question based on semantic similarity
    - semantic similarity (cosine similarity): pgvector done with HNSW algorithm + tsvector for maintaing similarity of words and cutting filler words
    - 30 total chunks that have the highest cosine similarity and 30 highest ranked tsrank scoring of query to the documents chunks (term frequency)
    - RRF: scores each of the 60 chunks --> top 20 are what gets returned
    - pgvector supports HNSW and fast and accuracy
    - TSrank same thing its already integrated in pgvector
