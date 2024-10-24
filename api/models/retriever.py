from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

def get_vector_retriever(collection):
    ATLAS_VECTOR_SEARCH_INDEX_NAME = 'vector_index'
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=[], 
        embedding=embedding_model, 
        collection=collection, 
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )
    retriever = vector_search.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    return retriever
