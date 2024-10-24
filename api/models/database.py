import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def connect_mongoDB():
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    uri = f'mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@cluster0.5sbsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    try:
        client = MongoClient(uri)
        db = client['CCandGender']
        collection = db['embeddings']
        print("Connection successful!")
    except Exception as e:
        print(f"Error: {e}")
    return collection
