# ==============================
# 🧠 Qdrant Cloud RAG Tool (OpenAI)
# ==============================
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

import os
from dotenv import load_dotenv
load_dotenv()
# os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")  
qdrant_url = os.environ['QDRANT_ENDPOINT']
qdrant_key = os.environ['QDRANT_API_KEY']
class RAGToolQdrantCloud:
    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()   

        self.client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_key,
        )
        self.db = None

    def build_index(self, csv_paths: list):
        """Create or rebuild a Qdrant Cloud collection."""
        all_docs = []
        for csv_path in csv_paths:
            loader = CSVLoader(csv_path)
            docs = loader.load()
            all_docs.extend(docs)
        # print(len(all_docs))
        self.db = QdrantVectorStore.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            url=qdrant_url, 
            api_key=qdrant_key,
            collection_name=self.collection_name,
            force_recreate=True
        )
        print(f"✅ Cloud Qdrant index built with {len(all_docs)} docs in '{self.collection_name}'.")

    def add_to_index(self, new_csv_path: str):
        """Add new documents to the existing cloud collection."""
        loader = CSVLoader(new_csv_path)
        new_docs = loader.load()

        db = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            url=qdrant_url, 
            api_key=qdrant_key,
                        
        )
        db.add_documents(new_docs)
        print(f"✅ Added {len(new_docs)} new docs to collection '{self.collection_name}' on Qdrant Cloud.")

# ==============================
# 🚀 Example Usage
# ==============================
rag_tool = RAGToolQdrantCloud("MentalHealthData")

rag_tool.build_index([
    "/content/Therapist_answers.csv",
    "/content/chat.csv",
    "/content/sentment_classification.csv"
])