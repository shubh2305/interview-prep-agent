from pymongo import AsyncMongoClient
from dotenv import load_dotenv
load_dotenv()


class Database:
    def __init__(self):
        self.client = AsyncMongoClient(os.getenv("MONGOD_URI"))
        self.db = self.client["expense-tracker"]
        self.collection = self.db["resume-vector-store"]

    def insert_document(self, document):
        self.collection.insert_one(document)

    def get_document(self, document_id):
        return self.collection.find_one({"user_id": document_id})

    def update_document(self, document_id, document):
        self.collection.update_one({"_id": document_id}, {"$set": document})
        