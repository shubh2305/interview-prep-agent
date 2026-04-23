from pymongo import AsyncMongoClient
from dotenv import load_dotenv
import os
load_dotenv()


class Database:
    def __init__(self):
        self.client = AsyncMongoClient(os.getenv("MONGOD_URI"))
        self.db = self.client["expense-tracker"]
        self.collection = self.db["resume-vector-store"]

    async def insert_document(self, document):
        await self.collection.insert_one(document)

    async def get_document(self, document_id):
        return await self.collection.find_one({"user_id": document_id})

    async def update_document(self, document_id, document):
        await self.collection.update_one({"_id": document_id}, {"$set": document})
        