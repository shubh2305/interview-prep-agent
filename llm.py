from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from utils import clean_response
import json

class LLM:
    def __init__(self):
        self.llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            task="conversational",
            huggingfacehub_api_token="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",
            max_new_tokens=2000
        )
        self.chat_model = ChatHuggingFace(llm=self.llm)
        self.chat_model_with_retry = self.chat_model.with_retry(stop_after_attempt=3)

    async def invoke(self, messages):
        raw_response = await self.chat_model_with_retry.ainvoke(messages)
        cleaned_response = clean_response(raw_response.content)
        return json.loads(cleaned_response)
