from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import uuid

from llm import LLM
from database import Database
from main import ResumeProcessorState

file_path = '/content/shubham_q_resume.pdf'

class ResumeProcessor:
    def __init__(self):
        self.llm = LLM()
        self.database = Database()

    async def process_resume(self, state: ResumeProcessorState):
        print("Processing resume...")
        chunks = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            chunking_strategy="by_title"
        )

        resume_content = ''
        for chunk in chunks:
            resume_content += '\n'.join([element.text for element in chunk.metadata.orig_elements])
            resume_content += '\n'

        process_resume_prompt_template = self.get_process_resume_template()
        messages = process_resume_prompt_template.format_messages(
            resume_content=resume_content
        )

        print("Invoking LLM...")
        resume_data = await self.llm.invoke(messages)
        resume_data["user_id"] = str(uuid.uuid4())
        print("Inserting resume data into database...")
        await self.database.insert_one(resume_data)
        state["resume_data"] = resume_data

        print("Resume processed successfully!")
        return state["resume_data"]


    def get_process_resume_template(self):
        user_prompt = """
            Resume:
            {resume_content}
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=PROCESS_RESUME_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])


