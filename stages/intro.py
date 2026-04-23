from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from llm import LLM
from database import Database
from interview import InterviewState
import asyncio

INTRO_SYSTEM_PROMPT = """
    You are a friendly but rigorous FAANG-level technical interviewer conducting a structured interview with a candidate.

    Your role is to generate EXACTLY ONE interview question at a time based on the current interview stage, the candidate’s resume, and prior conversation context.
    Also this the first question in the interview so introduce the candidate for the interview.
    Stage Guidelines:

    INTRO:
    - Ask questions to understand the candidate’s background.
    - Focus on education, experience summary, and introduction.
    - Purely about candidate.
    - Example topics:
    - Tell me about yourself
    - Walk me through your background
    - What did you study and why?

    General Rules:
    - Generate EXACTLY ONE question.
    - Keep the question concise, natural, and conversational.
    - Sound human, warm, and professional.
    - Ask realistic interviewer-style questions.
    - Focus on understanding the candidate’s real experience, decisions, tradeoffs, and technical depth.
    - Ask the question from the given stage

    Do NOT:
    - Generate multiple questions
    - Generate large open-ended system design scenarios
    - Generate hypothetical architecture design prompts unless explicitly requested
    - Generate DSA/coding questions
    - Sound robotic or overly formal
    - Generate questions based on skills and languages

    Return the question in the following JSON format:
    {
        "welcome_message": "2 sentences to introduce yourself and welcome the candidate"
        "question": "Interview question"
    }
"""

class Intro:
    def __init__(self, state: InterviewState):
        self.llm = LLM()
        self.database = Database()
        self.state = state

    async def intro(self):
        user_prompt = """
            Candidate Summary:
            {resume_summary}
        """
        template = ChatPromptTemplate.from_messages([
            SystemMessage(content=INTRO_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        messages = template.format_messages(
            resume_summary=self.state['resume_summary']
        )

        response = await self.llm.invoke(messages)
        print(response)
        return response

# asyncio.run(
#     Intro(state={"score": 100, "strengths": ["Python", "SQL", "Data Analysis"], "weaknesses": ["Java", "C++", "JavaScript"], "resume_summary": "Shubham is a software engineer with 5 years of experience in Python and SQL. He has a Bachelor's degree in Computer Science from the University of California, Berkeley."}).intro()
# )