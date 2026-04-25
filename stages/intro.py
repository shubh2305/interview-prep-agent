import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from llm import LLM
from database import Database
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

EVALUATE_ANSWER_SYSTEM_PROMPT = """
    You are an experienced technical interviewer.

    Evaluate the candidate’s introduction (“Tell me about yourself”) based on the following criteria.

    Scoring Rules:
    - Each category must be scored from 0 to 2
    - 0 = Poor, 1 = Average, 2 = Strong
    - Be strict and objective

    Evaluation Criteria:

    1. Structure
    - Is the answer well-organized (present → past → skills → highlight → goal)?
    - Avoids rambling and has a logical flow

    2. Relevance
    - Focuses on professional experience relevant to the role
    - Avoids personal or unrelated details

    3. Clarity
    - Easy to understand
    - Concise and well-articulated

    4. Ownership
    - Uses active voice ("I built", "I designed")
    - Demonstrates responsibility and contribution

    5. Impact
    - Mentions meaningful work, achievements, or outcomes
    - Prefer quantified or clearly stated impact

    Instructions:
    - Evaluate only based on the provided answer
    - Do not assume missing information
    - Be critical but fair

    Return output in JSON format ONLY:

    {
        "structure": <0-2>,
        "relevance": <0-2>,
        "clarity": <0-2>,
        "ownership": <0-2>,
        "impact": <0-2>,
        "total": <sum>,
        "level": "weak | average | strong",
        "feedback": "2-3 concise sentences explaining strengths and areas of improvement",
        "improvement_suggestions": [
            "specific actionable suggestion 1",
            "specific actionable suggestion 2"
        ]
    }
"""

class IntroState(BaseModel):
    resume_summary: str = Field(default=""),

class Intro:
    def __init__(self, state: IntroState):
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
        return response

    async def evaluate_answer(self, answer: str):
        messages = [
            SystemMessage(content=EVALUATE_ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"""
                Answer: {answer}
            """),
        ]
        evaluation = await self.llm.invoke(messages)
        
        return evaluation

# asyncio.run(
#     Intro(state={"score": 100, "strengths": ["Python", "SQL", "Data Analysis"], "weaknesses": ["Java", "C++", "JavaScript"], "resume_summary": "Shubham is a software engineer with 5 years of experience in Python and SQL. He has a Bachelor's degree in Computer Science from the University of California, Berkeley."}).intro()
# )