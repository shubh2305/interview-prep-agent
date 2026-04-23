import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from llm import LLM
from database import Database
import asyncio
import random

GENERATE_QUESTIONS_SYSTEM_PROMPT = """
    You are a friendly but rigorous FAANG-level technical interviewer conducting a structured interview with a candidate.

    Your role is to generate EXACTLY ONE interview question at a time based on the current interview stage, the candidate’s resume, and prior conversation context.
    You will be given candidates' summary, their projects, their strengths, their technical skills
    Ask questions according to the given topic.
    Adjust the difficulty of the question based on the given diffuculty.
    The difficulty will range from 0 - 10

    Stage Guidelines:
    - Ask questions about the candidate’s projects, work experience, and accomplishments.
    - Probe implementation details, architecture choices, challenges, tradeoffs, and impact.
    - Ask conceptual or practical technical questions based on technologies listed in the resume.
    - Probe understanding of tools, frameworks, databases, infrastructure, and design choices.
    - Example topics:
    - Walk me through this project
    - Why did you choose this architecture?
    - How did you optimize for cost/performance/scalability?
    - Why did you use MongoDB in this project?
    - What is Redis and when would you use it?
    - How does Kafka guarantee message ordering?

    General Rules:
    - Generate EXACTLY ONE question.
    - Keep the question concise, natural, and conversational.
    - Sound human, warm, and professional.
    - Ask realistic interviewer-style questions.
    - Focus on understanding the candidate’s real experience, decisions, tradeoffs, and technical depth.
    - Strictly ask questions based on the context.

    Do NOT:
    - Generate multiple questions
    - Generate large open-ended system design scenarios
    - Generate hypothetical architecture design prompts unless explicitly requested
    - Generate DSA/coding questions
    - Sound robotic or overly formal

    Return the question in the following JSON format:
    {{
        "question": "Interview question"
    }}
"""

EVALUATE_ANSWERS_SYSTEM_PROMPT = """
    You are a FAANG-level technical interviewer.

Evaluate the candidate's answer using the provided rubric.

Return valid JSON only.
Be objective and critical.
Give strengths and weaknesses in an array of single words
follow_up_topics should not be more than 5
Do not inflate scores.
Evaluate the answer considering the experience of the candidate and the difficulty level of the question

Return JSON:
{{
  "score": 0-100,
  "dimension_scores": {{
    "technical_depth": 0-25,
    "clarity": 0-20,
    "tradeoff_reasoning": 0-20,
    "ownership": 0-15,
    "problem_solving": 0-20
  }},
  "strengths": [],
  "weaknesses": [],
  "feedback": "",
  "follow_up_topics": []
}}
"""

class ResumeDeepDiveState(BaseModel):
    user_summary: str = Field(default=""),
    projects: list[dict] = Field(default_factory=list),
    strengths: list[str] = Field(default_factory=list),
    avg_score: float = Field(default=100.0),
    difficulty: int = Field(default=0),
    topic: str = Field(default=""),
    scores: list[float] = Field(default_factory=list),
    weaknesses: list[str] = Field(default_factory=list),
    topics: list[str] = Field(default_factory=list),
    turn_count: int = Field(default=0),

class ResumeDeepDive:
    def __init__(self, state: ResumeDeepDiveState):
        self.state = state
        self.llm = LLM()
        self.database = Database()

        resume_deep_dive = StateGraph(ResumeDeepDiveState)
        resume_deep_dive.add_node("generate_questions", self.generate_questions)
        resume_deep_dive.add_node("evaluate_answers", self.evaluate_answers)

        resume_deep_dive.add_edge(START, "generate_questions")
        resume_deep_dive.add_edge("generate_questions", END)
        # resume_deep_dive.add_conditional_edges("evaluate_answers", self.check_score, {
        #     "yes": "generate_questions",
        #     "no": END
        # })

        self.user = self.database.get_document("u111")
        print(self.user)
        state.user_summary = self.user.candidate_summary
        state.projects = self.user.projects
        state.strengths = self.user.strength_signals
        state.topics = self.user.primary_domains + self.user.core_technologies

        self.resume_deep_dive_workflow = resume_deep_dive.compile()

    async def generate_questions(self, state: ResumeDeepDiveState):
        print("Generating question...")
        print(state.turn_count)
        state.turn_count += 1
        generate_questions_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=GENERATE_QUESTIONS_SYSTEM_PROMPT),
            HumanMessage(content="""
                User Summary: {user_summary}
                Projects: {projects}
                Strengths: {strengths}
                topic: {topics}
                difficulty: {difficulty}
            """),
        ])
        messages = generate_questions_prompt_template.format_messages(
            user_summary=state.user_summary,
            projects=state.projects,
            strengths=state.strengths,
            topics=state.topics,
            difficulty=state.difficulty
        )
        response = await self.llm.invoke(messages)
        print(response)
        return {"turn_count": state.turn_count}

    async def evaluate_answers(self, state: ResumeDeepDiveState):
        print("Evaluating answers...")
        state.avg_score = random.randint(0, 100)
        print("Average score: ", state.avg_score)
        return {"avg_score": state.avg_score}

    async def run(self):
        return await self.resume_deep_dive_workflow.ainvoke({"turn_count": 0})

    def check_score(self, state: ResumeDeepDiveState):
        if state.turn_count <= 5 or state.avg_score >= 50:
            return "yes"
        else:
            return "no"

asyncio.run(ResumeDeepDive({}).run())