import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt, Command
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
    question: str = Field(default=""),
    answer: str = Field(default=""),
    evaluation: dict = Field(default={}),

class ResumeDeepDive:
    def __init__(self):
        self.llm = LLM()
        self.database = Database()

        resume_deep_dive = StateGraph(ResumeDeepDiveState)
        resume_deep_dive.add_node("generate_questions", self.generate_questions)
        resume_deep_dive.add_node("get_answer", self.get_answer)
        resume_deep_dive.add_node("evaluate_answers", self.evaluate_answers)
        resume_deep_dive.add_node("update_state", self.update_state)

        resume_deep_dive.add_edge(START, "generate_questions")
        resume_deep_dive.add_edge("generate_questions", "get_answer")
        resume_deep_dive.add_edge("get_answer", "evaluate_answers")
        resume_deep_dive.add_edge("evaluate_answers", "update_state")
        resume_deep_dive.add_conditional_edges("update_state", self.check_score, {
            "yes": "generate_questions",
            "no": END
        })

        self.resume_deep_dive_workflow = resume_deep_dive.compile()

    async def generate_questions(self, state: ResumeDeepDiveState):
        print("Generating question...")
        print(state.turn_count)
        state.turn_count += 1
        state.topic = state.topics[random.randint(0, len(state.topics) - 1)]
        state.difficulty = random.randint(0, 10)
        messages = [
            SystemMessage(content=GENERATE_QUESTIONS_SYSTEM_PROMPT),
            HumanMessage(content=f"""
                User Summary: {state.user_summary}
                Projects: {state.projects}
                Strengths: {state.strengths}
                topic: {state.topic}
                difficulty: {state.difficulty}
            """),
        ]
        question = await self.llm.invoke(messages)
        print("Question: ", question["question"])
        return {"turn_count": state.turn_count, "question": question["question"]}

    def get_answer(self, state: ResumeDeepDiveState):
        print("Getting answer...")
        answer = input("Enter your answer: ")
        return {"answer": answer}

    async def evaluate_answers(self, state: ResumeDeepDiveState):
        print("Evaluating answers...")
        messages = [
            SystemMessage(content=EVALUATE_ANSWERS_SYSTEM_PROMPT),
            HumanMessage(content=f"""
                Question: {state.question}
                Answer: {state.answer}
            """),
        ]
        evaluation = await self.llm.invoke(messages)
        print("Evaluation: ", evaluation)

        return {"evaluation": evaluation, "scores": state.scores + [evaluation["score"]]}

    async def run(self, state: ResumeDeepDiveState):
        state["turn_count"] = 0
        return await self.resume_deep_dive_workflow.ainvoke(state)

    def check_score(self, state: ResumeDeepDiveState):
        if state.turn_count > 15:
            return "no"
        if state.turn_count <= 5 or state.avg_score >= 50:
            return "yes"
        else:
            return "no"

    def update_state(self, state: ResumeDeepDiveState):
        if state.evaluation['score'] < 60:
            state.difficulty = 1
            state.topics = state.user_summary['primary_domains'] + state.user_summary['core_technologies']
        elif state.evaluation['score'] < 80:
            state.difficulty = max(state.difficulty + 1, 10)
            state.topics = state.evaluation.get('follow_up_topics', [])
        else:
            state.difficulty = max(state.difficulty + 2, 10)
            state.topics = state.evaluation.get('follow_up_topics', [])
        
        state.strengths = list(set(state.evaluation.get('strengths', []) + state.strengths))
        state.weaknesses = list(set(state.evaluation.get('weaknesses', []) + state.weaknesses))

        if state.turn_count >= 5:
            state.avg_score = sum(state.scores) / state.turn_count

    async def get_user(self):
        state = {}
        self.user = await self.database.get_document("u111")
        state["user_summary"] = self.user["candidate_summary"]
        state["projects"] = self.user["notable_projects"]
        state["topics"] = self.user["primary_domains"] + self.user["core_technologies"]
        return state

async def main():
    resume_deep_dive = ResumeDeepDive()
    state =await resume_deep_dive.get_user()
    await resume_deep_dive.run(state)

asyncio.run(main())