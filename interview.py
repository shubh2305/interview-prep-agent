from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import asyncio
from stages.intro import Intro

class InterviewState(BaseModel):
    score: int = Field(default=0),
    strengths: list[str] = Field(default_factory=list),
    weaknesses: list[str] = Field(default_factory=list),
    resume_summary: str = Field(default="")

class Interview:
    def __init__(self, state: InterviewState):
        interview_graph = StateGraph(InterviewState)

        interview_graph.add_node("intro", self.intro)
        interview_graph.add_node("resume_deep_dive", self.resume_deep_dive)
        interview_graph.add_node("behavioral_questions", self.behavioral_questions)
        interview_graph.add_node("conclusion", self.conclusion)

        interview_graph.add_edge(START, "intro")
        interview_graph.add_edge("intro", "resume_deep_dive")
        interview_graph.add_edge("resume_deep_dive", "behavioral_questions")
        interview_graph.add_edge("behavioral_questions", "conclusion")
        interview_graph.add_edge("conclusion", END)

        self.interview_workflow = interview_graph.compile()
        

    async def intro(self, state: InterviewState):
        print("Starting the intro stage...")
        # intro_stage = Intro(self.state)
        # intro_response = await intro_stage.intro()
        return self.state

    async def resume_deep_dive(self, state: InterviewState):
        print("Starting the resume deep dive stage...")
        # resume_deep_dive = ResumeDeepDive(self.state)
        # resume_deep_dive_response = await resume_deep_dive.resume_deep_dive()
        return self.state

    async def behavioral_questions(self, state: InterviewState):
        print("Starting the behavioral questions stage...")
        # behavioral_questions = BehavioralQuestions(self.state)
        # behavioral_questions_response = await behavioral_questions.behavioral_questions()
        return self.state

    async def conclusion(self, state: InterviewState):
        print("Starting the conclusion stage...")
        # conclusion = Conclusion(self.state)
        # conclusion_response = await conclusion.conclusion()
        return self.state

    async def run(self):
        return await self.interview_workflow.ainvoke()

interview = Interview(state={"score": 100, "strengths": ["Python", "SQL", "Data Analysis"], "weaknesses": ["Java", "C++", "JavaScript"], "resume_summary": "Shubham is a software engineer with 5 years of experience in Python and SQL. He has a Bachelor's degree in Computer Science from the University of California, Berkeley."})
asyncio.run(interview.run())