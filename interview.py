from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import asyncio
from stages.intro import Intro

class InterviewState(BaseModel):
    intro_result: dict = Field(default={}),
    resume_deep_dive_result: dict = Field(default={}),
    behavioral_questions_result: dict = Field(default={}),
    resume_summary: str = Field(default=""),

class Interview:
    def __init__(self, user_id: str):
        self.llm = LLM()
        self.database = Database()
        interview_graph = StateGraph(InterviewState)
        self.user = await self.database.get_document(user_id)
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
        intro_stage = Intro({"resume_summary": self.state["resume_summary"]})
        intro_question = await intro_stage.intro()
        print("Welcome message: ", intro_question["welcome_message"])
        print("Question: ", intro_question["question"])
        answer = input("Enter your answer: ")
        evaluation = await intro_stage.evaluate_answer(answer)

        return {"intro_result": evaluation}

    async def resume_deep_dive(self, state: InterviewState):
        print("Starting the resume deep dive stage...")
        resume_deep_dive = ResumeDeepDive()
        resume_deep_dive_state = {
            "user_summary": self.user["candidate_summary"],
            "projects": self.user["notable_projects"],
            "topics": self.user["primary_domains"] + self.user["core_technologies"],
        }
        resume_deep_dive_response = await resume_deep_dive.run(resume_deep_dive_state)
        resume_deep_dive_result = {
            "avg_score": resume_deep_dive_response["avg_score"],
            "strengths": resume_deep_dive_response["strengths"],
            "weaknesses": resume_deep_dive_response["weaknesses"],
        }
        return {"resume_deep_dive_result": resume_deep_dive_result}

    async def behavioral_questions(self, state: InterviewState):
        print("Starting the behavioral questions stage...")
        total_questions = random.randint(3, 5)

        behavioral_questions = BehavioralQuestions(total_questions)
        behavioral_questions_response = await behavioral_questions.run(state["behavioral_questions_result"])
        behavioral_questions_result = {
            "avg_score": behavioral_questions_response["avg_score"],
            "feedback": behavioral_questions_response["feedback"],
        }
        return {"behavioral_questions_result": behavioral_questions_result}

    async def conclusion(self, state: InterviewState):
        print("Starting the conclusion stage...")
        prompt = """
            You are an experienced technical interviewer.

            Your task is to conclude the interview.

            Output format (JSON only):
            {
                "conclusion": "2-3 concise sentences concluding the interview"
            }
        """

        response = await self.llm.invoke(prompt)
        print("Conclusion: ", response["conclusion"])
        return state

    async def run(self):
        state = InterviewState()
        return await self.interview_workflow.ainvoke(state)