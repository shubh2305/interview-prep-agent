import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

BEHAVIORAL_QUESTIONS_SYSTEM_PROMPT = """
    You are an experienced technical interviewer.

    Your task is to ask ONE behavioral interview question to the candidate.

    Context:
    - Role: Backend Engineer
    - Candidate experience: {experience_level}
    - Covered behavioral intents: {covered_intents}

    Behavioral intents to choose from:
    - challenge_story
    - failure_story
    - ownership_example
    - conflict_resolution
    - decision_making
    - time_management
    - learning_from_mistake

    Instructions:
    1. Ask only ONE question.
    2. Do NOT repeat or rephrase any already covered intents.
    3. Prefer asking about weak areas if provided.
    4. Question should encourage a STAR-format answer (Situation, Task, Action, Result).
    5. Keep the question clear, concise, and realistic.
    6. Avoid generic phrasing — make it slightly specific to backend/engineering context when possible.

    Output format (JSON only):
    {
        "question": "<behavioral question>",
        "intent": "<selected_intent>"
    }
"""

EVALUATE_ANSWERS_SYSTEM_PROMPT = """
    You are an experienced technical interviewer.

    Evaluate the candidate’s answer to a behavioral interview question.

    Your evaluation must be strict, objective, and based ONLY on the provided answer.

    --------------------------------------------------
    Evaluation Criteria (Score each from 0 to 2):

    1. Clarity
    - Is the answer easy to understand and well-structured?
    - Avoids rambling and confusion

    2. Ownership
    - Does the candidate clearly describe their own actions ("I did")?
    - Avoids overusing "we" without personal contribution

    3. Problem Solving
    - Did the candidate identify and address the problem effectively?
    - Shows logical thinking and decision-making

    4. Impact
    - Did the actions lead to a meaningful outcome?
    - Includes measurable or clearly described results

    5. Reflection
    - Does the candidate demonstrate learning or improvement?
    - Mentions what they would do differently or lessons learned

    --------------------------------------------------
    STAR Framework Check (Do NOT score separately, but consider in evaluation):
    - Situation: Context is clear
    - Task: Responsibility is defined
    - Action: Candidate’s actions are explained
    - Result: Outcome is described

    --------------------------------------------------
    Scoring Rules:
    - 0 = Poor / Missing
    - 1 = Partial / Average
    - 2 = Strong / Clear
    - Be strict and consistent

    --------------------------------------------------
    Output JSON ONLY:

    {
        "feedback": "2-3 concise sentences summarizing performance",
        "score": 0-10
    }

    --------------------------------------------------
    Important Instructions:
    - Do NOT assume missing details
    - Penalize vague or generic answers
    - Reward specific examples and clear ownership
    - Keep feedback actionable and concise
"""

class BehavioralQuestionsState(BaseModel):
    total_questions: int = Field(default=0)
    experience_level: str = Field(default="")
    scores: list[dict] = Field(default_factory=list)
    avg_score: float = Field(default=0.0)
    question: str = Field(default="")
    answer: str = Field(default="")
    feedback: list[dict] = Field(default_factory=list)
    improvement_suggestions: list[dict] = Field(default_factory=list)

class BehavioralQuestions:
    def __init__(self, total_questions: int):
        self.total_questions = total_questions
        self.llm = LLM()
        self.database = Database()

        self.behavioral_stage = StateGraph(BehavioralQuestionsState)
        self.behavioral_stage.add_node("generate_questions", self.generate_questions)
        self.behavioral_stage.add_node("get_answer", self.get_answer)
        self.behavioral_stage.add_node("evaluate_answers", self.evaluate_answers)
        self.behavioral_stage.add_node("update_state", self.update_state)

        self.behavioral_stage.add_edge(START, "generate_questions")
        self.behavioral_stage.add_edge("generate_questions", "get_answer")
        self.behavioral_stage.add_edge("get_answer", "evaluate_answers")
        self.behavioral_stage.add_edge("evaluate_answers", "update_state")
        self.behavioral_stage.add_conditional_edges("update_state", self.check_score, {
            "yes": "generate_questions",
            "no": END
        })

        self.behavioral_stage_workflow = self.behavioral_stage.compile()    

    async def run(self, experience_level: str):
        state = {
            "experience_level": experience_level,
        }
        return await self.behavioral_stage_workflow.ainvoke(state)

    async def generate_questions(self, state: BehavioralQuestionsState):
        print("Generating questions...")
        state.total_questions += 1
        messages = [
            SystemMessage(content=BEHAVIORAL_QUESTIONS_SYSTEM_PROMPT),
            HumanMessage(content=f"""
                Experience level: {state.experience_level}
                Covered intents: {state.covered_intents}
            """),
        ]
        question = await self.llm.invoke(messages)
        return {"total_questions": state.total_questions, "question": question["question"], "intent": question["intent"]}

    def get_answer(self, state: BehavioralQuestionsState):
        print("Getting answer...")
        answer = input("Enter your answer: ")
        return {"answer": answer}

    async def evaluate_answers(self, state: BehavioralQuestionsState):
        print("Evaluating answers...")
        messages = [
            SystemMessage(content=EVALUATE_ANSWERS_SYSTEM_PROMPT),
            HumanMessage(content=f"""
                Question: {state.question}
                Answer: {state.answer}
            """),
        ]
        evaluation = await self.llm.invoke(messages)
        return {"feedback": state.feedback + [evaluation["feedback"]], "scores": state.scores + [evaluation["score"]]}

    def update_state(self, state: BehavioralQuestionsState):
        print("Updating state...")
        state.avg_score = sum(state.scores) / state.total_questions
        return {"avg_score": state.avg_score}

    def check_score(self, state: BehavioralQuestionsState):
        print("Checking score...")
        if state.total_questions >= self.total_questions:
            return "no"
        else:
            return "yes"