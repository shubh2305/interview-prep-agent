from langgraph.graph import StateGraph, START, END

from process_resume import ResumeProcessor

async def process_resume(state: ResumeProcessorState):
    resume_processor = ResumeProcessor()
    await resume_processor.process_resume(state)

async def start_interview(state: ResumeProcessorState):
    interview = Interview("123")
    await interview.run()

class ResumeProcessorState(BaseModel):
    resume_data: dict = Field(default_factory=dict)

graph = StateGraph(ResumeProcessorState)

graph.add_node("process_resume", process_resume)

graph.add_edge(START, "process_resume")
graph.add_edge("process_resume", "start_interview")
graph.add_edge("start_interview", END)

workflow = graph.compile()

workflow.ainvoke({"resume_data": {}})


def main():
    print("Hello from interview-prep!")


if __name__ == "__main__":
    main()
