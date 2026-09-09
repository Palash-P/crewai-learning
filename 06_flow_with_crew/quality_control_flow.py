from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Structured Evaluation Output
# ============================================================

class EvaluationResult(BaseModel):
    score: int
    reason: str


# ============================================================
# LLM
# ============================================================

llm = LLM(
    model="gemini/gemini-2.5-flash"
)


# ============================================================
# Research Agent
# ============================================================

researcher = Agent(
    role="AI Researcher",
    goal="Research AI agents and provide accurate information",
    backstory=(
        "You are an AI researcher who explains technical "
        "concepts clearly and accurately."
    ),
    llm=llm
)


# ============================================================
# Research Task
# ============================================================

research_task = Task(
    description="""
    Explain what AI agents are.

    Cover:
    - What an AI agent is
    - How agents differ from normal LLM applications
    - Common components of an AI agent
    """,
    expected_output="""
    A clear and detailed explanation of AI agents,
    including their definition, differences from normal
    LLM applications, and common components.
    """,
    agent=researcher
)


# ============================================================
# Research Crew
# ============================================================

research_crew = Crew(
    agents=[researcher],
    tasks=[research_task]
)


# ============================================================
# Evaluator Agent
# ============================================================

evaluator = Agent(
    role="Research Quality Evaluator",
    goal=(
        "Evaluate whether research is accurate, relevant, "
        "complete, and sufficiently detailed"
    ),
    backstory=(
        "You are a strict research reviewer. "
        "You evaluate research based on accuracy, relevance, "
        "completeness, detail, and clarity."
    ),
    llm=llm
)


# ============================================================
# Evaluation Task
# ============================================================

evaluation_task = Task(
    description="""
    Evaluate the following research:

    {research}

    Determine whether the research is:

    1. Accurate
    2. Relevant
    3. Sufficiently detailed
    4. Clearly written

    Give an overall quality score from 0 to 100.

    Provide a brief explanation for the score.
    """,

    expected_output="""
    A structured evaluation containing:

    - score: integer from 0 to 100
    - reason: brief explanation of the score
    """,

    agent=evaluator,

    # Force the evaluator to return our Pydantic structure
    output_pydantic=EvaluationResult
)


# ============================================================
# Flow State
# ============================================================

class ResearchState(BaseModel):
    research: str = ""
    retry_count: int = 0
    evaluation: EvaluationResult | None = None


# ============================================================
# Research Flow
# ============================================================

class ResearchFlow(Flow[ResearchState]):

    # --------------------------------------------------------
    # Step 1: Run Research
    # --------------------------------------------------------

    @start()
    def run_research(self):

        for attempt in range(1, 4):

            self.state.retry_count = attempt

            print(f"\n🔎 Research attempt {attempt}/3\n")

            research_result = research_crew.kickoff()

            self.state.research = str(research_result)

            print("Research stored in Flow state.")

            # Temporary basic quality check
            if len(self.state.research) >= 500:

                print("\n✅ Research is sufficiently detailed!")

                return self.state.research

            print("\n❌ Research is too short.")

        print("\n🛑 Maximum research attempts reached.")

        return self.state.research


    # --------------------------------------------------------
    # Step 2: Evaluate Research
    # --------------------------------------------------------

    @listen(run_research)
    def evaluate_research(self):

        print("\n🧐 Evaluating research...\n")

        # Pass the current research to the evaluator
        evaluation_task.description = f"""
        Evaluate the following research:

        {self.state.research}

        Determine whether the research is:

        1. Accurate
        2. Relevant
        3. Sufficiently detailed
        4. Clearly written

        Give an overall quality score from 0 to 100.

        Provide a brief explanation for the score.
        """

        result = Crew(
            agents=[evaluator],
            tasks=[evaluation_task]
        ).kickoff()

        # Get structured Pydantic output
        self.state.evaluation = result.pydantic

        print("Evaluation stored in Flow state.")

        print("\n===== EVALUATION =====")
        print(f"Score : {self.state.evaluation.score}")
        print(f"Reason: {self.state.evaluation.reason}")

        return self.state.evaluation


    # --------------------------------------------------------
    # Step 3: Decide What To Do
    # --------------------------------------------------------

    @router(evaluate_research)
    def check_evaluation(self):

        print("\n🚦 Checking evaluation...\n")

        if self.state.evaluation.score >= 80:

            print("✅ Evaluation passed.")

            return "good"

        print("❌ Evaluation failed.")

        return "bad"


    # --------------------------------------------------------
    # Step 4: Good Research Path
    # --------------------------------------------------------

    @listen("good")
    def good_research(self):

        print("\n🎉 Research approved!")

        return (
            f"Research approved with score "
            f"{self.state.evaluation.score}/100."
        )


    # --------------------------------------------------------
    # Step 5: Bad Research Path
    # --------------------------------------------------------

    @listen("bad")
    def bad_research(self):

        print("\n⚠️ Research needs improvement.")

        return (
            f"Research rejected with score "
            f"{self.state.evaluation.score}/100."
        )


# ============================================================
# Start Flow
# ============================================================

flow = ResearchFlow()

result = flow.kickoff()


# ============================================================
# Final Result
# ============================================================

print("\n================================")
print("         FINAL RESULT")
print("================================")

print(result)