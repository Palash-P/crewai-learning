from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Configuration
# ============================================================

MAX_RETRIES = 3
QUALITY_THRESHOLD = 80


# ============================================================
# Structured Evaluation Output
# ============================================================

class EvaluationResult(BaseModel):
    score: int
    reason: str


# ============================================================
# Flow State
# ============================================================

class ResearchState(BaseModel):
    research: str = ""
    retry_count: int = 0
    evaluation: EvaluationResult | None = None


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
    description="",
    expected_output="""
    A clear and detailed explanation of AI agents.
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
        "complete, and sufficiently detailed."
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
    description="",
    expected_output="""
    Return a structured evaluation containing:

    - score: integer from 0 to 100
    - reason: brief explanation of the score
    """,
    agent=evaluator,
    output_pydantic=EvaluationResult
)


# ============================================================
# Research Flow
# ============================================================

class ResearchFlow(Flow[ResearchState]):

    # --------------------------------------------------------
    # Research + Retry Loop
    # --------------------------------------------------------

    @start()
    def run_research(self):

        for attempt in range(1, MAX_RETRIES + 1):

            self.state.retry_count = attempt

            print(
                f"\n🔎 Research attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            # First attempt is intentionally weak.
            if attempt == 1:

                research_task.description = """
                Give a VERY short explanation of AI agents.

                Only provide 2-3 sentences.
                Do not explain the components in detail.
                """

            else:

                research_task.description = f"""
                Improve the previous research using the
                evaluator's feedback.

                Previous research:
                {self.state.research}

                Evaluator score:
                {self.state.evaluation.score}

                Evaluator feedback:
                {self.state.evaluation.reason}

                Now provide a much better explanation of AI agents.

                Cover:
                - What an AI agent is
                - How agents differ from normal LLM applications
                - Common components of an AI agent
                """

            # ------------------------------------------------
            # Run Research
            # ------------------------------------------------

            research_result = research_crew.kickoff()

            self.state.research = str(research_result)

            print("Research completed.")


            # ------------------------------------------------
            # Evaluate Research
            # ------------------------------------------------

            print("\n🧐 Evaluating research...")

            evaluation_task.description = f"""
            Evaluate the following research:

            --- RESEARCH ---
            {self.state.research}
            --- END RESEARCH ---

            Determine whether the research is:

            1. Accurate
            2. Relevant
            3. Sufficiently detailed
            4. Clearly written

            Give an overall quality score from 0 to 100.

            Explain why you gave that score.
            """

            evaluation_result = Crew(
                agents=[evaluator],
                tasks=[evaluation_task]
            ).kickoff()

            self.state.evaluation = evaluation_result.pydantic

            print("\n===== EVALUATION =====")
            print(
                f"Score : {self.state.evaluation.score}"
            )
            print(
                f"Reason: {self.state.evaluation.reason}"
            )


            # ------------------------------------------------
            # Decision
            # ------------------------------------------------

            if self.state.evaluation.score >= QUALITY_THRESHOLD:

                print("\n✅ Research approved!")

                return self.state.research


            # ------------------------------------------------
            # Retry
            # ------------------------------------------------

            if attempt < MAX_RETRIES:

                print(
                    "\n🔄 Research quality is too low."
                )

                print(
                    f"Retrying... "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )

                continue


            # ------------------------------------------------
            # Maximum Attempts Reached
            # ------------------------------------------------

            print(
                "\n🛑 Maximum retries reached."
            )

            return self.state.research


# ============================================================
# Run Flow
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