from crewai import Agent, Crew, Task, LLM
from crewai.flow.flow import Flow, start, listen
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

llm = LLM(model="gemini/gemini-2.5-flash")


# ============================================================
# STATE
# ============================================================

class ContentState(BaseModel):
    topic: str = ""
    research: str = ""
    content: str = ""
    review_score: int = 0
    review_feedback: str = ""


# ============================================================
# AGENTS
# ============================================================

researcher = Agent(
    role="Research Specialist",
    goal="Research the given topic and provide accurate, useful information",
    backstory=(
        "You are an experienced research specialist. "
        "You gather relevant facts and organize them clearly. "
        "You avoid inventing information."
    ),
    llm=llm,
)

writer = Agent(
    role="Content Writer",
    goal="Create clear and engaging content using the provided research",
    backstory=(
        "You are an experienced technical content writer. "
        "You transform research into clear, useful, and well-structured content."
    ),
    llm=llm,
)

reviewer = Agent(
    role="Content Reviewer",
    goal="Evaluate content for accuracy, relevance, clarity, and completeness",
    backstory=(
        "You are a strict content reviewer. "
        "You identify weaknesses and provide actionable feedback."
    ),
    llm=llm,
)


# ============================================================
# STRUCTURED REVIEW OUTPUT
# ============================================================

class ReviewResult(BaseModel):
    score: int
    feedback: str


# ============================================================
# FLOW
# ============================================================

class ContentApprovalFlow(Flow[ContentState]):

    @start()
    def research_topic(self):

        self.state.topic = "How AI agents are changing software development"

        research_task = Task(
            description=f"""
            Research this topic:

            {self.state.topic}

            Provide accurate and useful information that can
            be used by a writer to create an article.
            """,
            expected_output=(
                "A structured research summary containing "
                "important facts, concepts, and examples."
            ),
            agent=researcher,
        )

        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            verbose=True,
        )

        result = crew.kickoff()

        self.state.research = result.raw

        return self.state.research


    @listen(research_topic)
    def write_content(self, research):

        writing_task = Task(
            description=f"""
            Write an article about:

            {self.state.topic}

            Use the following research:

            {research}

            Create clear, professional content.
            Do not invent facts that are not supported
            by the research.
            """,
            expected_output=(
                "A clear, professional article of approximately "
                "500 words."
            ),
            agent=writer,
        )

        crew = Crew(
            agents=[writer],
            tasks=[writing_task],
            verbose=True,
        )

        result = crew.kickoff()

        self.state.content = result.raw

        return self.state.content


    @listen(write_content)
    def review_content(self, content):

        review_task = Task(
            description=f"""
            Review the following article:

            {content}

            Evaluate it on:

            1. Accuracy
            2. Relevance
            3. Clarity
            4. Completeness

            Give a score from 0 to 100.

            Score 80 or higher means the content is acceptable.
            """,
            expected_output="""
            Return a structured review containing:

            - score: integer from 0 to 100
            - feedback: concise explanation of the score
            """,
            agent=reviewer,
            output_pydantic=ReviewResult,
        )

        crew = Crew(
            agents=[reviewer],
            tasks=[review_task],
            verbose=True,
        )

        result = crew.kickoff()

        review = result.pydantic

        self.state.review_score = review.score
        self.state.review_feedback = review.feedback

        return review


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    flow = ContentApprovalFlow()

    result = flow.kickoff()

    print("\n================================")
    print("       FINAL RESULT")
    print("================================")

    print(f"\nTopic: {flow.state.topic}")
    print(f"\nReview Score: {flow.state.review_score}")
    print(f"\nReview Feedback: {flow.state.review_feedback}")

    print("\nFinal Content:")
    print(flow.state.content)