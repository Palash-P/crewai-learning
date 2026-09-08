from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, start, listen
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash"
)


# -------------------------
# Research Crew
# -------------------------

researcher = Agent(
    role="AI Researcher",
    goal="Research AI agents and provide useful information",
    backstory="You are an AI researcher who explains technical concepts clearly.",
    llm=llm
)

research_task = Task(
    description="""
    Explain what AI agents are.

    Cover:
    - What an AI agent is
    - How agents differ from normal LLM applications
    - Common components of an AI agent
    """,
    expected_output="A clear explanation of AI agents.",
    agent=researcher
)

research_crew = Crew(
    agents=[researcher],
    tasks=[research_task]
)


# -------------------------
# Writing Crew
# -------------------------

writer = Agent(
    role="Technical Writer",
    goal="Turn research into a beginner-friendly article",
    backstory="You are an excellent technical writer.",
    llm=llm
)

writing_task = Task(
    description="""
    Write a beginner-friendly article about AI agents.

    Use the research provided to you.
    """,
    expected_output="A well-structured beginner-friendly article.",
    agent=writer
)

writing_crew = Crew(
    agents=[writer],
    tasks=[writing_task]
)


# -------------------------
# Flow
# -------------------------

class ArticleFlow(Flow):

    @start()
    def research(self):
        print("\n🔎 Running Research Crew...\n")

        result = research_crew.kickoff()

        return result

    @listen(research)
    def write(self, research_result):
        print("\n✍️ Running Writing Crew...\n")

        # We will connect the research to the writing task
        writing_task.description = f"""
        Write a beginner-friendly article about AI agents.

        Use the following research:

        {research_result}

        Turn this research into a clear article.
        """

        result = writing_crew.kickoff()

        return result


flow = ArticleFlow()

result = flow.kickoff()

print("\n===== FINAL RESULT =====\n")
print(result)