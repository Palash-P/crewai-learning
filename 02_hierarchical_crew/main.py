from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash"
)

researcher = Agent(
    role="Researcher",
    goal="Research the assigned topic and provide accurate information",
    backstory="You are an experienced researcher who gathers and organizes useful information.",
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Create a clear and well-structured article from the research",
    backstory="You are an experienced technical writer.",
    llm=llm
)

manager = Agent(
    role="Project Manager",
    goal="Coordinate the team and delegate tasks to the appropriate agents to successfully complete the project",
    backstory="You are an experienced project manager who understands how to coordinate teams, delegate work, and ensure that projects are completed successfully.",
    llm=llm
)

task = Task(
    description="""
    Create a beginner-friendly article explaining AI agents.

    The article should explain:
    1. What AI agents are
    2. How AI agents differ from normal LLM applications
    3. The main components of an AI agent
    4. Practical applications of AI agents
    """,
    expected_output="""
    A clear, well-structured beginner-friendly article about AI agents.
    """
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    process=Process.hierarchical,
    manager_agent=manager
)

result = crew.kickoff()

print(result)