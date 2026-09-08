from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash"
)

search_tool = SerperDevTool()

researcher = Agent(
    role="Research Assistant",
    goal="Find accurate and relevant information from the web",
    backstory=(
        "You are a research assistant who searches the web "
        "to find reliable information and summarizes it clearly."
    ),
    tools=[search_tool],
    llm=llm
)

task = Task(
    description="""
    Search the web and find the current population of India.

    Explain the approximate population and mention the source or
    sources used to obtain the information.
    """,
    expected_output="""
    A concise answer containing:
    1. India's approximate current population
    2. The source of the information
    3. A brief explanation
    """,
    agent=researcher
)

crew = Crew(
    agents=[researcher],
    tasks=[task]
)

result = crew.kickoff()

print("\n===== FINAL RESULT =====\n")
print(result)