from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash"
)

researcher = Agent(
    role="AI Researcher",
    goal="Research and explain AI agents clearly",
    backstory="You are an experienced AI researcher.",
    llm=llm
)

research_task = Task(
    description="""
    Explain what AI agents are.

    Cover:
    1. What an AI agent is
    2. How it differs from a normal LLM call
    3. Components of an AI agent
    4. A real-world example
    """,
    expected_output="""
    A beginner-friendly explanation covering all four points.
    """,
    agent=researcher
)

analyst = Agent(
    role="AI Analyst",
    goal="Analyze research about AI agents and identify the most important insights",
    backstory="You are an AI analyst who turns research into useful insights.",
    llm=llm
)

analysis_task = Task(
    description="""
    Analyze the research produced by the researcher.

    Identify:
    1. The most important concepts
    2. The key differences between AI agents and normal LLM applications
    3. Important practical applications
    4. The main conclusions from the research
    """,
    expected_output="""
    A structured analysis containing the key insights,
    practical applications, and conclusions.
    """,
    agent=analyst,
    context=[research_task]
)

writer = Agent(
    role="Technical Writer",
    goal="Turn research and analysis into a clear, professional report",
    backstory="You are an experienced technical writer who explains complex AI concepts simply.",
    llm=llm
)

writing_task = Task(
    description="""
    Write a professional markdown report about AI agents.

    Use the research and analysis provided to you.

    The report should include:
    1. Introduction
    2. What AI agents are
    3. AI agents vs normal LLM applications
    4. Components of AI agents
    5. Practical applications
    6. Conclusion

    Keep the explanation clear and beginner-friendly.
    """,
    expected_output="""
    A polished markdown report about AI agents.
    """,
    agent=writer,
    context=[research_task, analysis_task]
)

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential
)

result = crew.kickoff()

print(result)