from crewai import Agent, LLM
from dotenv import load_dotenv
from crewai import Task, Crew

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash"
)


# ============================================================
# Technology Researcher
# ============================================================

technology_researcher = Agent(
    role="Technology Researcher",
    goal="Research the technology and AI capabilities of a startup",
    backstory=(
        "You are a technology analyst who investigates "
        "a startup's technical products, AI capabilities, "
        "technology stack, and technical differentiation."
    ),
    llm=llm
)


# ============================================================
# Competitor Researcher
# ============================================================

competitor_researcher = Agent(
    role="Competitor Researcher",
    goal="Identify and analyze the startup's main competitors",
    backstory=(
        "You are a competitive intelligence analyst who "
        "identifies competitors and compares their products, "
        "features, positioning, and differentiation."
    ),
    llm=llm
)


# ============================================================
# Funding Researcher
# ============================================================

funding_researcher = Agent(
    role="Funding Researcher",
    goal="Research the startup's funding and investors",
    backstory=(
        "You are a financial research analyst who investigates "
        "startup funding rounds, investors, funding amounts, "
        "and investment history."
    ),
    llm=llm
)

# ============================================================
# Technology Research Task
# ============================================================

technology_task = Task(
    description="""
    Research the technology and AI capabilities of OpenAI.

    Focus on:
    - Major AI technologies and products
    - AI models
    - Developer technologies
    - Technical differentiation

    Provide factual and concise findings.
    """,
    expected_output="""
    A concise technology research report covering:
    - Major technologies
    - AI models/products
    - Developer technologies
    - Technical differentiation
    """,
    agent=technology_researcher,
    async_execution=True
)


# ============================================================
# Competitor Research Task
# ============================================================

competitor_task = Task(
    description="""
    Research the major competitors of OpenAI.

    Focus on:
    - Major competitors
    - Their major AI products
    - How they compete with OpenAI
    - Important differences in positioning

    Provide factual and concise findings.
    """,
    expected_output="""
    A concise competitor analysis containing:
    - Major competitors
    - Their relevant products
    - Competitive differences
    - Market positioning
    """,
    agent=competitor_researcher,
    async_execution=True
)


# ============================================================
# Funding Research Task
# ============================================================

funding_task = Task(
    description="""
    Research OpenAI's funding and investors.

    Focus on:
    - Major funding rounds
    - Major investors
    - Important funding events
    - Approximate funding amounts where available

    Provide factual and concise findings.
    """,
    expected_output="""
    A concise funding report containing:
    - Major funding rounds
    - Important investors
    - Funding amounts where available
    - Important funding events
    """,
    agent=funding_researcher,
    async_execution=True
)

# ============================================================
# Parallel Crew
# ============================================================

crew = Crew(
    agents=[
        technology_researcher,
        competitor_researcher,
        funding_researcher
    ],

    tasks=[
        technology_task,
        competitor_task,
        funding_task
    ]
)

result = crew.kickoff()

print("\n================================")
print("       FINAL RESULT")
print("================================")

print(result)