from crewai import Agent, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from crewai import Task, Crew

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash"
)
search_tool = SerperDevTool()


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
    tools=[search_tool],
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
    tools=[search_tool],
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
    tools=[search_tool],
    llm=llm
)

# ============================================================
# Synthesizer
# ============================================================

synthesizer = Agent(
    role="Research Synthesizer",
    goal="Combine research findings into one clear startup analysis",
    backstory=(
        "You are a senior business and technology analyst. "
        "You combine findings from multiple researchers into "
        "a coherent, accurate, and useful final report."
    ),
    llm=llm
)

# ============================================================
# Technology Research Task
# ============================================================

technology_task = Task(
    description="""
    Research OpenAI's current technology and AI capabilities
    using web search.

    Search for reliable and recent information.

    Focus on:
    - Current AI models and products
    - Developer technologies
    - Current technical capabilities
    - Major technical developments
    - Technical differentiation

    Prefer official OpenAI sources and other highly reliable
    sources.

    Do not rely only on your existing knowledge.
    """,
    expected_output="""
    A concise technology research report containing:
    - Current technologies and products
    - AI models
    - Developer technologies
    - Recent developments
    - Technical differentiation
    - Sources used
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
# Synthesis Task
# ============================================================

synthesis_task = Task(
    description="""
    Combine the findings from the technology, competitor,
    and funding researchers into one final report about OpenAI.

    Organize the report into:

    1. Technology and AI capabilities
    2. Major competitors
    3. Funding and investors
    4. Overall analysis

    Do not invent information.
    Use the research provided by the other tasks.
    """,

    expected_output="""
    A clear and well-structured final report containing:

    - Technology analysis
    - Competitor analysis
    - Funding analysis
    - Overall conclusion
    """,

    agent=synthesizer,

    # This task waits for the asynchronous research tasks
    context=[
        technology_task,
        competitor_task,
        funding_task
    ]
)


# ============================================================
# Parallel Crew
# ============================================================

crew = Crew(
    agents=[
        technology_researcher,
        competitor_researcher,
        funding_researcher,
        synthesizer
    ],

    tasks=[
        technology_task,
        competitor_task,
        funding_task,
        synthesis_task
    ]
)

result = crew.kickoff()

print("\n================================")
print("       FINAL RESULT")
print("================================")

print(result)