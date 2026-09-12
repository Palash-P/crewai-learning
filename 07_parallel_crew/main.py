from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

load_dotenv()

llm = LLM(
    model="openai/gpt-5.6-luna"
)

search_tool = SerperDevTool()


# ============================================================
# Structured Output Schema
# ============================================================

class ResearchResult(BaseModel):
    summary: str
    key_findings: list[str]
    sources: list[str]


# ============================================================
# Technology Researcher
# ============================================================

technology_researcher = Agent(
    role="Technology Researcher",
    goal="Research the current technology and AI capabilities of a startup",
    backstory=(
        "You are a technology analyst who investigates "
        "AI models, developer platforms, products, technical "
        "capabilities, and recent technological developments."
    ),
    tools=[search_tool],
    llm=llm
)


# ============================================================
# Competitor Researcher
# ============================================================

competitor_researcher = Agent(
    role="Competitor Researcher",
    goal="Identify and analyze the major current competitors of a startup",
    backstory=(
        "You are a competitive intelligence analyst who researches "
        "competitors, their products, strengths, weaknesses, "
        "and strategic positioning."
    ),
    tools=[search_tool],
    llm=llm
)


# ============================================================
# Funding Researcher
# ============================================================

funding_researcher = Agent(
    role="Funding Researcher",
    goal="Research the current funding history and investors of a startup",
    backstory=(
        "You are a financial research analyst specializing in "
        "startup funding, investment rounds, investors, valuations, "
        "and strategic financial relationships."
    ),
    tools=[search_tool],
    llm=llm
)


# ============================================================
# Research Tasks
# ============================================================

technology_task = Task(
    description="""
    Research OpenAI's current technology and AI capabilities.

    IMPORTANT:
    - Use the web search tool to gather current information.
    - Do not rely only on your existing knowledge.
    - Prefer official OpenAI sources and other reliable sources.
    - Focus on recent and currently relevant information.
    - Do not invent information.

    Research:
    1. Current AI models and major products
    2. Developer APIs and platforms
    3. Important technical capabilities
    4. Recent technology developments
    5. Technical differentiation

    For every major finding, provide the source URL.
    """,
    expected_output="""
    A structured research result containing:
    - summary
    - key_findings
    - sources
    """,
    agent=technology_researcher,
    async_execution=True,
    output_pydantic=ResearchResult
)


competitor_task = Task(
    description="""
    Research OpenAI's current major competitors.

    IMPORTANT:
    - Use the web search tool.
    - Find current information rather than relying only on existing knowledge.
    - Prefer recent and reliable sources.
    - Do not invent information.

    Identify and analyze major competitors such as:
    - Anthropic
    - Google
    - Meta
    - xAI
    - Microsoft
    - Amazon
    - DeepSeek
    - Mistral

    For each relevant competitor, research:
    1. Major AI products/models
    2. Main competitive advantage
    3. How they compare with OpenAI
    4. Important recent developments

    Provide source URLs for major findings.
    """,
    expected_output="""
    A structured research result containing:
    - summary
    - key_findings
    - sources
    """,
    agent=competitor_researcher,
    async_execution=True,
    output_pydantic=ResearchResult
)


funding_task = Task(
    description="""
    Research OpenAI's current funding and investors.

    IMPORTANT:
    - Use the web search tool to find current information.
    - Do not rely only on your existing knowledge.
    - Prefer official announcements and reliable financial/news sources.
    - Be especially careful with dates, funding amounts,
      investors, and valuations.
    - Distinguish confirmed information from reported estimates.
    - Do not invent information.

    Research:
    1. Major funding rounds
    2. Funding amounts
    3. Important investors
    4. Recent financing
    5. Current valuation information when reliably available
    6. Major strategic investment relationships

    Provide source URLs for major findings.
    """,
    expected_output="""
    A structured research result containing:
    - summary
    - key_findings
    - sources
    """,
    agent=funding_researcher,
    async_execution=True,
    output_pydantic=ResearchResult
)


# ============================================================
# Synthesizer
# ============================================================

synthesizer = Agent(
    role="Research Synthesizer",
    goal="Combine specialist research into an accurate evidence-based startup analysis",
    backstory=(
        "You are a senior business and technology analyst. "
        "You combine research from multiple specialists into "
        "a clear, accurate, evidence-based final report. "
        "You never invent facts and clearly identify uncertainty "
        "or disagreement between sources."
    ),
    llm=llm
)


# ============================================================
# Synthesis Task
# ============================================================

synthesis_task = Task(
    description="""
    Create a comprehensive analysis of OpenAI using ONLY the
    research provided by the Technology, Competitor, and Funding
    research tasks.

    IMPORTANT RULES:

    1. Do NOT rely on your own existing knowledge for factual claims.
    2. Do NOT invent facts, numbers, dates, models, investors,
       products, or events.
    3. Preserve important source URLs from the research.
    4. If research contains conflicting information, explicitly
       mention the conflict instead of choosing an unsupported value.
    5. Clearly distinguish confirmed information from reported
       estimates or uncertain information.
    6. Do not present unsupported claims as facts.

    Organize the final report into:

    ## 1. Technology and AI Capabilities
    ## 2. Major Competitors
    ## 3. Funding and Investors
    ## 4. Overall Analysis

    Include a Sources section under each major section where
    appropriate.
    """,
    expected_output="""
    A clear, evidence-based report containing:

    1. Technology and AI capabilities
    2. Major competitors
    3. Funding and investors
    4. Overall strategic analysis

    Preserve relevant source URLs and clearly identify
    uncertainty or conflicting information.
    """,
    agent=synthesizer,
    context=[
        technology_task,
        competitor_task,
        funding_task
    ]
)


# ============================================================
# Crew
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


# ============================================================
# Execute
# ============================================================

result = crew.kickoff()


print("\n================================")
print("       FINAL RESULT")
print("================================")

print(result)