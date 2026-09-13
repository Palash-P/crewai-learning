from crewai import Agent, Crew, Task, LLM
from crewai.knowledge.source.text_file_knowledge_source import (
    TextFileKnowledgeSource
)
from dotenv import load_dotenv


load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = LLM(
    model="openai/gpt-5.6-luna"
)


# ============================================================
# Knowledge Sources
# ============================================================

company_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "company.txt",
        "product.txt",
        "policies.txt"
    ]
)


# ============================================================
# Agent
# ============================================================

assistant = Agent(
    role="Company Knowledge Assistant",
    goal=(
        "Answer questions accurately using the company's "
        "provided knowledge"
    ),
    backstory=(
        "You are an internal company assistant. "
        "You answer questions using the company's knowledge base. "
        "If the information is not available, clearly say "
        "that it is not available."
    ),
    llm=llm,
    verbose=True
)


# ============================================================
# Task
# ============================================================

task = Task(
    description="""
    Answer the following question using the available company knowledge:

    What is NovaFlow and what capabilities does it provide?

    Do not invent information.
    Use only information available in the company knowledge.
    """,
    expected_output="""
    A concise answer explaining:
    - What NovaFlow is
    - Its main capabilities
    - The technologies/integrations it supports
    """,
    agent=assistant
)


# ============================================================
# Crew
# ============================================================

crew = Crew(
    agents=[assistant],
    tasks=[task],
    verbose=True,
    knowledge_sources=[company_knowledge]
)


# ============================================================
# Run
# ============================================================

result = crew.kickoff()

print("\n================================")
print("       FINAL RESULT")
print("================================")
print(result)