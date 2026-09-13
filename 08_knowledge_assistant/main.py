from crewai import Agent, Crew, Task, LLM
from crewai.knowledge.source.text_file_knowledge_source import (
    TextFileKnowledgeSource
)
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# LLM
# -----------------------------

llm = LLM(
    model="openai/gpt-5.6-luna"
)

# -----------------------------
# Knowledge Base
# -----------------------------

company_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "company.txt",
        "product.txt",
        "policies.txt"
    ]
)

# -----------------------------
# Agent
# -----------------------------

assistant = Agent(
    role="Company Knowledge Assistant",
    goal=(
        "Answer questions accurately using company knowledge "
        "and remember useful information from the conversation"
    ),
    backstory=(
        "You are an internal company assistant. "
        "Use the company knowledge base for company facts "
        "and use conversation memory when the user provides "
        "personal context or information."
    ),
    llm=llm,
    verbose=True
)

# -----------------------------
# Conversation
# -----------------------------

questions = [
    "Remember that our company has 20 employees.",
    "What is NovaFlow?",
    "Would NovaFlow be suitable for our company?"
]

# -----------------------------
# Tasks
# -----------------------------

tasks = []

for question in questions:

    task = Task(
        description=f"""
        Respond to the following user message:

        {question}

        Rules:
        - Use company knowledge when the question requires company facts.
        - Use previous conversation context when relevant.
        - Remember useful information provided by the user.
        - Do not invent information.
        - If information is unavailable, clearly say so.
        """,
        expected_output="""
        A concise and accurate response that uses
        relevant knowledge and conversation context.
        """,
        agent=assistant
    )

    tasks.append(task)

# -----------------------------
# Crew
# -----------------------------

crew = Crew(
    agents=[assistant],
    tasks=tasks,
    knowledge_sources=[company_knowledge],
    memory=True,
    verbose=True
)

# -----------------------------
# Execute
# -----------------------------

result = crew.kickoff()

print("\n================================")
print("       FINAL RESULT")
print("================================")
print(result)