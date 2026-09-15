# CrewAI Knowledge Assistant

A practical CrewAI project demonstrating **Knowledge Retrieval, RAG-style grounding, Agent Memory, and persistent memory across separate executions**.

## Overview

This project builds a company knowledge assistant for a fictional company called **NovaTech AI**.

The assistant can:

* Retrieve information from company knowledge files
* Answer questions using retrieved knowledge
* Remember information provided by the user
* Retrieve previously stored memories in later executions
* Combine external knowledge with conversational/user context

## Architecture

```text
                    ┌─────────────────────┐
                    │   User Question     │
                    └──────────┬──────────┘
                               │
                  ┌────────────▼────────────┐
                  │      CrewAI Agent       │
                  └────────────┬────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
        ┌────────▼────────┐       ┌─────────▼─────────┐
        │    Knowledge    │       │      Memory       │
        │    Retrieval    │       │    Retrieval      │
        └────────┬────────┘       └─────────┬─────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               │
                        ┌──────▼──────┐
                        │     LLM     │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │    Answer   │
                        └─────────────┘
```

## Project Structure

```text
08_knowledge_assistant/
│
├── main.py
│
└── knowledge/
    ├── company.txt
    ├── product.txt
    └── policies.txt
```

## Knowledge Base

The assistant uses three text files:

### company.txt

Contains information about:

* NovaTech AI
* Company history
* Teams
* Main product

### product.txt

Contains information about:

* NovaFlow
* Workflow automation
* AI document processing
* Email classification
* Data extraction
* API integrations
* Scheduled workflows
* Python
* REST APIs
* Webhooks

### policies.txt

Contains:

* Refund policy
* Support policy
* Data retention
* Security requirements

## Key CrewAI Concepts

### Agent

The agent acts as the company knowledge assistant.

```python
assistant = Agent(
    role="Company Knowledge Assistant",
    goal="Answer questions accurately using company knowledge",
    backstory="You are an internal company assistant.",
    llm=llm
)
```

### Knowledge

CrewAI Knowledge provides the agent with external information that can be retrieved during execution.

```python
company_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "company.txt",
        "product.txt",
        "policies.txt"
    ]
)
```

### Memory

Memory allows the agent to store and retrieve relevant information from previous interactions.

```python
crew = Crew(
    agents=[assistant],
    tasks=tasks,
    knowledge_sources=[company_knowledge],
    memory=True
)
```

## Knowledge vs Memory

| Feature  | Knowledge                      | Memory                           |
| -------- | ------------------------------ | -------------------------------- |
| Purpose  | External/reference information | Previous interaction information |
| Example  | NovaFlow supports Python       | User's company has 20 employees  |
| Source   | Knowledge files                | Previous interactions            |
| Main use | Grounding answers              | Maintaining context              |

### Simple mental model

```text
Knowledge → "What information can I retrieve?"

Memory → "What do I remember from previous interactions?"
```

## Persistent Memory Experiment

The project demonstrates memory across separate program executions.

### First execution

The user provides:

```text
Remember that our company has 20 employees.
```

CrewAI stores this information in memory.

### Second execution

The user asks:

```text
How many employees does our company have?
```

The agent uses its `search_memory` capability and retrieves:

```text
NovaTech AI has 20 employees.
```

This demonstrates that memory can persist beyond a single Python process.

## Example Knowledge Question

```text
What is NovaFlow?
```

Example answer:

```text
NovaFlow is an AI-powered workflow automation platform
from NovaTech AI designed primarily for small and
medium-sized businesses.
```

## Example Memory Question

First execution:

```text
Remember that our company has 20 employees.
```

Later execution:

```text
How many employees does our company have?
```

Expected result:

```text
NovaTech AI has 20 employees.
```

## Technologies

* Python
* CrewAI
* Google Gemini
* CrewAI Knowledge
* CrewAI Memory
* Pydantic
* python-dotenv

## What I Learned

This project helped me understand:

1. CrewAI Knowledge Sources
2. Text-based knowledge retrieval
3. RAG-style grounding
4. Agent memory
5. Memory retrieval
6. Persistent memory
7. Knowledge vs Memory
8. Combining retrieved knowledge with contextual information

## LangGraph Comparison

Since similar concepts exist in LangGraph:

```text
CrewAI                         LangGraph

Agent                          Node
Crew                           Graph execution
Memory                         State / persistence
Knowledge retrieval             Retriever node
Task                            Node responsibility
Process                         Graph execution flow
```

The implementations differ, but both frameworks can be used to build stateful, retrieval-augmented agentic systems.

## Running the Project

From the repository root:

```bash
python 08_knowledge_assistant/main.py
```

Make sure your `.env` contains the required Gemini API configuration.

## Key Interview Takeaway

> Knowledge provides agents with external information they can retrieve, while memory allows agents to retain and retrieve relevant information from previous interactions. Combining both allows an agent to produce grounded and context-aware responses.
