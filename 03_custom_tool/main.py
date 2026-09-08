from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
import ast
import operator

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash"
)


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

def safe_calculate(expression: str):
    node = ast.parse(expression, mode="eval").body

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = safe_calculate(ast.unparse(node.left))
        right = safe_calculate(ast.unparse(node.right))
        return OPERATORS[type(node.op)](left, right)

    raise ValueError("Unsupported mathematical expression")

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression using basic arithmetic operations."""
    print(f"\n🔧 CALCULATOR TOOL CALLED: {expression}")

    try:
        result = safe_calculate(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def search_tool(query: str) -> str:
    """Search for information about a topic."""
    print(f"\n🔍 SEARCH TOOL CALLED: {query}")

    return f"Search results for: {query}"

calculator_agent = Agent(
    role="Research Assistant",
    goal="Research information and perform calculations accurately",
    backstory="You are an intelligent research assistant who knows when to search for information and when to perform calculations.",
    tools=[calculator, search_tool],
    llm=llm
)

task = Task(
    description="""
    Find information about the population of India
    and then calculate what the population would be
    after a 10% increase.

    Use the available tools when appropriate.
    """,
    expected_output="""
    Provide the original population, the calculation,
    and the population after a 10% increase.
    """,
    agent=calculator_agent
)

crew = Crew(
    agents=[calculator_agent],
    tasks=[task]
)



result = crew.kickoff()

print(result)