from crewai import Agent

researcher = Agent(
    role="AI Researcher",
    goal="Research and explain AI agents clearly",
    backstory="You are an experienced AI researcher."
)