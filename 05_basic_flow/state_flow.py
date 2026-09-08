from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel


class MyState(BaseModel):
    topic: str = ""
    research: str = ""


class ResearchFlow(Flow[MyState]):

    @start()
    def get_topic(self):
        self.state.topic = "AI Agents"

        print(f"Topic: {self.state.topic}")

        return self.state.topic

    @listen(get_topic)
    def research(self):
        self.state.research = f"Research completed for {self.state.topic}"

        print(f"Research: {self.state.research}")

        return self.state.research


flow = ResearchFlow()

result = flow.kickoff()

print("\n===== FINAL RESULT =====\n")
print(result)