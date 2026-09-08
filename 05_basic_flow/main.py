from crewai.flow.flow import Flow, start, listen


class BasicFlow(Flow):

    @start()
    def first_step(self):
        print("Step 1 running...")
        return "Hello from Step 1"

    @listen(first_step)
    def second_step(self, result):
        print("Step 2 running...")
        print(f"Received: {result}")

        return "Flow completed successfully"


flow = BasicFlow()

result = flow.kickoff()

print("\n===== FINAL RESULT =====\n")
print(result)