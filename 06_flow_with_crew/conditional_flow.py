from crewai.flow.flow import Flow, start, listen, router


class ConditionalFlow(Flow):

    @start()
    def get_score(self):
        score = 75
        print(f"Score: {score}")
        return score

    @router(get_score)
    def check_score(self, score):

        if score >= 50:
            return "pass"
        else:
            return "fail"

    @listen("pass")
    def passed(self):
        return "Student passed!"

    @listen("fail")
    def failed(self):
        return "Student failed!"


flow = ConditionalFlow()

result = flow.kickoff()

print("\n===== FINAL RESULT =====\n")
print(result)