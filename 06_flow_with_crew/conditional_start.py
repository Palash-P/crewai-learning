from crewai.flow.flow import Flow, start, router


class RetryFlow(Flow):

    @start()
    def initial_step(self):
        print("Initial step")
        return "retry"

    @router(initial_step)
    def decide(self):
        return "retry"

    @start("retry")
    def retry_step(self):
        print("Retry step")


flow = RetryFlow()

flow.kickoff()