from shapnarrative_agents.llm_tools.llm_wrappers import LLMWrapper

class LLMBaseAgent:
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper