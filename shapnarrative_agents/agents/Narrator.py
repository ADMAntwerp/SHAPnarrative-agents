from .LLMBaseAgent import LLMBaseAgent
from shapnarrative_agents.llm_tools.llm_wrappers import LLMWrapper
import re


class NarratorAgent(LLMBaseAgent):
    """
    This class defines the Narrator agent that generates narratives based on an initial prompt or feedback.
    In Round 0, the Narrator only responds to the initial prompt.
    In subsequent rounds, it revises its previous answer based on feedback, which contains the initial prompt, last narrative and other agents' feedback.
    """
    @staticmethod
    def format_to_line_separated_sentences(text: str) -> str:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        return "\n".join(sentences)
    
    def generate(self, initial_prompt, last_narrative=None,faithful_feedback=None, coherence_feedback=None):
        if last_narrative is None and coherence_feedback is None:
            response = self.llm.generate_response(initial_prompt)
            return self.format_to_line_separated_sentences(response)
        else:
            final_prompt = f"""
            ***Context***:
            You are a helpful agent who writes model explanations (narratives) based on SHAP values. 
            Revise your previous narrative strictly according to the initial task and all given feedback. 
            This is your initial task: {initial_prompt}.

            ***Input text***:
            The following is the feedback. 
            ====================
            This is your previous narrative:{last_narrative}.
            This is the faithfulness-issue feedback:{faithful_feedback}.
            This is the coherence-issue feedback:{coherence_feedback}.
            ====================

            ***Output Structure***:
            The narrative MUST comply with all format related rules and content related rules from the initial task.

            ***Guidelines***:
            1) Do not modify the part of the narrative that isn't mentioned in the feedback. 
            2) You MUST return the narrative only. DO NOT chitchat.
            """

            response = self.llm.generate_response(final_prompt)
            return self.format_to_line_separated_sentences(response)