from typing import Type, Tuple, Optional,List, Sequence
import pandas as pd
import shap
import numpy as np
from shapnarrative_metrics.llm_tools.llm_wrappers import LLMWrapper
import json
from .LLMBaseAgent import LLMBaseAgent
import re
import numbers

class CoherenceAgent(LLMBaseAgent):
    
    """
    This is a Cohrence Agent. It is designed to evaluate the coherence of a narrative and provide revision instructions for improvement. 
    The agent uses a large language model (LLM) to analyze the narrative and generate feedback based on the defined criteria for coherence. 
    """
    def __init__(self, llm):
        """
        Parameters:
        -----------
        llm : optional
            LLM wrapper (not directly used here).
        """
        self.llm = llm
        self.model = llm.model 

    @staticmethod
    def format_to_line_separated_sentences(text: str) -> str:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        return "\n".join(sentences)
    
    def give_feedback(self, narrative: str) -> str:
        prompt = f"""
        ***Context***:
        You are a critic tasked with providing revision instructions to improve the coherence quality of the given narrative.
        You should first examine if coherence-related issues exist in the given narrative and then output revision instructions.

        Definition of coherence:
        Coherence refers to the overall quality of how sentences work together in a text. A coherent text is well-structured, logically organized, and builds a unified body of information on its topic. 
        It should present information that flows smoothly, avoiding abrupt transitions or disjoint statements.

        ***Input text***:
        The following is the given narrative. 
        ====================
        \"\"\"{narrative}\"\"\"
        ====================

        ***Output Structure***:
        Your output MUST include:
        1) Explicit revision commands written in a clear, standardized format, such as: “Change ___ to ___”, “Insert ___ before ___”, “Delete ___”, “Reorder ___ after ___”, etc. 
        2) A concise explanation following each command that briefly justifies the change.

        ***Guidelines***:
        1) If there are no coherence issues, reply only: no coherence issue.
        2) Focus on meaningful coherence improvements. Avoid nitpicking or unnecessary edits. 
        """
        # print(f"prompt for Coherence agent: {prompt}")
        response = self.llm.generate_response(prompt)
        return self.format_to_line_separated_sentences(response)
        