import pandas as pd
import numpy as np
from shapnarrative_agents.agents.LLMBaseAgent import LLMBaseAgent
from typing import Optional, Any, Dict

class FaithfulCritic(LLMBaseAgent):
    """
    A simplified faithful critic agent that generates
    plain-language instructions based on the SHAP explanation table.

    This version is for when the feature number is 8.
    """

    def __init__(self, ds_info: Dict,llm: Optional[Any] = None):
        """
        Parameters:
        -----------
        ds_info : dict
            Contains dataset information, including feature descriptions.
        llm : optional
            LLM wrapper (not directly used here).
        """
        self.ds_info = ds_info
        self.llm = llm
        # self.model = llm.model 
        self.model = getattr(llm, "model", None)

    def generate_rank_feedback(self, shap_df: pd.DataFrame, df_diff: pd.DataFrame) -> str:
        """
        Generates feedback about rank differences.
        """
        feedback_lines = []

        expected_len = min(8, len(shap_df)) 
        actual_len = len(df_diff)

        if actual_len != expected_len:
            # print ("-⚠️ Warning: The narrative may include a missing or extra feature. Please check.")
            # pass
            feedback_lines.append("-⚠️ Warning: The narrative may include a missing or extra feature. Please check.")

        for i in range(min(expected_len, actual_len)):
            row = shap_df.iloc[i]
            feature = row["feature_name"]
            rank_diff = df_diff.iloc[i]["rank"]

            if pd.isnull(rank_diff) or np.isinf(rank_diff):
                feedback_lines.append(
                    f"- Warning: Feature '{feature}' may be missing from the explanation or contain a malformed rank."
                )
            elif np.isfinite(rank_diff) and rank_diff != 0:
                ordinal_position = self._ordinal(i + 1)
                feedback_lines.append(
                    f"- Please move the sentence about Feature '{feature}' to be the {ordinal_position}-mentioned feature in your explanation." #Please remind you should in total mention four feautres in your explanation.
                )

        if feedback_lines:
            feedback = "Regarding the feature rank issues in the narrative:\n" + "\n".join(feedback_lines)
        else:
            feedback = "No rank corrections needed."

        return feedback.strip()

    def _ordinal(self, n: int) -> str:
        """
        Converts an integer into its ordinal string representation.
        E.g. 1 -> 'first', 2 -> 'second', etc.

        Parameters:
        -----------
        n : int

        Returns:
        --------
        str
        """
        ordinals = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6:"sixth",
            7:"seventh",
            8:"eighth",
        }
        return ordinals.get(n, f"{n}th")

    def generate_sign_feedback(self, shap_df: pd.DataFrame, df_diff: pd.DataFrame, extraction: dict) -> str:
        """
        Generates feedback about sign differences.

        Parameters:
        -----------
        shap_df : pd.DataFrame
            DataFrame with columns including 'feature_name'.
        df_diff : pd.DataFrame
            DataFrame with columns ['rank', 'sign', 'value'] corresponding to each feature.

        Returns:
        --------
        str : A human-readable feedback string.
        """
        rank_to_feature = {}
        for feat_name, feat_info in extraction.items():
            rank = feat_info.get('rank')
            if rank is not None:
                rank_to_feature[rank] = feat_name
        
        feedback_lines = []

        expected_len = min(8, len(shap_df))
        actual_len = len(df_diff)

        if actual_len != expected_len:
            pass

        for i in range(min(expected_len, actual_len)):
            feature = rank_to_feature.get(i, f"Unknown_{i}")
            sign_diff = df_diff.iloc[i]["sign"]

            if pd.isnull(sign_diff):
                feedback_lines.append(
                    f"- You didn't mention whether Feature '{feature}' has a positive or negative influence for the prediction. Please check the SHAP table and clarify its influence."
                )
            elif np.isinf(sign_diff):
                feedback_lines.append(
                    f"- Warning: Some features extracted by the model were not in the real shap table (e.g., '{feature}')."
                )
            elif sign_diff == 1:
                feedback_lines.append(
                    f"- The positive or negative influence of Feature '{feature}' has been stated incorrectly. Please change it to be the opposite."
                )

        if feedback_lines:
            feedback = "Regarding the influence direction in the narrative:\n" + "\n".join(feedback_lines)
        else:
            feedback = "No sign corrections needed."

        return feedback.strip()
    
    def generate_value_feedback (self, shap_df: pd.DataFrame, df_diff: pd.DataFrame) -> str:
        """
        Generates feedback about avlue differences.

        Parameters:
        -----------
        shap_df : pd.DataFrame
            DataFrame with columns including 'feature_name'.
        df_diff : pd.DataFrame
            DataFrame with columns ['rank', 'sign', 'value'] corresponding to each feature.

        Returns:
        --------
        str : A human-readable feedback string.
        """
        feedback_lines = []

        expected_len = min(8, len(shap_df))
        actual_len = len(df_diff)

        if actual_len != expected_len:
            pass

        for i in range(min(expected_len, actual_len)):
            row = shap_df.iloc[i]
            feature = row["feature_name"]
            value_diff = df_diff.iloc[i]["value"]
            feature_value= row["feature_value"]

            if pd.isnull(value_diff) or value_diff == 0:
                continue

            if np.isinf(value_diff):
                feedback_lines.append(
                    f"- Warning: Some features extracted by the model were not in the real feature list (e.g., '{feature}')."
                )

            else:
                feedback_lines.append(
                    f"You extracted a wrong feature value for Feature {feature}. Please change the value to be {feature_value}."
                )

        if feedback_lines:
            feedback = "Regarding the feature value issues in the narrative:\n" + "\n".join(feedback_lines)
        else:
            feedback = "No value corrections needed."

        return feedback.strip()

    def give_feedback(self, shap_df: pd.DataFrame, df_diff: pd.DataFrame, extraction: dict) -> str:
        """
        Generates combined feedback from rank, sign, and value checks.
        If all feedbacks are 'No ... corrections needed.', returns a confirmation message.
        Otherwise, combines them, summarizes using the LLM, and returns the concise feedback.
        """
        # Step 1: Generate all feedbacks
        rank_feedback = self.generate_rank_feedback(shap_df, df_diff)
        sign_feedback = self.generate_sign_feedback(shap_df, df_diff, extraction)
        value_feedback = self.generate_value_feedback(shap_df, df_diff)

        # Step 2: Check if all feedbacks say 'No ... corrections needed.'
        if (rank_feedback == "No rank corrections needed." and
            sign_feedback == "No sign corrections needed." and
            value_feedback == "No value corrections needed."):
            return "After checking, the narrative is 100% faithful to the SHAP table."

        # Step 3: Combine feedback when no llm 
        combined_feedback = "To improve the narrative:"+ "\n\n" +rank_feedback + "\n\n" + sign_feedback+"\n\n" + value_feedback

        # Step 4: Summarize with the LLM when llm
        if self.llm:
            prompt = f"""
            ***Context***:
            You are a critic tasked with providing instructions on how to improve a narrative. 
            To do so, you are given feedback on the narrative, and you should summarize it clearly and concisely.

            ***Input text***:
            The following is the feedback. 
            ====================
            {combined_feedback}
            ====================

            ***Output Structure***:
            Free format (no strict structure required).

            ***Guidelines***:
            When you summarize, make sure to include all feedback; do not lose any information from the feedback provided.
            """
            try:
                response = self.llm.generate_response(prompt)
                return response.strip()
            except Exception:
                # Fall back if LLM call fails
                pass

        # Fallback / no-LLM path
        return combined_feedback.strip()