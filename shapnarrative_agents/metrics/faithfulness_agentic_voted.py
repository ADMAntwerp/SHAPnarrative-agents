import numpy as np
import pandas as pd
from dataclasses import asdict
from shapnarrative_agents.agents.FaithfulEvaluator import FaithfulEvaluator
from shapnarrative_agents.experiment_management.experiment_dataclass_agentic_coherence import AgenticNarrativeExperiment, ExperimentMetrics
from typing import Type


"""
This script is used to compute the faithfulness results on ensemble method. using the extraction_voted as the final dictionary. 
"""


def compute_faithfulness_agentic(experiments: list[Type[AgenticNarrativeExperiment]]) -> list[Type[ExperimentMetrics]]:
    def average_zero(df):
        values = df.values.flatten()
        numeric_values = values[np.isfinite(values)]
        num_zeros = np.sum(numeric_values == 0)
        accuracy= (80-(len(numeric_values)-num_zeros)) / 80
        return accuracy

    num_feat = 10
    rank_cols = [f"rank_{i}" for i in range(num_feat)]
    sign_cols = [f"sign_{i}" for i in range(num_feat)]
    value_cols = [f"value_{i}" for i in range(num_feat)]
    metrics = []

    for experiment in experiments:
        rank_diff_dict = {}
        sign_diff_dict = {}
        value_diff_dict = {}
        rank_accuracy_dict = {}
        sign_accuracy_dict = {}
        value_accuracy_dict = {}

        for model_name, all_instances in experiment.extractions_dict_voted.items(): # here we use the voted version of extraction dict
            # Determine max number of rounds observed across all instances
            max_rounds = max(len(rounds) for rounds in all_instances)

            rank_by_round = {r: [[np.nan] * num_feat for _ in range(len(all_instances))] for r in range(max_rounds)}
            sign_by_round = {r: [[np.nan] * num_feat for _ in range(len(all_instances))] for r in range(max_rounds)}
            value_by_round = {r: [[np.nan] * num_feat for _ in range(len(all_instances))] for r in range(max_rounds)}

            for i, instance_rounds in enumerate(all_instances):
                explanation = experiment.explanation_list[i]

                for round_num, round_data in enumerate(instance_rounds):  
                    if isinstance(round_data, dict):
                        extractions = [round_data]  # voted version
                    else:
                        extractions = round_data  # normal version
                    
                    for extraction in extractions:
                        rank_diff, sign_diff, value_diff, _, _ = FaithfulEvaluator.get_diff(extraction, explanation)
                    
                        rank_diff = rank_diff[:num_feat]
                        sign_diff = sign_diff[:num_feat]
                        value_diff = value_diff[:num_feat]

                        # Pad with NaNs
                        rank_diff += [np.nan] * (num_feat - len(rank_diff))
                        sign_diff += [np.nan] * (num_feat - len(sign_diff))
                        value_diff += [np.nan] * (num_feat - len(value_diff))

                        rank_by_round[round_num][i] = rank_diff
                        sign_by_round[round_num][i] = sign_diff
                        value_by_round[round_num][i] = value_diff


            # Convert to DataFrames and compute accuracy for each round
            rank_diff_dict[model_name] = {}
            sign_diff_dict[model_name] = {}
            value_diff_dict[model_name] = {}
            rank_accuracy_dict[model_name] = {}
            sign_accuracy_dict[model_name] = {}
            value_accuracy_dict[model_name] = {}

            for r in range(max_rounds):
                rank_df = pd.DataFrame(rank_by_round[r], columns=rank_cols)
                sign_df = pd.DataFrame(sign_by_round[r], columns=sign_cols)
                value_df = pd.DataFrame(value_by_round[r], columns=value_cols)

                rank_diff_dict[model_name][f"round_{r}"] = rank_df
                sign_diff_dict[model_name][f"round_{r}"] = sign_df
                value_diff_dict[model_name][f"round_{r}"] = value_df

                rank_accuracy_dict[model_name][f"round_{r}"] = average_zero(rank_df)
                sign_accuracy_dict[model_name][f"round_{r}"] = average_zero(sign_df)
                value_accuracy_dict[model_name][f"round_{r}"] = average_zero(value_df)

        metric = ExperimentMetrics(
            **asdict(experiment),
            rank_diff=rank_diff_dict,
            sign_diff=sign_diff_dict,
            value_diff=value_diff_dict,
            rank_accuracy=rank_accuracy_dict,
            sign_accuracy=sign_accuracy_dict,
            value_accuracy=value_accuracy_dict,
        )
        metrics.append(metric)

    return metrics
