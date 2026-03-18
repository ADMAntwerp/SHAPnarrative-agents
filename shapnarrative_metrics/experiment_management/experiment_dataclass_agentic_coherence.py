from dataclasses import dataclass, field
import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator
from typing import Tuple


@dataclass
class AgenticNarrativeExperiment:

    """This class contains data of one experiment where an experiment is defined as 
       one single list of narratives for a fixed data/prompt/model type with the corresponding embeddings and extractions in dicts.
       Comment out related agents if they are not present in differen designs. 
       Comment out or leave faithful_critic_model:str depends on if the faithful critic's version need llm. """
    

    dataset: str
    id: Tuple
    prompt_type: str
    tar_model_name: str
    num_feat: int
    idx_list: list[str]


    explanation_list: list[pd.DataFrame]
    results_list: list[pd.DataFrame]
    narrative_list: list[str]
    narrator_model: str
    extractions_dict: dict[list[dict]]
    #Add this if majority vote; if not, only keep extractions_dict
    extractions_dict_voted: dict[list[dict]]
    feedback_list: list[str]
    coherence_model:str
    feedback_all_coherence: list[str] 
    #Depends on if the faithful critic's version need llm
    faithful_critic_model:str 



@dataclass
class ExperimentMetrics(AgenticNarrativeExperiment):

    """This class inherits from an experiment dataclass and contains additional metrics"""
   
    #faithfulness
    rank_diff: dict= field(default_factory=dict)
    sign_diff: dict= field(default_factory=dict)
    value_diff:dict= field(default_factory=dict)
    rank_accuracy: dict= field(default_factory=dict)
    sign_accuracy: dict= field(default_factory=dict)
    value_accuracy: dict= field(default_factory=dict)