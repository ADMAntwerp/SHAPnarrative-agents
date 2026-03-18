import argparse
import yaml
import pandas as pd
from shapnarrative_metrics.llm_tools.llm_wrappers import GptApi, ClaudeApi, MistralApi,DeepSeekApi, OpenRouterAPI
from shapnarrative_metrics.llm_tools.generation import GenerationModel
from shapnarrative_metrics.experiment_management.experiment_manager_no_baseline import ExperimentManager
from shapnarrative_metrics.experiment_management.experiment_dataclass import NarrativeExperiment
# from shapnarrative_metrics.llm_tools.embedding_wrappers import EmbedWrapper, VoyageEmbedder
from shapnarrative_metrics.misc_tools.manipulations import full_inversion, shap_permutation
import json
import numpy as np
import pickle
import dill
from dataclasses import dataclass
from typing import Tuple


"""THIS IS THE MAIN SCRIPT TO GENERATE NARRATIVE TOGETHER WITH EXTRACTIONS FOR ALL TYPES OF PARAMETERS AND COMBIANTIONS
   DEPENDING ON THE SPECIFIC EXPERIMENT MANUALLY CHANGE PATHS, MANIPULATE BOOL and MANIPULATE FUNC etc"""

#PATHS
HUMAN_NARRATIVE_PATHS="data/human_written.json"
TEMP_SAVE_PATH="results/temp/latest_experiment.pkl"
# SAVE_PATHS=["results/old_results/test_newdatasets.pkl"]
SAVE_PATHS=[f"results/standard_experiments/experiment_{i}/experiment_8fea.pkl" for i in range(4,5)]
# if with baseline narratives
# BASELINE_PATH="data/baseline_narratives.json"

#DEFAULT PARAMS
SIZE_LIMIT=20
GEN_MODEL_LIST=["deepseek-v3"]
# GEN_MODEL_LIST=["deepseek-v3","gpt-5-chat","claude-sonnet-3.7","mistral-medium-2508","llama-3.3-70b"]
EXT_MODEL_LIST=["deepseek-v3"]
# EXT_MODEL_LIST=["deepseek-v3","gpt-5-chat","claude-sonnet-3.7","mistral-medium-2508","llama-3.3-70b"]

# EMB_MODEL_LIST=["voyage-large-2-instruct"]
DATASET_NAMES=["diabetes","stroke","fifa","credit","student"]  #["fifa","credit","student"]
TAR_MODEL_NAMES=["RF"]
PROMPT_TYPES=["long"]
# PROMPT_TYPES=["long","short"]

NUM_FEAT=8
MANIPULATE=0
MANIPULATE_FUNC=full_inversion
# MANIPULATE_FUNC=shap_permutation
APPEND_HUMAN=False

#LOAD CONFIG:
with open("config/keys.yaml", "r") as file:
    config_data = yaml.safe_load(file)
OpenAI_API_key = config_data["API_keys"]["OpenAI"]
Llama_API_key = config_data["API_keys"]["OpenRouter"]
Claude_API_key =config_data["API_keys"]["Claude"]
Gemini_API_key = config_data["API_keys"]["Gemini"]
Mistral_API_key=config_data["API_keys"]["Mistral"]
Voyage_API_key = config_data["API_keys"]["Voyage"]
DeepSeek_API_key_2 = config_data["API_keys"]["OpenRouter"]

#LLM PARAMS
SYSTEM_ROLE="You are a helpful agent that writes model explanations (narratives) based on SHAP values."
SYSTEM_ROLE_EXTRACTION="You are an analyst that extracts information from a narrative."
GENERATION_TEMPERATURE=0
EXTRACTION_TEMPERATURE=0

#Generation models
LLM_WRAPPERS={ 
            "deepseek-v3": OpenRouterAPI(DeepSeek_API_key_2, model="deepseek/deepseek-chat-v3-0324",system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
            "deepseek-v3.2": OpenRouterAPI(DeepSeek_API_key_2, model="deepseek/deepseek-v3.2-exp",system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
            "gpt-5-chat": GptApi(OpenAI_API_key,model="gpt-5-chat-latest", system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
            "claude-sonnet-3.7": ClaudeApi(Claude_API_key,model="claude-3-7-sonnet-20250219", system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
            "claude-sonnet-4.5": ClaudeApi(Claude_API_key,model="claude-sonnet-4-5-20250929", system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
            # "gemini-2.5-flash": GoogleApi(Gemini_API_key,model="gemini-2.5-flash", system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
            "llama-3.3-70b": OpenRouterAPI(Llama_API_key, model="meta-llama/llama-3.3-70b-instruct", system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
            "mistral-medium-2508": MistralApi(api_key=Mistral_API_key, model="mistral-medium-2508" ,system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
            }
# Extraction models
EXT_WRAPPERS={
            # "deepseek-v3": DeepSeekApi(DeepSeek_API_key,model="deepseek-chat", system_role=SYSTEM_ROLE, temperature=EXTRACTION_TEMPERATURE), 
            "deepseek-v3": OpenRouterAPI(DeepSeek_API_key_2, model="deepseek/deepseek-chat-v3-0324",system_role=SYSTEM_ROLE_EXTRACTION, temperature=EXTRACTION_TEMPERATURE),
            "deepseek-v3.2": OpenRouterAPI(DeepSeek_API_key_2, model="deepseek/deepseek-v3.2-exp",system_role=SYSTEM_ROLE_EXTRACTION, temperature=EXTRACTION_TEMPERATURE),
            "gpt-5-chat": GptApi(OpenAI_API_key,model="gpt-5-chat-latest", system_role=SYSTEM_ROLE_EXTRACTION, temperature=EXTRACTION_TEMPERATURE),
            "claude-sonnet-4.5": ClaudeApi(Claude_API_key,model="claude-sonnet-4-5-20250929", system_role=SYSTEM_ROLE_EXTRACTION, temperature=EXTRACTION_TEMPERATURE),
            # "claude-sonnet-3.7": ClaudeApi(Claude_API_key,model="claude-3-7-sonnet-20250219", system_role=SYSTEM_ROLE_EXTRACTION, temperature=EXTRACTION_TEMPERATURE),
            # "gemini-2.5-flash": GoogleApi(Gemini_API_key,model="gemini-2.5-flash", system_role=SYSTEM_ROLE, temperature=EXTRACTION_TEMPERATURE),
            "llama-3.3-70b": OpenRouterAPI(Llama_API_key, model="meta-llama/llama-3.3-70b-instruct", system_role=SYSTEM_ROLE_EXTRACTION, temperature=EXTRACTION_TEMPERATURE),
            "mistral-medium-2508": MistralApi(api_key=Mistral_API_key, model="mistral-medium-2508" ,system_role=SYSTEM_ROLE_EXTRACTION, temperature=EXTRACTION_TEMPERATURE),
            }

# EMB_WRAPPERS={
#                "voyage-large-2-instruct": VoyageEmbedder(api_key=Voyage_API_key, model="voyage-large-2-instruct")
#                }

DS_PATHS={
    "fifa": "data/fifa_dataset",
    "credit": "data/credit_dataset",
    "student": "data/student_dataset",
    "diabetes": "data/diabetes_dataset",
    "stroke": "data/stroke_dataset",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a series of narratives and save them")
    #In argparse, This is part of the command-line interface (CLI) syntax.; any argument that starts with -- is considered a flag or option that the user can pass via the terminal.
    #So this line: parser.add_argument("--SIZE_LIMIT", ...) means that from the command line, you can do: python script.py --SIZE_LIMIT 100
    parser.add_argument("--SAVE_PATHS", '--save_paths_list' , nargs='+', default=SAVE_PATHS, type=list, help=f"Path to dir where the narratives will be saved")
    #if with baseline narratives
    # parser.add_argument("--BASELINE_PATH", '--baseline_narratives' ,default=BASELINE_PATH, type=str, help=f"path where saved baseline narratives")        
    parser.add_argument("--NUM_FEAT", default=NUM_FEAT, type=int, help=f"Number of features to use in narrative")
    parser.add_argument("--DATASET_NAMES", '--dataset_names_list', nargs='+', default=DATASET_NAMES,  help=f"Names of datasets in experiment")
    parser.add_argument("--TAR_MODEL_NAMES", '--target_names_list', nargs='+', default=TAR_MODEL_NAMES,  help=f"Names of target models")
    parser.add_argument("--GEN_MODEL_LIST", '--gen_model_names_list', nargs='+', default=GEN_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {LLM_WRAPPERS.keys()}")
    parser.add_argument("--EXT_MODEL_LIST", '--ext_model_names_list', nargs='+', default=EXT_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {EXT_WRAPPERS.keys()}")
    # parser.add_argument("--EMB_MODEL_LIST", '--emb_model_names_list', nargs='+', default=EMB_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {EMB_WRAPPERS.keys()}")
    parser.add_argument("--PROMPT_TYPES", '--prompt_types_list', nargs='+', default=PROMPT_TYPES,  help=f"List of types for prompt experiment")
    parser.add_argument("--SIZE_LIMIT", type=int, default=SIZE_LIMIT,  help=f"Max number of samples to be taken from test set")
    #This allows you to run from terminal like: python run_experiments.py --WARM_START 1
    parser.add_argument("--WARM_START", type=int, default=0,  help=f"Warm start for experiments from existing temp dict")
    parser.add_argument("--MANIPULATE", type=int, default=MANIPULATE,  help=f"Manipulate experiments or not")

    args = parser.parse_args()

    gen_models=[LLM_WRAPPERS[model_name] for model_name in args.GEN_MODEL_LIST]
    ext_models=[EXT_WRAPPERS[model_name] for model_name in args.EXT_MODEL_LIST]
    # emb_models=[EMB_WRAPPERS[model_name] for model_name in args.EMB_MODEL_LIST]
    
    print(f"MANIP = {args.MANIPULATE}")

    for save_path in args.SAVE_PATHS:

        manager=ExperimentManager(
            dataset_names=args.DATASET_NAMES,
            tar_model_names=args.TAR_MODEL_NAMES,
            generation_models=gen_models,
            extraction_models=ext_models,
            # embedding_models=emb_models,
            prompt_types=args.PROMPT_TYPES,
            ds_paths=DS_PATHS,
            size_lim=args.SIZE_LIMIT,
            num_feat=args.NUM_FEAT,
            # baseline_path=args.BASELINE_PATH, 
        )
        #bool(args.WARM_START) Converts the integer 0 or 1 into a Python Boolean: False or True.
        experiment_list=manager.run_experiments(temp_save_path=TEMP_SAVE_PATH, warm_start=bool(args.WARM_START),manipulate=bool(args.MANIPULATE), manipulation_func=MANIPULATE_FUNC)

        #ADD HUMAN WRITTEN NARRATIVES FROM DICT IN THE SAME FORMAT THE LLM EXPERIMENTS
        if APPEND_HUMAN is True:

            with open(HUMAN_NARRATIVE_PATHS,"r") as f:
                human_dict=json.load(f)
            experiment_list=manager.append_human(
                                                 experiments=experiment_list,
                                                 human_dict=human_dict,
                                                 ext_models=[EXT_WRAPPERS[ext_model_name] for ext_model_name in EXT_MODEL_LIST],
                                                #  emb_models=[EMB_WRAPPERS[emb_model_name] for emb_model_name in EMB_MODEL_LIST]
                                                 )
            

        with open(save_path, "wb") as f:
            dill.dump(experiment_list, f)