import argparse
import yaml
import pandas as pd
from shapnarrative_agents.llm_tools.llm_wrappers import GptApi, ClaudeApi, MistralApi,DeepSeekApi, OpenRouterAPI
import numpy as np
import dill
from dataclasses import dataclass
from typing import Tuple
from shapnarrative_agents.experiment_management.experiment_manager_vote import AgenticExperimentManager


"""THIS IS THE MAIN SCRIPT OF THE COHERENT AND COHERENT-RULE DESIGN, WITH ALL FOUR AGENTS. FOR ALL TYPES OF PARAMETERS AND COMBIANTIONS
   DEPENDING ON THE SPECIFIC EXPERIMENT MANUALLY CHANGE PATHS, SIZE LIMIT, NUM_FEAT etc"""

#PATHS
TEMP_SAVE_PATH="results/temp_agent/latest_experiment.pkl"
SAVE_PATHS=[f"final_test.pkl" ]
BASELINE_PATH="data/baseline_narratives.json"

SIZE_LIMIT=2
#["deepseek-v3","gpt-5-chat","claude-sonnet-4.5","mistral-medium-2508","llama-3.3-70b"]
# Original version, the length of each agent's LLMs should be the same
# Ensemble version, there should be more than one models to be multiple voters of FFE. 
NARRATOR_MODEL_LIST=["claude-sonnet-4.5"]   
FAITHFULEVALUATOR_MODEL_LIST=["claude-sonnet-4.5", "gpt-5-chat"]    
#comment this out if using critic_no_llm version
FAITHFULCRITIC_MODEL_LIST=["claude-sonnet-4.5"]   
COHERENCE_MODEL_LIST=["claude-sonnet-4.5"]   
DATASET_NAMES=["diabetes"] 
TAR_MODEL_NAMES=["RF"]
PROMPT_TYPES=["long"]

NUM_FEAT=4
MANIPULATE=0

#LOAD CONFIG:
with open("config/keys.yaml", "r") as file:
    config_data = yaml.safe_load(file)
# version of deepseek api, from official web, thus no specific model checkpoint; should change to open router
# DeepSeek_API_key = config_data["API_keys"]["DeepSeek"]
OpenAI_API_key = config_data["API_keys"]["OpenAI"]
Llama_API_key = config_data["API_keys"]["OpenRouter"]
Claude_API_key =config_data["API_keys"]["Claude"]
Mistral_API_key=config_data["API_keys"]["Mistral"]
Voyage_API_key = config_data["API_keys"]["Voyage"]
DeepSeek_API_key_2 = config_data["API_keys"]["OpenRouter"]

#LLM PARAMS
SYSTEM_ROLE_NARRATOR="You are a helpful agent that writes model explanations (narratives) based on SHAP values." 
SYSTEM_ROLE_EVALUATOR="You are an analyst that extracts information from a narrative."
#comment this out if using critic_no_llm version
SYSTEM_ROLE_CRITIC="You are an agent that summarizes the information."
SYSTEM_ROLE_COHERENCE= """
You are a coherence evaluation agent.
Your task is to assess the logical coherence of the given narrative explanation.
Coherence refers to how well the parts of the narrative fit together to form a unified and logically consistent explanation. 
"""
# SYSTEM_ROLE_EXTRACTION="You are an analyst that extracts information from a narrative."
TEMPERATURE=0


NARRATOR_WRAPPERS={ 
            # "deepseek-v3": DeepSeekApi(DeepSeek_API_key,model="deepseek-chat", system_role=SYSTEM_ROLE_NARRATOR, temperature=TEMPERATURE),
            "deepseek-v3.2": OpenRouterAPI(DeepSeek_API_key_2, model="deepseek/deepseek-v3.2-exp",system_role=SYSTEM_ROLE_NARRATOR, temperature=TEMPERATURE),
            "gpt-5-chat": GptApi(OpenAI_API_key,model="gpt-5-chat-latest", system_role=SYSTEM_ROLE_NARRATOR, temperature=TEMPERATURE),
            "claude-sonnet-4.5": ClaudeApi(Claude_API_key,model="claude-sonnet-4-5-20250929", system_role=SYSTEM_ROLE_NARRATOR, temperature=TEMPERATURE),
            "llama-3.3-70b": OpenRouterAPI(Llama_API_key, model="meta-llama/llama-3.3-70b-instruct",system_role=SYSTEM_ROLE_NARRATOR, temperature=TEMPERATURE),
            "mistral-medium-2508": MistralApi(api_key=Mistral_API_key, model="mistral-medium-2508" ,system_role=SYSTEM_ROLE_NARRATOR, temperature=TEMPERATURE),
            }

FAITHFULEVALUATOR_WRAPPERS={
            # "deepseek-v3": DeepSeekApi(DeepSeek_API_key,model="deepseek-chat", system_role=SYSTEM_ROLE_EVALUATOR, temperature=TEMPERATURE),
            "deepseek-v3.2": OpenRouterAPI(DeepSeek_API_key_2, model="deepseek/deepseek-v3.2-exp",system_role=SYSTEM_ROLE_EVALUATOR, temperature=TEMPERATURE),
            "gpt-5-chat": GptApi(OpenAI_API_key,model="gpt-5-chat-latest", system_role=SYSTEM_ROLE_EVALUATOR, temperature=TEMPERATURE),
            "claude-sonnet-4.5": ClaudeApi(Claude_API_key,model="claude-sonnet-4-5-20250929", system_role=SYSTEM_ROLE_EVALUATOR, temperature=TEMPERATURE),
            "llama-3.3-70b": OpenRouterAPI(Llama_API_key, model="meta-llama/llama-3.3-70b-instruct",system_role=SYSTEM_ROLE_EVALUATOR, temperature=TEMPERATURE),
            "mistral-medium-2508": MistralApi(api_key=Mistral_API_key, model="mistral-medium-2508" ,system_role=SYSTEM_ROLE_EVALUATOR, temperature=TEMPERATURE),
            }


FAITHFULCRITIC_WRAPPERS={
            # "deepseek-v3": DeepSeekApi(DeepSeek_API_key,model="deepseek-chat", system_role=SYSTEM_ROLE_CRITIC, temperature=TEMPERATURE),
            "deepseek-v3.2": OpenRouterAPI(DeepSeek_API_key_2, model="deepseek/deepseek-v3.2-exp",system_role=SYSTEM_ROLE_CRITIC, temperature=TEMPERATURE),
            "gpt-5-chat": GptApi(OpenAI_API_key,model="gpt-5-chat-latest", system_role=SYSTEM_ROLE_CRITIC, temperature=TEMPERATURE),
            "claude-sonnet-4.5": ClaudeApi(Claude_API_key,model="claude-sonnet-4-5-20250929", system_role=SYSTEM_ROLE_CRITIC, temperature=TEMPERATURE),
            "llama-3.3-70b": OpenRouterAPI(Llama_API_key, model="meta-llama/llama-3.3-70b-instruct",system_role=SYSTEM_ROLE_CRITIC, temperature=TEMPERATURE),
            "mistral-medium-2508": MistralApi(api_key=Mistral_API_key, model="mistral-medium-2508" ,system_role=SYSTEM_ROLE_CRITIC, temperature=TEMPERATURE),
            }


COHERENCE_WRAPPERS={
            # "deepseek-v3": DeepSeekApi(DeepSeek_API_key,model="deepseek-chat", system_role=SYSTEM_ROLE_COHERENCE, temperature=TEMPERATURE),
            "deepseek-v3.2": OpenRouterAPI(DeepSeek_API_key_2, model="deepseek/deepseek-v3.2-exp",system_role=SYSTEM_ROLE_COHERENCE, temperature=TEMPERATURE),
            "gpt-5-chat": GptApi(OpenAI_API_key,model="gpt-5-chat-latest", system_role=SYSTEM_ROLE_COHERENCE, temperature=TEMPERATURE),
            "claude-sonnet-4.5": ClaudeApi(Claude_API_key,model="claude-sonnet-4-5-20250929", system_role=SYSTEM_ROLE_COHERENCE, temperature=TEMPERATURE),
            "llama-3.3-70b": OpenRouterAPI(Llama_API_key, model="meta-llama/llama-3.3-70b-instruct",system_role=SYSTEM_ROLE_COHERENCE, temperature=TEMPERATURE),
            "mistral-medium-2508": MistralApi(api_key=Mistral_API_key, model="mistral-medium-2508" ,system_role=SYSTEM_ROLE_COHERENCE, temperature=TEMPERATURE),
            }



DS_PATHS={
    # "fifa": "data/fifa_dataset",
    # "credit": "data/credit_dataset",
    # "student": "data/student_dataset",
    "diabetes": "data/diabetes_dataset",
    "stroke": "data/stroke_dataset",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agentic narrative generation experiment.")
    parser.add_argument("--SAVE_PATHS", '--save_paths_list' , nargs='+', default=SAVE_PATHS, type=list, help=f"Path to dir where the narratives will be saved")
    parser.add_argument("--BASELINE_PATH", '--baseline_narratives' ,default=BASELINE_PATH, type=str, help=f"path where saved baseline narratives")    
    parser.add_argument("--NUM_FEAT", default=NUM_FEAT, type=int, help=f"Number of features to use in narrative")
    parser.add_argument("--DATASET_NAMES", '--dataset_names_list', nargs='+', default=DATASET_NAMES,  help=f"Names of datasets in experiment")
    parser.add_argument("--TAR_MODEL_NAMES", '--target_names_list', nargs='+', default=TAR_MODEL_NAMES,  help=f"Names of target models")
    parser.add_argument("--NARRATOR_MODEL_LIST", '--narrator_model_names_list', nargs='+', default=NARRATOR_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {NARRATOR_WRAPPERS.keys()}")
    parser.add_argument("--FAITHFULEVALUATOR_MODEL_LIST", '--faithfulevaluator_model_names_list', nargs='+', default=FAITHFULEVALUATOR_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {FAITHFULEVALUATOR_WRAPPERS.keys()}")
    #comment this out if using critic_no_llm version
    parser.add_argument("--FAITHFULCRITIC_MODEL_LIST", '--faithfulcritic_model_names_list', nargs='+', default=FAITHFULCRITIC_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {FAITHFULCRITIC_WRAPPERS.keys()}")
    parser.add_argument("--COHERENCE_MODEL_LIST", '--coherence_model_names_list', nargs='+', default=COHERENCE_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {COHERENCE_WRAPPERS.keys()}")
    parser.add_argument("--PROMPT_TYPES", '--prompt_types_list', nargs='+', default=PROMPT_TYPES,  help=f"List of types for prompt experiment")
    parser.add_argument("--SIZE_LIMIT", type=int, default=SIZE_LIMIT,  help=f"Max number of samples to be taken from test set")
    parser.add_argument("--WARM_START", type=int, default=0,  help=f"Warm start for experiments from existing temp dict")
    parser.add_argument("--MANIPULATE", type=int, default=MANIPULATE,  help=f"Manipulate experiments or not")

    args = parser.parse_args()


    narrator_models=[NARRATOR_WRAPPERS[model_name] for model_name in args.NARRATOR_MODEL_LIST]
    faithfulevaluator_models=[FAITHFULEVALUATOR_WRAPPERS[model_name] for model_name in args.FAITHFULEVALUATOR_MODEL_LIST]
    #comment this out if using critic_no_llm version
    faithfulcritic_models=[FAITHFULCRITIC_WRAPPERS[model_name] for model_name in args.FAITHFULCRITIC_MODEL_LIST]
    coherence_models=[COHERENCE_WRAPPERS[model_name] for model_name in args.COHERENCE_MODEL_LIST]
    
    print(f"MANIP = {args.MANIPULATE}")

    for save_path in args.SAVE_PATHS:
        manager = AgenticExperimentManager(
            dataset_names=args.DATASET_NAMES,
            tar_model_names=args.TAR_MODEL_NAMES,
            narrator_models=narrator_models,
            faithfulevaluator_models=faithfulevaluator_models,
            #comment this out if using critic_no_llm version
            faithfulcritic_models=faithfulcritic_models,
            coherence_models=coherence_models,
            prompt_types=args.PROMPT_TYPES,
            ds_paths=DS_PATHS,
            size_lim=args.SIZE_LIMIT,
            num_feat=args.NUM_FEAT,
            baseline_path=args.BASELINE_PATH, 
        )

        experiment_list=manager.run_agentic_experiments(temp_save_path=TEMP_SAVE_PATH, warm_start=bool(args.WARM_START),manipulate=bool(args.MANIPULATE)) #, manipulation_func=MANIPULATE_FUNC

        with open(save_path, "wb") as f:
            dill.dump(experiment_list, f)