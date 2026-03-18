import pickle
import pandas as pd
import json
from collections import Counter
import math
import itertools
import time
import pickle
from typing import Type
from collections import Counter
from shapnarrative_metrics.agents.prompt import Prompt
from shapnarrative_metrics.agents.Narrator import NarratorAgent
from shapnarrative_metrics.agents.FaithfulEvaluator import FaithfulEvaluator
from shapnarrative_metrics.agents.FaithfulCritic import FaithfulCritic
from script.logger_utils import log_with_context   
from shapnarrative_metrics.agents.Coherence import CoherenceAgent
from shapnarrative_metrics.llm_tools.llm_wrappers import LLMWrapper
from shapnarrative_metrics.experiment_management.experiment_dataclass_agentic_coherence import AgenticNarrativeExperiment


class AgenticExperimentManager:
    
    """ 
    This class try to add multiple LLMs for faithfulevaluator to perform majority voting on their extractions, and save vorting results into extraction_dict_voted.

    A class to manage generation of narratives across different parameters. Input variables represent all experiment combinations to perform.
    Comment out or leave faithful_critic_model depends on if the faithful critic's version need llm.
    
    Attributes:
        dataset_names: (list) list of names for the target models to be used
        tar_model_names: (list) list of names for the target models to be used
        narrator_models: (list) list of narrator models
        faithfulevaluator_models: (list) list of faithfulevaluator models, that replace the extraction work of extraction models in the original experiment manager
        faithfulcritic_models:(list) list of faithfulevaluator models
        coherence_models:(list) list of faithfulevaluator models
        prompt_types: (list) list of prompt types allowed in the generation model class
        ds_paths: (dict) dictionary with ds_names as keys, and the paths to that dataset location as values
        size_lim: (int) number of narratives to create for a given dataset
        num_feat: (int) number of features to use in t
    """
        
    def __init__(
        self,
        dataset_names: list[str],
        tar_model_names: list[str],
        narrator_models: list[Type[LLMWrapper]],
        faithfulevaluator_models: list[Type[LLMWrapper]],
        # remove or add faithfulcritic's model
        faithfulcritic_models: list[Type[LLMWrapper]],
        coherence_models: list[Type[LLMWrapper]],
        prompt_types: list[str],
        ds_paths: dict,
        size_lim: int,
        num_feat: int,
        temperature: float = 0.0,
        baseline_path: str = "data/baseline_narratives.json",
    ):

        # add faithfulcritic_models or remove it depends on if llm needed; add coherence_models if the agentic system has it
        self.model_pairs = list(zip(narrator_models, faithfulcritic_models, coherence_models))    # narrator_models, coherence_models,faithfulcritic_models; no need faithfulevaluator_models because this will be multiple voters
        self.dataset_names = dataset_names
        self.tar_model_names = tar_model_names
        self.narrator_models=narrator_models
        self.faithfulevaluator_models=faithfulevaluator_models
        # add or remove it depends on if FFC llm needed
        self.faithfulcritic_models=faithfulcritic_models
        self.coherence_models=coherence_models
        self.prompt_types = prompt_types
        self.ds_paths = ds_paths
        self.size_limit = size_lim
        self.num_feat = num_feat
        self.temperature = temperature
        self.baseline_path = baseline_path
        self.baseline_lookup = self.load_narratives_json()

    def sample_equal_targets(self, df):

        """Given a dataframe representing a large test set randomly samples n_tot instances from it such that it has an equal number of classes
        Args:
            df: (pd.DataFrame) Test_set containing the class label in a "target" variable

        Returns:
            sampled_df: (pd.DataFrame) sampled test set to be used for narrative generation 
        """
        

        target_name=df.columns[-1]
        df_target_0 = df[df[target_name] == 0]
        df_target_1 = df[df[target_name] == 1]

        # Randomly sample entries from each subset
        sample_0 = df_target_0.sample(int(self.size_limit/2), random_state=42)
        sample_1 = df_target_1.sample(int(self.size_limit/2), random_state=42)

        # Combine the samples
        sampled_df = pd.concat([sample_0, sample_1])

        # Shuffle the combined sample to mix the rows
        sampled_df = sampled_df.sample(frac=1, random_state=42) 

        return sampled_df

    def dataset_extraction_tool(self, dataset_name, tar_model_name):

        """Given a dataset name reads in all necessary data from the data folder that we need for narrative generation

        Args:
            dataset_name: name of dataset to read the path through the self.ds_paths dictonary
            tar_model_name: name of the target model for which the SHAP values will be generated (e.g. RF)

        Returns:
            sampled_test_set: output of the sample_equal_targets method
            ds_info: dictionary that was prepared during data preprocessing containing necassary info for prompt
            tar_model: read-in pickled target model
        """
        
        test_set=pd.read_parquet(f"{self.ds_paths[dataset_name]}/test_cleaned.parquet") 
        sampled_test_set=self.sample_equal_targets(test_set)

        with open(f"{self.ds_paths[dataset_name]}/dataset_info", 'rb') as f:
            ds_info= pickle.load(f)
        
        with open(f"{self.ds_paths[dataset_name]}/{tar_model_name}.pkl", 'rb') as f:
            tar_model= pickle.load(f)

        return sampled_test_set, ds_info, tar_model

    def load_narratives_json(self):
        """Converts the JSON list of dicts into a fast lookup dictionary."""
        # Load Round 0 narratives from the JSON file
        with open(self.baseline_path, 'r', encoding='utf-8') as f:
            round0_data = json.load(f)

        lookup = {}
        for item in round0_data:
            key = (item['dataset'], item['index'])
            lookup[key] = item['round0_narrative']
        return lookup
        
    def get_baseline_narrative(self, dataset_name, idx):
        """Accesses the preloaded dictionary stored in lookup"""
        return self.baseline_lookup.get((dataset_name, idx), "No Round 0 narrative found.")
    

    def vote_across_extractions(self, extraction_per_llm):
        """
          Given a dict of {llm_name: extraction_dict}, returns a final voted extraction dict
          wrapped inside a list, where each feature field (e.g., sign, rank, value)
          is voted using mode.
        """
        def normalize_nan(val):
            if isinstance(val, float) and math.isnan(val):
                return "__nan__"
            return val

        def denormalize_nan(val):
            if val == "__nan__":
                return float("nan")
            return val
    
        def normalize_feature_keys(extraction):
            return {
                feat.strip(): {
                    field: val for field, val in fields.items()
                }
                for feat, fields in extraction.items()
            }

        llm_extractions = [normalize_feature_keys(ex) for ex in extraction_per_llm.values()]

        # select the most common feature keys across LLMs
        # Step 1: Collect all feature key sets
        feature_key_sets = [frozenset(ex.keys()) for ex in llm_extractions]
        # Step 2: Count how often each feature set appears
        feature_set_counts = Counter(feature_key_sets)
        # Step 3: Pick the most common feature key set
        most_common_feature_set = feature_set_counts.most_common(1)[0][0]
        reference_feature_keys = set(most_common_feature_set)

        # report extractions that don't match the reference feature keys, and igore it
        valid_extractions = []
        for i, (model_name, ex) in enumerate(zip(extraction_per_llm.keys(), llm_extractions)):
            if set(ex.keys()) != reference_feature_keys:
                warning_msg = (
                f"⚠️ Skipping LLM extraction {i} due to mismatched feature keys.\n"
                f"  Expected: {reference_feature_keys}\n"
                f"  Found:    {set(ex.keys())}"
            )
                print(warning_msg)
                log_with_context(
                    "warning",
                    warning_msg,
                    model=getattr(self, "narrator_model_active", "unknown"),
                    dataset=getattr(self, "ds_info", {}).get("dataset_name", ""),
                    instance=getattr(self, "current_instance", None),
                )
                continue
            valid_extractions.append(ex)

        if not valid_extractions:
            msg = "❌ No valid LLM extractions found. Returning empty result."
            print(msg)
            log_with_context(
                    "warning",
                    msg,
                    model=getattr(self, "narrator_model_active", "unknown"),
                    dataset=getattr(self, "ds_info", {}).get("dataset_name", ""),
                    instance=getattr(self, "current_instance", None),
                )
            return {}
        
        final_extraction = {}
        feature_keys = reference_feature_keys
        field_keys = valid_extractions[0][list(feature_keys)[0]].keys()

        temp_extraction = {}
        for feat in feature_keys:
            temp_extraction[feat] = {}
            for field in field_keys:
                raw_values = [ex[feat][field] for ex in valid_extractions]
                normalized_values = [normalize_nan(v) for v in raw_values]
                most_common_value = Counter(normalized_values).most_common(1)[0][0]
                temp_extraction[feat][field] = denormalize_nan(most_common_value)

        #Now sort features by voted rank before finalizing output
        sorted_features = sorted(
            temp_extraction.keys(),
            key=lambda f: temp_extraction[f].get("rank", float("inf"))
        )

        final_extraction = {
            feat: temp_extraction[feat] for feat in sorted_features
        }   


        return final_extraction

    def run_agentic_experiments(
         self,
         temp_save_path: str = "results/temp_agent/latest_experiment.pkl",
         warm_start: bool = False,
         manipulate: bool= False,
        ):
    
        """Executes experiments in all possible combinations we are interested in for the study using the combinations of the parameters provided at the start of class initialization.
        args:
            temp_save_path: saves intermediate results in temporary dir, in case an api call fails to avoid redoing everything.
            warm_start: start from latest temp_save_path result?
            manipulate: bool -- manipulate narratives or not
        returns:
            list of NarrativeExperiment dataclasses as defined in the experiment_dataclass module
        
            ATTENTION: the total iterations should multiply the len of critic model and len of coherence model when they are different from narrator models;
            here we didn't multiply them, because in this study we make all three models same and only one pertime of run.
        """
        experiments=[]
                
        #in case an api call failed in the previous experiment run, we can start from the last saved:
        if warm_start:
            with open(temp_save_path, 'rb') as f:
                experiments=pickle.load(f)
            existing_ids=[exp.id for exp in experiments]
        else:
            existing_ids = []
        
        # the total iterations we defined here is to only considering the len of narrator models.
        # becasue in this study, we either make all llm agent to be the same, or llm is different but we only care about one-to-one pair rather than all combinations.
        total_iterations = len(self.dataset_names) * len(self.tar_model_names) * len(self.narrator_models)* len(self.prompt_types)  
        
        #This is for when we make narrator and coherence model 1 to 1 pairs; add faithfulcritic_model into the zip if use llm
        # change to (narrator_model, coherence_model, faithfulcritic_model) when using them
        # Note: if you use the agentic version (use only narrator and faithfulevaluator, you should leave the following tuple pack to be (narrator_model,) 
        # rather than (narrator_model). Te parenthesis is important!)
        for index, (dataset_name, tar_model_name, (narrator_model, faithfulcritic_model, coherence_model), prompt_type) in enumerate(
        itertools.product(self.dataset_names, self.tar_model_names, self.model_pairs, self.prompt_types)): # use self.model_pairs
        
        #Version 1: This version is for when we stick to only one extraction model for all generation models (when only one extraction model)/ or we want to try all generation and extraction model combinations (when multiple extraction models)
        # for index, (dataset_name, tar_model_name, narrator_model, prompt_type) in enumerate(itertools.product(self.dataset_names,self.tar_model_names,self.narrator_models, self.prompt_types)):
        
            exp_id = (dataset_name, tar_model_name, narrator_model.model, prompt_type)
            if exp_id in existing_ids:
                continue

            print(f"****** Running experiment {index+1}/{total_iterations} with {exp_id} ******")
            
            per_start_time = time.time()

            sampled_test_set, ds_info, tar_model = self.dataset_extraction_tool(dataset_name, tar_model_name)
            prompt_generator = Prompt(ds_info=ds_info)
            narrator = NarratorAgent(narrator_model)

            # [choose between based on llm or not]
            faithful_critic = FaithfulCritic(ds_info,llm=faithfulcritic_model)
            # faithful_critic = FaithfulCritic(ds_info)
            coherence_agent = CoherenceAgent(llm=coherence_model)
            
            idx_list = []
            explanation_list = []
            narratives_all = []
            feedback_all = []
            results_list = []
            # Initialize extractions_dict for all LLMs
            extractions_dict = {llm.model: [] for llm in self.faithfulevaluator_models}
            extractions_list_voted = []
            feedback_all_coherence = []            

            for idx in sampled_test_set.index:
                #prepare information for vote_across_extactions
                self.ds_info = ds_info
                self.current_instance = idx
                self.narrator_model_active = narrator_model.model  # name string

                print(f"📌 Generating for instance index: {idx}")
                x = sampled_test_set[sampled_test_set.columns[0:-1]].loc[[idx]]
                y = sampled_test_set[sampled_test_set.columns[-1]].loc[[idx]]

                prompt_generator.gen_variables(tar_model, x, y, tree=True)
                shap_df = pd.DataFrame(prompt_generator.explanation_list[0])
                prompt = prompt_generator.generate_story_prompt(0, prompt_type=prompt_type, manipulate=manipulate )

                round_narratives = []
                round_feedback = []
                round_coherence = []
                round_extractions_per_llm = {llm.model: [] for llm in self.faithfulevaluator_models}
                round_extractions_voted = []

                # Round 0 narrative is from baseline narrative
                narrative = self.get_baseline_narrative(dataset_name,idx)
                print(f"📝 Baseline Narrative Round 0")
                round_narratives.append(narrative)

                for round_num in range(3):
                    extraction_per_llm = {}

                    # Extract using all LLMs
                    for llm in self.faithfulevaluator_models:
                        evaluator = FaithfulEvaluator(ds_info, llm=llm)
                        extraction = evaluator.generate_extractions([[narrative]])  # returns list of list
                        extraction_per_llm[llm.model] = extraction[0]
                        round_extractions_per_llm[llm.model].append(extraction)  # Save this round for this instance

                    # Voting step: majority or aggregation
                    final_extraction = self.vote_across_extractions(extraction_per_llm)
                    round_extractions_voted.append(final_extraction)

                    # Even though here use a specific faithful evaluator and a specific llm, in this action it acctually didn't leverage the real use of LLM;
                    # only use the static function inside faithful evaluator to get the diff.
                    main_eval = FaithfulEvaluator(ds_info, llm=self.faithfulevaluator_models[0])
                    rank_diff, sign_diff, value_diff, real_rank, extracted_rank = main_eval.get_diff(final_extraction, shap_df)

                    df_diff = pd.DataFrame({
                        "rank": rank_diff,
                        "sign": sign_diff,
                        "value": value_diff
                    })
                    
                    faithful_feedback = main_eval.give_feedback(df_diff, final_extraction)
                    faithful_feedback = faithful_critic.give_feedback(shap_df, df_diff, final_extraction)

                    round_feedback.append(faithful_feedback)
                    # print(f"🎯 Faithfulness Feedback Round {round_num}: {faithful_feedback[0:50]}...")

                    coherence_feedback=coherence_agent.give_feedback(narrative)
                    # print(f"💬 Coherence Feedback Round {round_num}: {coherence_feedback[0:50]}...")
                    round_coherence.append(coherence_feedback)

                    # if "100% faithful" in faithful_feedback.lower():
                    #     print("✅ Feedback indicates 100% faithful narrative, stopping further rounds.")
                    #     break

                    # add back coherence feedback when there is it.
                    narrative = narrator.generate(prompt, last_narrative=narrative, faithful_feedback=faithful_feedback, coherence_feedback=coherence_feedback) 
                    print(f"📝 Narrative Round {round_num + 1}")
                    round_narratives.append(narrative)

                narratives_all.append(round_narratives)
                feedback_all.append(round_feedback)
                explanation_list.append(prompt_generator.explanation_list[0])
                results_list.append(prompt_generator.result_df)
                idx_list.append(idx)
                feedback_all_coherence.append(round_coherence)
                # Append per-instance round_extractions list into the overall dict
                for llm in self.faithfulevaluator_models:
                    extractions_dict[llm.model].append(round_extractions_per_llm[llm.model])
                extractions_list_voted.append(round_extractions_voted)

            experiment = AgenticNarrativeExperiment(
                dataset=dataset_name,
                prompt_type=prompt_type,
                tar_model_name=tar_model_name,
                idx_list=idx_list,
                explanation_list=explanation_list,
                results_list=results_list,
                narrator_model=narrator_model.model,
                narrative_list=narratives_all,
                extractions_dict=extractions_dict,
                extractions_dict_voted={"voted": extractions_list_voted},
                #add or remove faithful critic model
                faithful_critic_model=faithful_critic.model,
                coherence_model=coherence_agent.model,
                feedback_list=feedback_all,
                feedback_all_coherence=feedback_all_coherence,
                # generation_time=None,
                id=exp_id,
                num_feat=self.num_feat
            )

            per_end_time = time.time()
            total_runtime = per_end_time - per_start_time
            print(f"\n✅ This experiment completed in {total_runtime:.2f} seconds "
                f"({total_runtime/60:.2f} minutes).")
        
            experiments.append(experiment)
            with open(temp_save_path, "wb") as f:
                pickle.dump(experiments, f)
        
        return experiments