import ast,re
from typing import Type, Tuple, Optional
import pandas as pd
import shap
import numpy as np
import time
from shapnarrative_metrics.llm_tools.llm_wrappers import LLMWrapper
from script.logger_utils import log_with_context

class FaithfulEvaluator:
    """
    This class defines a faithfulevaluator that extracts information from a narrative and feedback with what kind of faithful-errors.

    Attributes:
    -----------
    ds_info: dictionary
        Dictionary that contains short generic information about the dataset  
        Format: {
                 "dataset description": dataset_description (str), 
                 "target_description": target_description (str), 
                 "task_description": task_description (str), 
                 "feature_desc": pd.DataFrame that contains columns "feature_name", "feature_average", "feature_desc".
                 }
    llm : str, default="None"
        LLM API wrapper to do the extractions.
    """

    def __init__(
        self,
        ds_info: dict,
        llm: Type[LLMWrapper] = None,
    ):

        self.ds_info=ds_info
        self.feature_desc = ds_info["feature_description"]
        self.llm = llm

    def generate_prompt(self, narrative: str):

        """
        Prepared a extraction prompt based on the input narrative for the faithful llm to rextract the information from narrative.

        Parameters:
        -----------
        narrative : str
            A SHAP narrative that was generated to explain the prediction of a particular instance. 

        Returns:
        --------
        prompt_string: str
            A prompt for the extractor model 
        """

        prompt_string = f"""
        ***Context***:
        An LLM was used to create a narrative to explain and interpret a prediction made by another smaller classifier model. 
        The LLM was given an explanation of the classifier task, the training data, and provided with the exact names of all the features and their meaning. 
        Most importantly, the LLM was provided with a table that contains the feature values of that particular instance, the average feature values and their SHAP values which are a numeric measure of their importance. 
        
        You are an helpful agent tasked with improving the narrative. 
        To do so, you should extract some information about all the features that were mentioned in the narrative that will be given below.

        Here is some general info for you:
        Dataset description: {self.ds_info["dataset_description"]}.
        Target description: {self.ds_info["target_description"]}.
        Task description: {self.ds_info["task_description"]}.
        Feature descriptions: {self.feature_desc[["feature_name","feature_desc"]].to_string(index = False)}.
        
        ***Input text***: 
        The following is the given narrative.
        ====================
        \"\"\"{narrative}\"\"\"
        ====================

        ***Output Structure***:
        Provide your answer as a python dictionary with the keys as the feature names. 
        The values corresponding to the feature name keys are dictionaries themselves that contain the following inner keys: 
        1) "rank:" indicating the order of absolute importance of the feature starting from 0.
        2) "sign": the sign of whether the feature contributed towards target value 1 or against it (either +1 or -1 for sign value).
        3) "value": if the value of the feature is mentioned in a way that you can put an exact number on, add it. Only return numeric values here.
        If the description of the value is qualitative such as "many" or "often" and not mentioning an exact value, return "None" for its value.
        4) "assumption": give a short but complete 1 sentence summary of what the assumption is in the story for this feature.
        Provide this assumption as a general statement that could be fact checked later and that does not require this narrative as context.
        If no reason or suggestion is made in the story do not make something up and just return string 'None'.
        
        ***Guidelines***:
        1) Make sure that both the "rank", "sign", "value" and "assumption" keys and their values are always present in the inner dictionaries.
        2) Make sure that the "rank" key is sorted from 0 to an increasing value in the dictionary. The first element cannot have any other rank than 0.
        3) Make sure to use the exact names of the features as provided in the Feature descriptions, including capitalization. 
        4) Just provide the python dictionary as a string and add nothing else to the answer.
        """

        return prompt_string
    

    def extract_dict_from_str(self, extracted_str: str)->dict:
        """
        Extracts a dictionary from a string 
        Purpose: Takes the raw string output from the LLM (which is a dictionary printed as a string) and converts it into a real Python dictionary
        Automatically fixes common malformed outputs, clean the format, and retries safely.
        
        extracted_str : str
            The answer to the extractor prompt, usually a simple dictionary in string form but could be preceded by some sentences.

        Returns: extracted_dict: dict
            A prompt for the extractor model 
        """
        def log_with_lines(s: str, header: str = "LLM RAW") -> None:
            print(f"\n===== {header} =====")
            for i, line in enumerate(s.splitlines(), 1):
                print(f"{i:>3}: {line}")
            print("=" * 30)

        # ✅ Step 1: Early blank/null check (protects against None or "   \n")
        if not extracted_str or not extracted_str.strip():
            raise ValueError("LLM output is empty or blank.")

        # # ✅ Step 2: Strip once, remove code fences
        extracted_str = extracted_str.strip()

        start_index = extracted_str.find("{")
        end_index = extracted_str.rfind("}")
        if start_index == -1 or end_index == -1:
            log_with_lines(extracted_str, header="NO DICT FOUND")
            # raise ValueError("No dictionary-like structure found in LLM output.")
            return None
        
        dict_str = extracted_str[start_index:end_index + 1].strip()
 
        def sanitize(s: str) -> str:
            # 1) Remove trailing commas before closing braces/brackets
            s = re.sub(r",\s*([}\]])", r"\1", s)
            # 2) Replace missing values like '"value": ,' or '"rank": }' with None
            s = re.sub(r':\s*(?=[,}])', ': None', s)
            # 3) Replace JSON null/true/false with Python equivalents
            s = re.sub(r'(?<!")\bnull\b', 'None', s)
            s = re.sub(r'(?<!")\btrue\b', 'True', s)
            s = re.sub(r'(?<!")\bfalse\b', 'False', s)
            # 4) Balance curly braces
            opens, closes = s.count("{"), s.count("}")
            if opens > closes:
                s += "}" * (opens - closes)
            elif closes > opens:
                s = s.rstrip("}")  # remove extras at the end
                while s.count("}") > s.count("{"):
                    s = s[:-1]
            # 5) Ensure commas between key-value pairs are valid
            s = re.sub(r'(\})\s*(\w)', r'\1, \2', s)
            # 6) Remove stray leading/trailing quotes or text fragments
            s = s.strip(' \n\t;')
            # 7) Fix incomplete or missing quotes around known keys (rank, sign, value, assumption, etc.)
            # Case a: missing opening quote  → rank"  → "rank"
            s = re.sub(r'(?<!")\b(rank|sign|value|assumption)\b"', r'"\1"', s)
            # Case b: missing closing quote  → "rank  → "rank"
            s = re.sub(r'"\b(rank|sign|value|assumption)\b(?!")', r'"\1"', s)
            # Case c: no quotes at all before colon  → rank:  → "rank":
            s = re.sub(r'\b(rank|sign|value|assumption)\b(?=\s*:)', r'"\1"', s)
            return s
        
        # Safe access for logging
        model_name = getattr(getattr(self, "narrator_model", None), "model", "unknown")
        dataset_name = self.ds_info.get("dataset_name", "")
        instance_idx = getattr(self, "current_instance", None)
        
        cleaned = sanitize(dict_str)
        try:
            extracted_dict = ast.literal_eval(cleaned)
            return extracted_dict
        except (SyntaxError, ValueError) as e:
            log_with_lines(cleaned, header="UNPARSEABLE")
            final_msg = f"❌ Failed to parse LLM dictionary ({type(e).__name__}: {e}). Returning empty dict."
            print(final_msg)
            log_with_context("error", final_msg, model=model_name, dataset=dataset_name, instance=instance_idx)
            log_with_context("error", "Malformed dictionary content:\n" + cleaned,
                            model=model_name, dataset=dataset_name, instance=instance_idx)
            return {}

    def generate_extractions(self, narratives: list[str]| str):

        """
        Generate an extraction dictionary from the narrative.
        Automatically retry when LLMs fails the task for the first three tries so that not to interript the script from running.

        Args:
            narratives: (list of strings) all the narratives that we want to extract the features of

        Returns:
            extractions (list of dicts): A list containing the extracted feature dicts for each narrative
        """
        if isinstance(narratives, str):
            narratives = [narratives]

        def contains_missing_important(d: dict) -> bool:
            for feature, vals in d.items():
                if isinstance(vals, dict):
                    if vals.get("rank") is None:
                        return True
                    if vals.get("sign") is None:
                        return True
            return False

        extractions=[]
        for i, narrative in enumerate(narratives):
            extracted_dict = {}
            for attempt in range(3):
                response = self.llm.generate_response(self.generate_prompt(narrative))
                extracted_dict = self.extract_dict_from_str(response)

                if extracted_dict is None:
                    print(f"⚠️ Attempt {attempt+1}: No dictionary found. Retrying...")
                    continue

                # ✅ Solve rare case when one LLM extracted None for rank values.
                if not isinstance(extracted_dict, dict):
                    print(f"⚠️ Attempt {attempt+1}: Extraction not a dict. Retrying...")
                    continue

                if contains_missing_important(extracted_dict):
                    print(f"⚠️ Attempt {attempt+1}: Detected None in rank/sign fields. Retrying...")
                    continue

                break  

            else:
                # raise TypeError("Failed to extract valid dictionary from LLM response after 3 attempts.")
                print("⚠️ All retries had missing rank/sign. Proceeding with last extraction.")

            extractions.append(extracted_dict)
            print(f"✅ Extracted story {i + 1}/{len(narratives)} with {self.llm.model}")

        return extractions 
 
    @staticmethod
    def get_diff(extracted_dict: dict, explanation: pd.DataFrame):

        """
        Compares the extracted dict with the actual explanation (shap_table, which is provided outside and before the agent loop) and calculates their difference: 
        rank_diff, sign_diff and value_diff.

        Parameters:
        -----------
        extracted_dict : dict
            The dictionary extracted from the LLM answer.
        explanation: pd.DataFrame
            A dataframe containing a column with the SHAP values and feature values. AKA, the shap_df that was already prepared in the preliminary steps (from generator.explanation_list[0].head(4): GenerationModel).

        Returns: Tuple[5x list] 
        --------
        """

        ###STEP1: WE COMPUTE DIFFERENCE FOR ALL EXTRACTED FEATURES THAT ACTUALLY EXIST:
        #1)make sure the explanation is sorted by SHAP values (this should be already the case if generated with SHAPstory):
        explanation["abs_SHAP"] = explanation["SHAP_value"].abs()
        explanation = explanation.sort_values(by="abs_SHAP", ascending=False)
        explanation.drop(columns=["abs_SHAP"])

        #2) create a dataframe out of the extracted dict
        df_extracted=pd.DataFrame(extracted_dict).T
        df_extracted.reset_index(inplace=True)
        df_extracted.rename(columns={"index":"feature_name"},inplace=True)

        #3) filter the real explanation on the features that were present in the extraction dict 
        cat_dtype = pd.CategoricalDtype(df_extracted["feature_name"], ordered=True)
        explanation['feature_name']=explanation['feature_name'].astype(cat_dtype)
        df_real = explanation[explanation.feature_name.isin(df_extracted["feature_name"])].sort_values(by="feature_name")
        
        #4) get a list of feature names that have been extracted but do not exist (usually doesn't happen but good check)
        incorrect_features = df_extracted[~df_extracted['feature_name'].isin(df_real['feature_name'])]['feature_name'] 
        
        #5) now that we have a separate list of the hallucinated features, continue only with the overlap of existing features
        df_extracted=df_extracted[df_extracted['feature_name'].isin(df_real['feature_name'])] 
        sign_series=df_real["SHAP_value"].map(lambda x: int(np.sign(x)))
        df_real.insert(1,"sign",sign_series)
        df_real.insert(1,"rank", df_real.index)
        df_real=df_real.drop(columns=["SHAP_value","feature_desc"])
        
        #6) for all the real features replace any non-numeric extracted element with np.nan
        rank_array=np.array([np.nan if type(x) not in [np.float64, np.int64,np.float32, np.int32, int] else x for x in df_extracted["rank"].to_numpy()])
        sign_array=np.array([np.nan if type(x) not in [np.float64, np.int64,np.float32, np.int32, int] else x for x in df_extracted["sign"].to_numpy()])
        value_array=np.array([np.nan if type(x) not in [np.float64, np.int64,np.float32, np.int32, int] else x for x in df_extracted["value"].to_numpy()])

        #7) compute the difference arrays that we intend to output
        rank_diff=(rank_array-df_real["rank"].to_numpy()).astype(float)
        sign_diff=(sign_array*df_real["sign"].to_numpy()<=0).astype(float) #sign_diff will always be 0 or 1 even there is nan in sign_array
        value_diff=(value_array-df_real["feature_value"].to_numpy()).astype(float)

        #also useful to get actual real rank and extracted rank lists
        real_rank=df_real["rank"].to_numpy().astype(int)
        extracted_rank = df_extracted["rank"].astype("Int64").to_numpy()


        ###STEP 2: Now account for the fact that we ignored hallucinated features previously, and add a np.inf for the difference there.  
        for idx in sorted(incorrect_features.index.sort_values()):

            print("""*** Warning: Some features extracted by model were not in the real feature list ***.
            If this warning is encountered too often this could be a sign that something is wrong.""")

            
            if idx >= len(rank_diff):
                # Insert at the last position
                rank_diff = np.append(rank_diff, np.inf)
            else:
                # Insert at the specified index
                rank_diff = np.insert(rank_diff, idx, np.inf)

            if idx>=len(sign_diff):
                sign_diff = np.append(sign_diff, np.inf)
            else:
                sign_diff = np.insert(sign_diff, idx, np.inf)
        
            if idx>=len(value_diff):
                value_diff = np.append(value_diff, np.inf)
            else:
                value_diff = np.insert(value_diff, idx, np.inf)
        
        ###STEP 3:  Ignore very small feature value differences (< 1.0) ---
        # A narrative sometimes mentions a feature value only in a integer way, while the real feature value is a float with decimals, thus creating a small but non-zero value difference.
        # We clean and ignore this minor value difference.
        value_diff_clean = []

        for diff in value_diff:
            if np.isfinite(diff) and abs(diff) < 1:
                value_diff_clean.append(0.0)
            else:
                value_diff_clean.append(diff)

        value_diff = np.array(value_diff_clean)

        return rank_diff.tolist() , sign_diff.tolist(), value_diff.tolist(), real_rank.tolist(), extracted_rank.tolist()

   
    @staticmethod
    def give_feedback(df_diff: pd.DataFrame, extraction: dict) -> str:
        """
        Checks each row of df for only finite values that are not equal to 0 (treated as an error)
        Note the extraction dict as an input, however only used for extracting feature names.
        If no issues found, returns a '100.0% faithful' message.
        """
        # Build a mapping from row index (rank) to feature name
        rank_to_feature = {}
        for feat_name, feat_info in extraction.items():
            rank = feat_info.get('rank')
            if rank is not None:
                rank_to_feature[rank] = feat_name

        messages = []
        found_issues = False

        for idx, row in df_diff.iterrows():
            feature_name = rank_to_feature.get(idx, f"Unknown_{idx}")
            
            cols_with_otherthan_zero = []
            
            for col, val in row.items():
                if pd.notna(val) and val != 0: # ignore NaN, only count in finite values
                    cols_with_otherthan_zero.append(col)

            if cols_with_otherthan_zero:
                found_issues = True
                messages.append(f"Feature {feature_name} contains (an) errors in {cols_with_otherthan_zero} value.")
            
        if not found_issues:
            return "After checking, the narrative is 100% faithful to the SHAP table."
        
        return "\n".join(messages)
 