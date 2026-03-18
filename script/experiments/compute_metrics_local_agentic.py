import dill
from shapnarrative_agents.metrics.faithfulness_agentic_voted import compute_faithfulness_agentic
import argparse


"""
IN THIS SCRIPT, WE COMPUTE THE FATIFHULNESS OF THE GIVEN PICKLE FILES FROM THE EXPERIMENTS FROM THE AGENTIC SYSTEMS. 
"""

# N_range=2
# experiment_dir="standard_experiments_agentic_critic_no_llm"
# experiment_dir="old_results"

# EXPERIMENT_PATHS=[f"results/{experiment_dir}/experiment_{i}/experiment_worstbase.pkl" for i in range(1,N_range)]
# SAVE_PATHS=[f"results/{experiment_dir}/experiment_{i}/metrics_worstbase.pkl" for i in range(1,N_range)]

EXPERIMENT_PATHS=["final_test.pkl"]
SAVE_PATHS=["final_test.pkl" ]
 
if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Generate a series of narratives and save them")
    parser.add_argument("--EXPERIMENT_PATHS", '--experiment_paths_list' , nargs='+', default=EXPERIMENT_PATHS, type=list, help=f"Path to dir to read experiments from")
    parser.add_argument("--SAVE_PATHS", '--save_paths_list' , nargs='+', default=SAVE_PATHS, type=list, help=f"Path to dir where the metrics will be saved")

    args=parser.parse_args()

    for experiment_path, save_path in zip(args.EXPERIMENT_PATHS, args.SAVE_PATHS):

        with open(experiment_path, "rb") as f:
            experiments=dill.load(f)

        metrics=compute_faithfulness_agentic(experiments)

        with open(save_path, "wb") as f:
            dill.dump(metrics, f)
    