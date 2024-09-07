import os
import argparse
from pathlib import Path


# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# default track transformer path
DEFAULT_TRACK_TRANSFORMERS = {
    "libero_spatial": "./results/track_transformer/libero_track_transformer_libero-spatial/",
    "libero_object": "./results/track_transformer/libero_track_transformer_libero-object/",
    "libero_goal": "./results/track_transformer/libero_track_transformer_libero-goal/",
    "libero_10": "./results/track_transformer/libero_track_transformer_libero-100/",
}

# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_goal", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"], 
                    help="The name of the desired suite, where libero_10 is the alias of libero_long.")
parser.add_argument("-tt", "--track-transformer", default=None, help="Then path to the trained track transformer.")
parser.add_argument('--n_seeds', default=1, help="The number of seeds you want to train the policy with.")
args = parser.parse_args()

# training configs
CONFIG_NAME = "libero_vilt"

train_gpu_ids = [0, 1]
NUM_DEMOS = 10

root_dir = Path("./data/atm_libero/")
suite_name = args.suite
task_dir_list = [path for path in (root_dir/suite_name).iterdir() if path.is_dir()]
task_dir_list.sort()

# dataset
train_path_list = [str(task_dir/f"bc_train_{NUM_DEMOS}") for task_dir in task_dir_list]
val_path_list = [str(task_dir/"val") for task_dir in task_dir_list]
track_fn = args.track_transformer or DEFAULT_TRACK_TRANSFORMERS[suite_name]

for seed in range(args.n_seeds):
    commond = (f'python -m engine.train_bc --config-name={CONFIG_NAME} train_gpus="{train_gpu_ids}" '
                f'experiment=atm-policy_{suite_name.replace("_", "-")}_demo{NUM_DEMOS} '
                f'train_dataset="{train_path_list}" val_dataset="{val_path_list}" '
                f'model_cfg.track_cfg.track_fn={track_fn} '
                f'model_cfg.track_cfg.use_zero_track=False '
                f'model_cfg.spatial_transformer_cfg.use_language_token=False '
                f'model_cfg.temporal_transformer_cfg.use_language_token=False '
                f'seed={seed} ')

    os.system(commond)
