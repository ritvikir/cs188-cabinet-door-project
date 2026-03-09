import pathlib
import json

def create_cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "execution_count": None if cell_type == "code" else None,
        "outputs": [] if cell_type == "code" else None,
        "source": [line + "\n" for line in source.split("\n")]
    }

def main():
    cells = []

    # Title & Info
    cells.append(create_cell("# OpenCabinet Robust Diffusion Policy Training\nRun this notebook on a Google Colab instance with an A100 GPU for optimal performance.", "markdown"))
    
    # 0. GPU Check
    gpu_code = """# Verify GPU is available
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected! Go to Runtime > Change runtime type > GPU")
"""
    cells.append(create_cell(gpu_code))

    # 1. Mount Drive
    mount_code = """from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/.shortcut-targets-by-id/1o9yPiQvIuOgStlQNEHu1OPd6qRrwGMcJ/cs188_data
"""
    cells.append(create_cell(mount_code))

    # 2. Environment Setup
    setup_code = """# 2. Clone repository and install dependencies
import os
if not os.path.exists('robocasa'):
    !git clone https://github.com/robocasa-benchmark/robocasa.git
%cd robocasa
!pip install -e .
%cd ..

if not os.path.exists('diffusion_policy'):
    !git clone https://github.com/robocasa-benchmark/diffusion_policy.git
%cd diffusion_policy
!pip install -e .
%cd ..

# Install rendering dependencies
!sudo apt-get install -y libosmesa6-dev
"""
    cells.append(create_cell(setup_code))

    # 3. Dataset Download
    download_code = """# 3. Download the OpenCabinet Dataset
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

%cd robocasa
!python scripts/download_datasets.py --tasks OpenCabinet
%cd ..
"""
    cells.append(create_cell(download_code))
    
    # 4. Write Custom Configuration
    config_code = """# 4. Create Custom OpenCabinet.yaml Configuration
import os
import yaml

config_dir = "diffusion_policy/diffusion_policy/config/task/robocasa"
os.makedirs(config_dir, exist_ok=True)
config_path = os.path.join(config_dir, "OpenCabinet.yaml")

# Dataset lookup script
import robocasa
from robocasa.utils.dataset_registry_utils import get_ds_path
ds_path = get_ds_path("OpenCabinet", source="human")
# By default, robocasa download script downloads to ~/.robocasa or inside robocasa/datasets
# get_ds_path should resolve it correctly relative to your current python environment.


config_data = {
    "name": "OpenCabinet",
    "shape_meta": {
        "obs": {
            "robot0_agentview_right_image": {
                "shape": [3, 256, 256],
                "type": "rgb",
                "lerobot_keys": ['video.robot0_agentview_right']
            },
            "robot0_agentview_left_image": {
                "shape": [3, 256, 256],
                "type": "rgb",
                "lerobot_keys": ['video.robot0_agentview_left']
            },
            "robot0_eye_in_hand_image": {
                "shape": [3, 256, 256],
                "type": "rgb",
                "lerobot_keys": ['video.robot0_eye_in_hand']
            },
            "robot0_base_to_eef_pos": {
                "shape": [3],
                "lerobot_keys": ['state.end_effector_position_relative']
            },
            "robot0_base_to_eef_quat": {
                "shape": [4],
                "lerobot_keys": ['state.end_effector_rotation_relative']
            },
            "robot0_gripper_qpos": {
                "shape": [2],
                "lerobot_keys": ['state.gripper_qpos']
            },
            "lang_emb": {
                "shape": [768]
            }
        },
        "action": {
            "shape": [12],
            "lerobot_keys": ['action.end_effector_position', 'action.end_effector_rotation', 'action.gripper_close', 'action.base_motion', 'action.control_mode']
        }
    },
    "abs_action": False,
    "env_runner": {
        "_target_": "diffusion_policy.env_runner.robomimic_image_runner.RobomimicImageRunner",
        "dataset_path": ds_path,
        "n_train": 0,
        "n_train_vis": 1,
        "train_start_idx": 0,
        "n_test": 50,
        "n_test_vis": 4,
        "test_start_seed": 100000,
        "max_steps": 600,
        "n_obs_steps": "${n_obs_steps}",
        "n_action_steps": "${n_action_steps}",
        "render_obs_key": "robot0_agentview_right_image",
        "fps": 10,
        "crf": 22,
        "past_action": "${past_action_visible}",
        "tqdm_interval_sec": 1.0,
        "n_envs": 1,
        "env_kwargs": {
            "seed": 1111111,
            "obj_instance_split": "test",
            "style_ids": None,
            "layout_ids": None,
            "layout_and_style_ids": [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
            "clutter_mode": 1,
            "use_camera_obs": True,
            "use_object_obs": True,
            "camera_depths": False,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "camera_names": ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'],
            "camera_widths": 256,
            "camera_heights": 256,
            "ignore_done": True,
            "reward_shaping": False
        }
    },
    "dataset": {
        "_target_": "diffusion_policy.dataset.lerobot_dataset.LerobotDataset",
        "dataset_path": ds_path,
        "horizon": "${horizon}",
        "pad_before": "${eval:'${n_obs_steps}-1+${n_latency_steps}'}",
        "pad_after": "${eval:'${n_action_steps}-1'}",
        "n_obs_steps": "${dataset_obs_steps}",
        "rotation_rep": "rotation_6d",
        "use_legacy_normalizer": False,
        "use_cache": True,
        "seed": 42,
        "val_ratio": 0.02
    }
}

# Link shape references that PyYAML doesn't natively alias perfectly out-of-the-box
config_data["env_runner"]["shape_meta"] = config_data["shape_meta"]
config_data["env_runner"]["abs_action"] = config_data["abs_action"]
config_data["dataset"]["shape_meta"] = config_data["shape_meta"]
config_data["dataset"]["abs_action"] = config_data["abs_action"]

with open(config_path, "w") as f:
    yaml.dump(config_data, f, default_flow_style=False)
    
print(f"Created config: {config_path}")
"""
    cells.append(create_cell(config_code))
    
    # 5. Train execution
    train_code = """# 5. Train Diffusion Policy
# Note: Ensure you are running on an A100 for acceptable speed.
# Training will automatically log to wandb. Use `wandb login` prior to this cell if desired.
%cd diffusion_policy

# To run a quick test iteration replace the epochs and steps with small values
!python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/OpenCabinet
%cd ..
"""
    cells.append(create_cell(train_code))

    notebook = {
        "metadata": {
            "colab": {
                "name": "train_diffusion_colab.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0,
        "cells": [c for c in cells if c["outputs"] is not None or c["cell_type"] == "markdown"]
    }
    
    # re-add code output key since list comprehension strips it based on our cell maker, 
    # but we DO want empty [] output lists inside the json for valid notebook parsing
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []

    out_path = pathlib.Path(__file__).parent / "train_diffusion_colab.ipynb"
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=2)

    print(f"Created {out_path.name}")

if __name__ == "__main__":
    main()
