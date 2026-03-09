import re
import json

with open("train_colab.txt", "r") as f:
    text = f.read()

# 1. Update load_dataset definition
old_load = """def load_dataset(dataset_root, max_episodes=None):
    \"\"\"Load all state-action pairs from LeRobot parquet files.\"\"\"
    chunk_dir = os.path.join(dataset_root, 'data', 'chunk-000')
    parquet_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith('.parquet'))

    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]

    all_states, all_actions, episode_boundaries = [], [], [0]

    for pf in parquet_files:
        df = pq.read_table(os.path.join(chunk_dir, pf)).to_pandas()

        for _, row in df.iterrows():
            state = np.array(row['observation.state'], dtype=np.float32)
            action = np.array(row['action'], dtype=np.float32)
            all_states.append(state)
            all_actions.append(action)

        episode_boundaries.append(len(all_states))

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)

    print(f"Loaded {len(parquet_files)} episodes, {len(states)} timesteps")
    print(f"State dim:  {states.shape[-1]}")
    print(f"Action dim: {actions.shape[-1]}")
    return states, actions, episode_boundaries"""

new_load = """def load_dataset(dataset_root, max_episodes=None, chunk_size=1):
    \"\"\"Load all state-action pairs from LeRobot parquet files.\"\"\"
    chunk_dir = os.path.join(dataset_root, 'data', 'chunk-000')
    parquet_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith('.parquet'))

    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]

    all_states, all_actions, episode_boundaries = [], [], [0]

    for pf in parquet_files:
        df = pq.read_table(os.path.join(chunk_dir, pf)).to_pandas()

        ep_actions = []
        for _, row in df.iterrows():
            state = np.array(row['observation.state'], dtype=np.float32)
            action = np.array(row['action'], dtype=np.float32)
            all_states.append(state)
            ep_actions.append(action)
            
        if chunk_size > 1 and len(ep_actions) > 0:
            chunked_actions = []
            for i in range(len(ep_actions)):
                chunk = ep_actions[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    pad = [chunk[-1]] * (chunk_size - len(chunk))
                    chunk.extend(pad)
                chunked_actions.append(np.concatenate(chunk))
            all_actions.extend(chunked_actions)
        else:
            all_actions.extend(ep_actions)

        episode_boundaries.append(len(all_states))

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    
    if chunk_size > 1:
        actions = actions.reshape(-1, chunk_size, actions.shape[-1] // chunk_size)

    print(f"Loaded {len(parquet_files)} episodes, {len(states)} timesteps")
    print(f"State dim:  {states.shape[-1]}")
    print(f"Action dim: {actions.shape[-1]}")
    if chunk_size > 1:
        print(f"Chunk size: {chunk_size}")
    return states, actions, episode_boundaries"""

text = text.replace(old_load, new_load)

# 2. Update load_dataset call
text = text.replace("states, actions, ep_bounds = load_dataset(DATASET_ROOT)", "CHUNK_SIZE = 4\nstates, actions, ep_bounds = load_dataset(DATASET_ROOT, chunk_size=CHUNK_SIZE)")

# 3. Update CONFIG definition
old_config = '''CONFIG = {
    "model": "improved",  # "simple" or "improved"
    "hidden_dim": 512,
    "n_blocks": 6,
    "dropout": 0.1,
    "epochs": 200,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "lr_warmup_epochs": 10,
}'''
new_config = '''CONFIG = {
    "model": "improved",  # "simple" or "improved"
    "hidden_dim": 512,
    "n_blocks": 6,
    "dropout": 0.1,
    "epochs": 200,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "lr_warmup_epochs": 10,
    "chunk_size": CHUNK_SIZE,
}'''
text = text.replace(old_config, new_config)

# 4. Update ImprovedPolicy instantiation
old_improved = '''    model = ImprovedPolicy(
        STATE_DIM, ACTION_DIM,
        hidden_dim=CONFIG["hidden_dim"],
        n_blocks=CONFIG["n_blocks"],
        dropout=CONFIG["dropout"],
    ).to(device)'''
new_improved = '''    model = ImprovedPolicy(
        STATE_DIM, ACTION_DIM,
        hidden_dim=CONFIG["hidden_dim"],
        n_blocks=CONFIG["n_blocks"],
        dropout=CONFIG["dropout"],
        chunk_size=CONFIG["chunk_size"],
    ).to(device)'''
text = text.replace(old_improved, new_improved)

with open("train_colab.txt", "w") as f:
    f.write(text)

# Also create the ipynb file
split_markers = [
    "!pip install -q pyarrow",
    "import os\\n\\nDRIVE_ROOT",
    "import json\\nimport pyarrow",
    "import numpy as np\\n\\ndef load_dataset",
    "import matplotlib.pyplot as plt",
    "ep_lengths = ",
    "import torch\\nimport torch.nn as nn",
    "class SimplePolicy",
    "# ── Hyperparameters",
    "def train_one_epoch",
    "CHECKPOINT_DIR =",
    "fig, (ax1, ax2) =",
    "# Load the best checkpoint",
    "from IPython.display import HTML"
]

ipynb_text = text
for marker in split_markers:
    ipynb_text = ipynb_text.replace(marker, "<SPLIT>\\n" + marker)

blocks = [b.strip() for b in ipynb_text.split("<SPLIT>") if b.strip()]

cells = []
for block in blocks:
    source_lines = [line + "\n" for line in block.split("\n")]
    if source_lines:
        source_lines[-1] = source_lines[-1][:-1] # remove last newline
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

notebook = {
 "cells": cells,
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
with open("train_colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
