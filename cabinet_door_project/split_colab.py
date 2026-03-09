import json

with open("train_colab.txt", "r") as f:
    text = f.read()

# Split the text into cells based on logical blocks
cells = []
current_cell = []

lines = text.split("\n")
for i, line in enumerate(lines):
    current_cell.append(line)
    
    # Define breakpoint conditions for new cells
    is_breakpoint = False
    
    # Break after major imports/setup
    if line == "else:":
        if i + 1 < len(lines) and "print(\"WARNING: No GPU detected! Go to Runtime > Change runtime type > GPU\")" in lines[i+1]:
            # skip this one, it's the GPU check
            pass
            
    if line == "drive.mount('/content/drive')":
        is_breakpoint = True
        
    if line == "!pip install -q pyarrow matplotlib":
        is_breakpoint = True
        
    if "print(f\"Dataset root: {DATASET_ROOT}\")" in line and "else:" in lines[i-(min(15, i)) : i]:
        # we don't want to break inside the if/else block, break after the whole thing
        pass
        
    if line == "        print(f\"  {item}/\" if os.path.isdir(full) else f\"  {item}\")":
        is_breakpoint = True
        
    if line == "        else:":
        if i + 1 < len(lines) and "            print(f\"  {c}: type={type(val).__name__}, sample={val}\")" in lines[i+1]:
            pass
            
    if line == "            print(f\"  {c}: type={type(val).__name__}, sample={val}\")":
        is_breakpoint = True
        
    if line == "states, actions, ep_bounds = load_dataset(DATASET_ROOT, chunk_size=CHUNK_SIZE)":
        is_breakpoint = True
        
    if line == "plt.show()":
        # Don't break if it's the last line
        if i < len(lines) - 1:
           is_breakpoint = True
           
    if line == "print(f\"Train: {len(train_ds)}, Val: {len(val_ds)}\")":
        is_breakpoint = True
        
    if line == "                        num_workers=2, pin_memory=True)":
        is_breakpoint = True
        
    if line == "        return out":
        if i + 1 < len(lines) and "# ── Hyperparameters" in lines[i+1]:
             is_breakpoint = True
             
    if line == "scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)":
         is_breakpoint = True
         
    if line == "    return total_loss / max(n_batches, 1)":
         is_breakpoint = True
         
    if line == "print(f\"Checkpoints saved to: {CHECKPOINT_DIR}\")":
         is_breakpoint = True
         
    if line == "print(f\"Saved training_curves.png to {CHECKPOINT_DIR}\")":
         is_breakpoint = True
         
    if line == "model.eval()":
         # Check if it's the specific eval after loading
         if i >= 2 and "model.load_state_dict" in lines[i-1]:
             is_breakpoint = True
             
    if line == "    print(\"This is fine — training only uses the parquet state-action data.\")":
         is_breakpoint = True
    
    if is_breakpoint:
        # Save current cell and start a new one
        if any(c.strip() for c in current_cell):
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [c + "\n" for c in current_cell]
            })
        current_cell = []

# Add the last cell if it has content
if any(c.strip() for c in current_cell):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [c + "\n" for c in current_cell]
    })

# Remove the trailing newline from the last line of each cell
for cell in cells:
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

notebook = {
 "cells": cells,
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("train_colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
