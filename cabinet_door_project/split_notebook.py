import json

with open('train_colab.ipynb', 'r') as f:
    nb = json.load(f)

# Assuming all code is in the first cell
code = nb['cells'][0]['source']
if isinstance(code, list):
    code = "".join(code)

# Split points (substrings that start a new cell)
split_markers = [
    "from google.colab import drive",
    "import os\n\nDRIVE_ROOT",
    "import pyarrow.parquet",
    "def load_dataset(",
    "import matplotlib.pyplot as plt",
    "class CabinetDemoDataset",
    "class SimplePolicy(nn.Module):",
    "# ── Hyperparameters",
    "def train_one_epoch(",
    "CHECKPOINT_DIR =",
    "fig, (ax1, ax2)",
    "# Pick a random episode",
    "from IPython.display import HTML"
]

cells_code = []
current_cell = []
lines = code.split('\n')

i = 0
while i < len(lines):
    line = lines[i]
    
    # check if line starts a new cell
    is_marker = False
    for marker in split_markers:
        # Note: marker might span multiple lines, but our markers are single lines or start differently
        # Actually some markers have \n. Let's just check if the line starts with marker text (up to \n)
        marker_first_line = marker.split('\n')[0]
        if line.startswith(marker_first_line):
            is_marker = True
            break
            
    if is_marker and len(current_cell) > 0:
        cells_code.append("\n".join(current_cell))
        current_cell = []
        
    current_cell.append(line)
    i += 1

if current_cell:
    cells_code.append("\n".join(current_cell))

new_cells = []
for c in cells_code:
    if c.strip():
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in c.split('\n')][:-1] + [c.split('\n')[-1]] # keep newline formatting
        })

nb['cells'] = new_cells

with open('train_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Split notebook into {len(new_cells)} cells.")
