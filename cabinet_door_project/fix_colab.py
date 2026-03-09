import json

with open("train_colab.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Fix dataset visualization
        if "axes[1].boxplot([actions[:, i] for i in range(actions.shape[1])], showfliers=False)" in source:
            new_source = source.replace(
                "axes[1].boxplot([actions[:, i] for i in range(actions.shape[1])], showfliers=False)",
                "plot_actions = actions[:, 0, :] if actions.ndim > 2 else actions\naxes[1].boxplot([plot_actions[:, i] for i in range(plot_actions.shape[1])], showfliers=False)"
            )
            cell["source"] = [line + ("\n" if not line.endswith("\n") else "") for line in new_source.splitlines()]
            cell["outputs"] = [] # Clear the error output
            
        # Fix evaluation visualization
        if "ax.plot(ep_actions_gt[:, i], label=\"Ground truth\", alpha=0.7)" in source:
            old_plot = """    ax.plot(ep_actions_gt[:, i], label="Ground truth", alpha=0.7)
    ax.plot(ep_actions_pred[:, i], label="Predicted", alpha=0.7, linestyle="--")"""
            
            new_plot = """    if ep_actions_gt.ndim > 2:
        ax.plot(ep_actions_gt[:, 0, i], label="Ground truth", alpha=0.7)
        ax.plot(ep_actions_pred[:, 0, i], label="Predicted", alpha=0.7, linestyle="--")
    else:
        ax.plot(ep_actions_gt[:, i], label="Ground truth", alpha=0.7)
        ax.plot(ep_actions_pred[:, i], label="Predicted", alpha=0.7, linestyle="--")"""
            new_source = source.replace(old_plot, new_plot)
            cell["source"] = [line + ("\n" if not line.endswith("\n") else "") for line in new_source.splitlines(keepends=False)]

with open("train_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
    
