import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["No", "Low", "Medium", "High"]
x = np.arange(len(categories))

qwen3    = [85.7, 83.7, 80.3, 80.7]
ft_qwen3 = [94.0, 92.7, 90.0, 80.7]

qwen25    = [84.0, 82.7, 81.0, 74.7]
ft_qwen25 = [94.0, 93.7, 89.0, 79.3]

gemma    = [73.7, 77.3, 71.0, 73.3]
ft_gemma = [94.3, 90.0, 89.3, 84.3]

llama    = [70.3, 68.3, 66.0, 63.3]
ft_llama = [93.3, 88.3, 85.0, 78.3]


# Color per model family
colors = {
    "qwen3":  "#FF9800",
    "qwen25": "#2196F3",
    "gemma":  "#F44336",
    "llama":  "#4CAF50",
}

base_marker = "o"
ft_marker   = "s"

models = [
    ("Qwen3-VL",   qwen3,  ft_qwen3,  colors["qwen3"]),
    ("Qwen2.5-VL", qwen25, ft_qwen25, colors["qwen25"]),
    ("Gemma 3",    gemma,  ft_gemma,  colors["gemma"]),
    ("Llama 3.2",  llama,  ft_llama,  colors["llama"]),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (name, base, ft, color) in zip(axes, models):
    # Base model
    ax.plot(x, base, color=color, linestyle="-", linewidth=2,
            marker=base_marker, markersize=8, label=f"{name}")
    # Fine-tuned model
    ax.plot(x, ft, color=color, linestyle="--", linewidth=2,
            marker=ft_marker, markersize=8, label=f"FT {name}", alpha=0.85)
    # Shaded gap
    ax.fill_between(x, base, ft, color=color, alpha=0.12)

    ax.set_title(name, fontsize=16, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 10))
    ax.legend(loc="lower left", fontsize=13, framealpha=0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.supxlabel("Occlusion Level", fontsize=16, y=0.02)
fig.supylabel("Accuracy (%)", fontsize=16, x=0.02)
fig.suptitle(
    "Comparison of VLMs — Occluded COCO Dataset\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
    fontsize=18, y=1.01
)

plt.tight_layout()
plt.savefig("./coco_grid_results.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved.")