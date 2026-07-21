import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["No", "Low", "Medium", "High"]
x = np.arange(len(categories))

# Model data for Exp 1
qwen3    = [96.0, 90.6, 85.0, 58.0]
ft_qwen3 = [95.6, 93.0, 85.4, 55.2]

qwen25    = [89.4, 80.0, 56.8, 14.4]
ft_qwen25 = [88.6, 86.8, 69.6, 27.6]

gemma    = [86.2, 79.5, 36.0, 13.1]
ft_gemma = [94.7, 86.2, 68.0, 20.4]

llama    = [91.8, 85.6, 65.6, 28.6]
ft_llama = [94.2, 91.8, 86.6, 56.4]

# Model data for Exp 2
# qwen3 = [98.9, 92.2, 76.6, 50.2]
# ft_qwen3 = [98.9, 98.6, 89.5, 52.0]

# qwen25 =     [97.5, 87.1, 47.0, 22.7]
# ft_qwen25 =  [97.6, 86.0, 55.2, 30.4]

# gemma =    [95.9, 44.5, 24.1, 16.1]
# ft_gemma = [98.3, 71.9, 44.1, 28.4]

# llama =    [88.5, 74.3, 51.4, 29.1]
# ft_llama = [96.5, 82.4, 72.7, 42.7]


# Color per model family
colors = {
    "qwen3":  "#FF9800",
    "qwen25": "#2196F3",
    "gemma":  "#F44336",
    "llama":  "#4CAF50",
}

models = [
    ("Qwen3-VL",    qwen3,  ft_qwen3,  colors["qwen3"]),
    ("Qwen2.5-VL",  qwen25, ft_qwen25, colors["qwen25"]),
    ("Gemma 3",     gemma,  ft_gemma,  colors["gemma"]),
    ("Llama 3.2",   llama,  ft_llama,  colors["llama"]),
]

base_marker = "o"
ft_marker   = "s"

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

# Shared axis labels
fig.supxlabel("Occlusion Level", fontsize=16, y=0.02)
fig.supylabel("Accuracy (%)", fontsize=16, x=0.02)
fig.suptitle(
    "Comparison of VLMs — Experiment 1\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
    fontsize=18, y=1.01
)
# fig.suptitle(
#     "Comparison of VLMs — Experiment 2\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
#     fontsize=18, y=1.01
# )

plt.tight_layout()
plt.savefig("./exp1_grid_results.png", dpi=300, bbox_inches="tight")
# plt.savefig("./exp2_grid_results.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved.")