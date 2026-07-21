import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["No", "Low", "Medium", "High"]
x = np.arange(len(categories))

# Model data for Exp 1
qwen3 = [96.7, 90.7, 86.7, 58.7]
ft_qwen3 = [96.7, 95.3, 86.7, 54.0]

qwen25 =     [89.3, 80.0, 56.7, 14.7]
ft_qwen25 =  [88.7, 88.0, 69.3, 28.0]

gemma =    [86.7, 79.3, 36.0, 13.3]
ft_gemma = [94.0, 86.0, 68.0, 22.0]

llama =    [92.7, 84.0, 64.0, 30.7]
ft_llama = [93.3, 91.3, 86.7, 55.3]

# Model data for Exp 2
# qwen3 = [98.8, 90.2, 77.3, 49.2]
# ft_qwen3 = [99.2, 98.7, 89.4, 50.0]

# qwen25 =     [97.5, 86.9, 46.2, 22.7]
# ft_qwen25 =  [97.5, 85.6, 56.1, 29.5]

# gemma =    [95.1, 45.8, 23.5, 16.7]
# ft_gemma = [97.5, 69.9, 46.9, 28.8]

# llama =    [87.7, 74.5, 53.0, 28.0]
# ft_llama = [97.1, 84.3, 75.0, 40.2]

# Color per model family
colors = {
    "qwen3": "#FF9800",   # Orange
    "qwen25":  "#2196F3",   # Blue
    "gemma": "#F44336",   # Red
    "llama": "#4CAF50",   # Green
}

# Marker per base vs FT
base_marker = "o"
ft_marker   = "s"

fig, ax = plt.subplots(figsize=(9, 5.5))

def plot_model(ax, x, base, ft, color, name):
    # Base model — solid line, circle markers
    ax.plot(x, base, color=color, linestyle="-",  linewidth=2,
            marker=base_marker, markersize=8, label=name)
    # FT model — dashed line, square markers, slightly transparent fill
    ax.plot(x, ft,   color=color, linestyle="--", linewidth=2,
            marker=ft_marker,   markersize=8, label=f"FT {name}",
            alpha=0.85)
    # Shade the gap between base and FT
    ax.fill_between(x, base, ft, color=color, alpha=0.08)
    # Annotate each FT point with the delta vs base
    # for xi, b, f in zip(x, base, ft):
    #     delta = f - b
    #     sign  = "+" if delta >= 0 else ""
    #     ax.annotate(f"{sign}{delta:.0f}",
    #                 xy=(xi, max(b, f)),
    #                 xytext=(0, 6), textcoords="offset points",
    #                 ha="center", va="bottom",
    #                 fontsize=7.5, color=color, fontweight="bold")

plot_model(ax, x, qwen3, ft_qwen3, colors["qwen3"], "Qwen3-VL")
# plot_model(ax, x, qwen25,  ft_qwen25,  colors["qwen25"],  "Qwen2.5-VL")
# plot_model(ax, x, gemma, ft_gemma, colors["gemma"], "Gemma 3")
# plot_model(ax, x, llama, ft_llama, colors["llama"], "Llama 3.2")

# Formatting
ax.set_xlabel("Occlusion Level", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 110)
ax.set_yticks(range(0, 101, 10))
# Experiment 1 title
# ax.set_title("Comparison of VLMs — Experiment 1\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
            #  fontsize=12, pad=14)
# Experiment 2 title
# ax.set_title("Comparison of VLMs — Experiment 2\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
#              fontsize=12, pad=14)
# Individual models title
ax.set_title("Qwen3-VL\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
              fontsize=12, pad=14)
# ax.set_title("Comparison of Qwen2.5-VL on Experiment 1\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
#               fontsize=12, pad=14)
# ax.set_title("Comparison of Gemma 3 on Experiment 1\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
#               fontsize=12, pad=14)
# ax.set_title("Comparison of Llama 3.2 on Experiment 1\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
#               fontsize=12, pad=14)

# Legend — two-column, neatly placed
ax.legend(loc="lower left", ncol=2, fontsize=9.5, framealpha=0.85)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
# plt.savefig("./exp1_comparative_results.png", dpi=300, bbox_inches="tight")
# plt.savefig("./exp2_comparative_results.png", dpi=300, bbox_inches="tight")
plt.savefig("./exp1_Qwen3-VL_comparative_results.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved.")