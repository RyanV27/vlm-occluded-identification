import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["Low", "Medium", "High"]
x = np.arange(len(categories))

# Model data for COCO Exp
# qwen3 = [84.0, 78.0, 82.0]
# ft_qwen3 = [92.0, 89.0, 82.0]

# qwen25 =     [82.0, 81.0, 72.0]
# ft_qwen25 =  [94.0, 89.0, 80.0]

# gemma =    [78.0, 71.0, 70.0]
# ft_gemma = [90.0, 90.0, 87.0]

# llama =    [67.0, 68.0, 55.0]
# ft_llama = [90.0, 85.0, 79.0]

qwen3 = [83.0, 80.0, 81.0]
ft_qwen3 = [93.0, 89.5, 81.0]

qwen25 =     [82.5, 81.0, 74.0]
ft_qwen25 =  [93.5, 89.0, 79.5]

gemma =    [77.5, 71.0, 71.5]
ft_gemma = [90.0, 91.0, 85.5]

llama =    [68.3, 66.0, 63.3]
ft_llama = [89.0, 85.5, 79.0]

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
plot_model(ax, x, qwen25,  ft_qwen25,  colors["qwen25"],  "Qwen2.5-VL")
plot_model(ax, x, gemma, ft_gemma, colors["gemma"], "Gemma 3")
plot_model(ax, x, llama, ft_llama, colors["llama"], "Llama 3.2")

# Formatting
ax.set_xlabel("Occlusion Level", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 110)
ax.set_yticks(range(0, 101, 10))
# Experiment title
ax.set_title("Comparison of VLMs — Occluded COCO Dataset\n(Solid = Pre-trained Model, Dashed = Fine-tuned Model)",
             fontsize=12, pad=14)

# Legend — two-column, neatly placed
ax.legend(loc="lower right", ncol=2, fontsize=9.5, framealpha=0.85)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("./coco_comparative_results.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved.")