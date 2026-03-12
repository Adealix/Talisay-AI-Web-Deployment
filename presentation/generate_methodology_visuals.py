"""
Talisay AI – Defense Presentation: Methodology Section Visuals
==============================================================
Generates two publication-quality figures:
  1. Transfer Learning Architecture (B)
  2. Testing Model / Prediction Pipeline (C)

Output PNGs are saved in the same folder as this script.

Requirements:
    pip install matplotlib
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, text, text2="", bg="#2E86AB", fg="white",
             font_size=9, radius=0.02, border="#FFFFFF", lw=1.5,
             multi_line_gap=0.012):
    """Draw a rounded rectangle with centred text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        linewidth=lw, edgecolor=border, facecolor=bg, zorder=3
    )
    ax.add_patch(box)
    if text2:
        ax.text(x, y + multi_line_gap, text, ha="center", va="center",
                fontsize=font_size, color=fg, fontweight="bold", zorder=4)
        ax.text(x, y - multi_line_gap * 1.6, text2, ha="center", va="center",
                fontsize=font_size - 1.2, color=fg, style="italic", zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=font_size, color=fg, fontweight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.5, head=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f"-|>,head_width=0.18,head_length=0.14",
                                color=color, lw=lw),
                zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B – TRANSFER LEARNING ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def fig_transfer_learning():
    """
    Shows the two-model transfer learning setup:
      • MobileNetV2  → Color Classifier (green / yellow / brown)
      • EfficientNetB0 → TalisayGuard  (talisay vs. non-talisay)
    Both share the same ImageNet pre-trained weights pipeline.
    """
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Palette ─────────────────────────────────────────────────────────────
    C_IMAGENET  = "#F4A261"   # orange  – shared source
    C_PRETRAIN  = "#E76F51"   # dark-orange – freeze layers
    C_FINETUNE1 = "#2A9D8F"   # teal – MobileNetV2 fine-tune head
    C_FINETUNE2 = "#264653"   # dark – EfficientNetB0 fine-tune head
    C_OUTPUT    = "#E9C46A"   # yellow – outputs
    C_LABEL     = "#FFFFFF"
    C_BG        = "#1A1A2E"

    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.95, "Transfer Learning Architecture",
            ha="center", va="center", fontsize=16, color="white",
            fontweight="bold")
    ax.text(0.5, 0.91, "Pre-trained ImageNet weights → Fine-tuned for Talisay Fruit Analysis",
            ha="center", va="center", fontsize=10, color="#AAAAAA", style="italic")

    # ── ImageNet source (top-centre) ─────────────────────────────────────────
    draw_box(ax, 0.50, 0.82, 0.34, 0.065,
             "ImageNet Pre-trained Weights",
             "1.2 M images · 1000 classes",
             bg=C_IMAGENET, fg="white", font_size=10, radius=0.025)

    # ── Split arrows ─────────────────────────────────────────────────────────
    arrow(ax, 0.36, 0.785, 0.25, 0.715, color="#AAAAAA")
    arrow(ax, 0.64, 0.785, 0.75, 0.715, color="#AAAAAA")

    # ─── LEFT BRANCH: MobileNetV2 ────────────────────────────────────────────
    # Frozen backbone
    draw_box(ax, 0.25, 0.675, 0.30, 0.07,
             "MobileNetV2 Backbone",
             "Frozen layers  ·  ImageNet weights",
             bg=C_PRETRAIN, fg="white", font_size=9)
    # Depthwise conv stack visual
    for i, xoff in enumerate([-0.07, -0.02, 0.03, 0.08]):
        col = "#1B7A6E" if i % 2 == 0 else "#14524A"
        draw_box(ax, 0.25 + xoff, 0.585, 0.048, 0.055,
                 f"DW\nConv\n{'×'*(i+1)}", bg=col, fg="white",
                 font_size=7, radius=0.015, lw=1)
    ax.text(0.25, 0.543, "Depthwise-Separable Convolutional Blocks  (96 layers)",
            ha="center", fontsize=7.5, color="#AAAAAA")

    # Fine-tuned head
    draw_box(ax, 0.25, 0.475, 0.30, 0.065,
             "Custom Classification Head",
             "GlobalAvgPool → Dense(128) → Dropout(0.3) → Dense(3)",
             bg=C_FINETUNE1, fg="white", font_size=8.5)

    # Output
    draw_box(ax, 0.25, 0.390, 0.30, 0.065,
             "Color Classification Output",
             "Green  |  Yellow  |  Brown",
             bg=C_OUTPUT, fg="#222222", font_size=9)

    # Vertical arrows (left branch)
    for y1, y2 in [(0.638, 0.612), (0.557, 0.508), (0.442, 0.423)]:
        arrow(ax, 0.25, y1, 0.25, y2)

    ax.text(0.25, 0.340,
            "Fine-tuned on:\nexisting_datasets + own_datasets\n(Green / Yellow / Brown)",
            ha="center", fontsize=8, color="#AAAAAA", linespacing=1.5)

    ax.text(0.25, 0.83, "MODEL 1", ha="center", fontsize=9,
            color=C_FINETUNE1, fontweight="bold")

    # ─── RIGHT BRANCH: EfficientNetB0 ────────────────────────────────────────
    draw_box(ax, 0.75, 0.675, 0.30, 0.07,
             "EfficientNetB0 Backbone",
             "Frozen layers  ·  ImageNet weights",
             bg=C_PRETRAIN, fg="white", font_size=9)
    # MBConv block visuals
    for i, xoff in enumerate([-0.09, -0.04, 0.01, 0.06, 0.11]):
        col = "#374151" if i % 2 == 0 else "#1F2937"
        draw_box(ax, 0.75 + xoff, 0.585, 0.044, 0.055,
                 f"MB\nConv\n{i+1}", bg=col, fg="white",
                 font_size=7, radius=0.015, lw=1)
    ax.text(0.75, 0.543, "MBConv Blocks with Squeeze-and-Excitation  (7 stages)",
            ha="center", fontsize=7.5, color="#AAAAAA")

    draw_box(ax, 0.75, 0.475, 0.30, 0.065,
             "Custom Binary Classification Head",
             "GlobalAvgPool → Dense(128) → Dropout(0.3) → Dense(1)",
             bg=C_FINETUNE2, fg="white", font_size=8.5)

    draw_box(ax, 0.75, 0.390, 0.30, 0.065,
             "Guard Output  (Binary)",
             "0 = Non-Talisay  |  1 = Talisay  (threshold 0.55)",
             bg=C_OUTPUT, fg="#222222", font_size=9)

    for y1, y2 in [(0.638, 0.612), (0.557, 0.508), (0.442, 0.423)]:
        arrow(ax, 0.75, y1, 0.75, y2)

    ax.text(0.75, 0.340,
            "Fine-tuned on:\nPositive: Talisay images (all colors)\nNegative: non-talisay dataset",
            ha="center", fontsize=8, color="#AAAAAA", linespacing=1.5)

    ax.text(0.75, 0.83, "MODEL 2", ha="center", fontsize=9,
            color="#8AB4F8", fontweight="bold")

    # ─── Combined legend / shared detail ─────────────────────────────────────
    ax.plot([0.5, 0.5], [0.22, 0.28], color="#555555", lw=1, ls="--")
    draw_box(ax, 0.50, 0.185, 0.70, 0.065,
             "Shared Training Strategy",
             "Input 224×224 RGB  ·  Adam(lr=0.001)  ·  50 epochs  ·  "
             "EarlyStopping(patience=10)  ·  ReduceLROnPlateau  ·  "
             "Data Augmentation (flip, rotation, brightness, contrast)",
             bg="#16213E", fg="#DDDDDD", font_size=8.2, radius=0.03, lw=1)

    # ─── Legend patches ───────────────────────────────────────────────────────
    leg_items = [
        mpatches.Patch(color=C_IMAGENET, label="Pre-trained source (ImageNet)"),
        mpatches.Patch(color=C_PRETRAIN, label="Frozen backbone layers"),
        mpatches.Patch(color=C_FINETUNE1, label="Fine-tuned head – MobileNetV2"),
        mpatches.Patch(color=C_FINETUNE2, label="Fine-tuned head – EfficientNetB0"),
        mpatches.Patch(color=C_OUTPUT, label="Model output"),
    ]
    ax.legend(handles=leg_items, loc="lower right",
              fontsize=8, framealpha=0.25,
              labelcolor="white",
              facecolor="#0F3460", edgecolor="#AAAAAA")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "methodology_B_transfer_learning.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"[saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C – TESTING MODEL / PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def fig_testing_pipeline():
    """
    Six-stage end-to-end prediction pipeline flowchart.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    C_BG      = "#0D1B2A"
    C_INPUT   = "#E63946"
    C_STAGE   = "#457B9D"
    C_GUARD   = "#E76F51"
    C_DETECT  = "#2A9D8F"
    C_SEG     = "#8338EC"
    C_COLOR   = "#2DC653"
    C_DIM     = "#F4A261"
    C_PRED    = "#E9C46A"
    C_OUT     = "#06D6A0"
    C_REJECT  = "#E63946"
    C_ARROW   = "#AAAAAA"

    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    ax.text(0.5, 0.965, "Testing Model  –  Prediction Pipeline",
            ha="center", va="center", fontsize=16, color="white",
            fontweight="bold")
    ax.text(0.5, 0.940, "End-to-end inference from raw image to oil yield estimate",
            ha="center", va="center", fontsize=10, color="#AAAAAA", style="italic")

    # ── Stage definitions (x_centre, y_centre, width, height, label, sublabel, color) ──
    STAGES = [
        # INPUT
        (0.50, 0.875, 0.36, 0.058, "INPUT", "Talisay Fruit Image  (camera / upload)", C_INPUT),
        # Stage 1
        (0.50, 0.785, 0.36, 0.065,
         "Stage 1 – TalisayGuard Validation",
         "EfficientNetB0 CNN · Binary classifier (Talisay / Non-Talisay) · threshold 0.55",
         C_GUARD),
        # Stage 2
        (0.50, 0.685, 0.36, 0.065,
         "Stage 2 – YOLOv8 Object Detection",
         "Detects Talisay fruit(s) + ₱5 coin · outputs bounding boxes + confidence",
         C_DETECT),
        # Stage 3
        (0.50, 0.585, 0.36, 0.065,
         "Stage 3 – Image Segmentation",
         "YOLOv8-seg masks · isolates fruit from background for clean feature extraction",
         C_SEG),
        # Stage 4
        (0.50, 0.485, 0.36, 0.065,
         "Stage 4 – Color Classification",
         "MobileNetV2 CNN → Green / Yellow / Brown  (HSV fallback if low-confidence)",
         C_COLOR),
        # Stage 5
        (0.50, 0.385, 0.36, 0.065,
         "Stage 5 – Dimension Estimation",
         "₱5 coin (24 mm) as pixel-to-mm reference · computes length, width, kernel mass",
         C_DIM),
        # Stage 6
        (0.50, 0.285, 0.36, 0.065,
         "Stage 6 – Oil Yield Prediction",
         "Ensemble: Random Forest + Gradient Boosting · output clipped 45 % – 65 %",
         C_PRED),
        # OUTPUT
        (0.50, 0.175, 0.36, 0.065,
         "OUTPUT",
         "Predicted Oil Yield %  ·  Confidence Interval  ·  Maturity Stage",
         C_OUT),
    ]

    for (x, y, w, h, label, sub, col) in STAGES:
        draw_box(ax, x, y, w, h, label, sub,
                 bg=col, fg="white", font_size=9, radius=0.022)

    # ── Main flow arrows ──────────────────────────────────────────────────────
    ys = [s[1] for s in STAGES]
    for i in range(len(ys) - 1):
        y_top = ys[i] - STAGES[i][3] / 2
        y_bot = ys[i + 1] + STAGES[i + 1][3] / 2
        arrow(ax, 0.50, y_top, 0.50, y_bot, color=C_ARROW)

    # ── REJECT branch from Stage 1 ───────────────────────────────────────────
    guard_y   = STAGES[1][1]
    guard_h   = STAGES[1][3]
    guard_x   = STAGES[1][0]
    guard_w   = STAGES[1][2]

    # Arrow going right from guard box
    rx_start  = guard_x + guard_w / 2
    rx_end    = 0.865
    arrow(ax, rx_start, guard_y, rx_end, guard_y, color=C_REJECT)

    # Reject box
    draw_box(ax, 0.91, guard_y, 0.085, 0.048,
             "REJECT", "Not a\nTalisay", bg="#7B1D1D", fg="white",
             font_size=8, radius=0.018)

    # Label
    ax.text(rx_start + 0.01, guard_y + 0.025, "fail",
            fontsize=7.5, color=C_REJECT, fontweight="bold")
    ax.text(rx_start - 0.03, guard_y + 0.025, "pass",
            fontsize=7.5, color="#66FF66", fontweight="bold")

    # ── Multi-fruit note beside Stage 6 ──────────────────────────────────────
    pred_x = STAGES[6][0]
    pred_y = STAGES[6][1]
    pred_w = STAGES[6][2]

    draw_box(ax, 0.885, pred_y, 0.155, 0.065,
             "Multi-Fruit Mode",
             "Per-fruit prediction\n→ batch avg + stats",
             bg="#16213E", fg="#DDDDDD", font_size=8, radius=0.018, lw=1)
    ax.annotate("", xy=(0.885 - 0.155/2, pred_y),
                xytext=(pred_x + pred_w / 2, pred_y),
                arrowprops=dict(arrowstyle="<-", color="#888888", lw=1.2))

    # ── Legend: stage colour key ──────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_GUARD,  label="Guard / Validation"),
        mpatches.Patch(color=C_DETECT, label="Object Detection (YOLO)"),
        mpatches.Patch(color=C_SEG,    label="Segmentation"),
        mpatches.Patch(color=C_COLOR,  label="Color Classification (CNN)"),
        mpatches.Patch(color=C_DIM,    label="Dimension Estimation"),
        mpatches.Patch(color=C_PRED,   label="Ensemble ML Prediction"),
        mpatches.Patch(color=C_OUT,    label="Final Output"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=7.5, framealpha=0.3,
              labelcolor="white", facecolor="#0F3460",
              edgecolor="#AAAAAA", title="Pipeline Stages",
              title_fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "methodology_C_testing_pipeline.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"[saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating Methodology Section visuals …")
    fig_transfer_learning()
    fig_testing_pipeline()
    print("\nDone. Two PNG files saved in:", OUT_DIR)
