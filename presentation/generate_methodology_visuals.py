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
    Transfer learning architecture laid out as a staged training pipeline:
      1) Data preparation
      2) Frozen feature-extraction backbone
      3) Custom classifier head
      4) Training configuration / fitting
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Palette ─────────────────────────────────────────────────────────────
    C_BG = "#F8FAFC"
    C_TITLE = "#0F172A"
    C_SUB = "#475569"
    C_SECTION = "#E2E8F0"

    C_DATA = "#FDE68A"
    C_BACKBONE = "#C7F9CC"
    C_HEAD = "#FBCFE8"
    C_TRAIN = "#DDD6FE"
    C_ARROW = "#64748B"

    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.95, "3.4 Transfer Learning Architecture",
            ha="center", va="center", fontsize=20, color=C_TITLE,
            fontweight="bold")
    ax.text(0.5, 0.915,
            "ImageNet transfer learning pipeline used for Talisay fruit classification and guard validation",
            ha="center", va="center", fontsize=10.5, color=C_SUB, style="italic")

    # ── Section labels ────────────────────────────────────────────────────────
    ax.text(0.13, 0.865, "1. Data Prep", fontsize=10, color=C_TITLE, fontweight="bold")
    ax.text(0.43, 0.865, "2. Feature Extraction (Frozen)", fontsize=10, color=C_TITLE, fontweight="bold")
    ax.text(0.77, 0.865, "3. Classifier Head + Training", fontsize=10, color=C_TITLE, fontweight="bold", ha="center")

    # ── Left lane: data preparation ───────────────────────────────────────────
    left_steps = [
        ("Raw Dataset", "train_dataset.csv"),
        ("Class Balancing", "resampling / weighted classes"),
        ("Augmentation", "flip · rotation · brightness"),
        ("Normalize", "pixel values to [0, 1]"),
        ("Resize to 224×224", "RGB input tensor"),
    ]

    y_positions = np.linspace(0.78, 0.42, len(left_steps))
    for idx, ((title, sub), y) in enumerate(zip(left_steps, y_positions)):
        draw_box(ax, 0.13, y, 0.20, 0.06, title, sub,
                 bg=C_DATA, fg="#111827", font_size=8.5,
                 radius=0.015, border="#94A3B8", lw=1.2)
        if idx < len(left_steps) - 1:
            arrow(ax, 0.13, y - 0.037, 0.13, y_positions[idx + 1] + 0.038,
                  color=C_ARROW, lw=1.3)

    # ── Middle lane: frozen backbone block ───────────────────────────────────
    backbone_frame = FancyBboxPatch(
        (0.29, 0.31), 0.29, 0.50,
        boxstyle="round,pad=0.015",
        linewidth=1.3, edgecolor="#94A3B8", facecolor=C_SECTION,
        linestyle="--", zorder=1
    )
    ax.add_patch(backbone_frame)

    draw_box(ax, 0.435, 0.77, 0.23, 0.06,
             "ImageNet Pre-trained Backbone",
             "MobileNetV2 / EfficientNetB0 (frozen)",
             bg=C_BACKBONE, fg="#0F172A", font_size=8.5,
             radius=0.015, border="#94A3B8", lw=1.2)

    backbone_blocks = [
        "Conv Stem 32", "Bottleneck ×2", "Bottleneck ×3",
        "Bottleneck ×4", "Bottleneck ×2", "Conv 1×1 + BN"
    ]
    backbone_y = np.linspace(0.70, 0.40, len(backbone_blocks))
    for i, (label, y) in enumerate(zip(backbone_blocks, backbone_y)):
        draw_box(ax, 0.435, y, 0.20, 0.05, label,
                 "", bg="#DCFCE7", fg="#111827", font_size=8,
                 radius=0.012, border="#86EFAC", lw=1.0)
        if i < len(backbone_blocks) - 1:
            arrow(ax, 0.435, y - 0.031, 0.435, backbone_y[i + 1] + 0.031,
                  color=C_ARROW, lw=1.2)

    ax.text(0.435, 0.335, "Feature maps transferred to custom head",
            ha="center", fontsize=8, color=C_SUB, style="italic")

    # Connect left lane to frozen backbone
    arrow(ax, 0.23, 0.42, 0.315, 0.76, color=C_ARROW, lw=1.4)

    # ── Right lane: classifier head + training ───────────────────────────────
    right_steps = [
        ("GlobalAvgPool2D", "flattens spatial feature maps"),
        ("Dense(128) + ReLU", "Dropout(0.5)"),
        ("Dense(N classes)", "Softmax / Sigmoid output"),
        ("Compile", "Adam + loss + metrics"),
        ("Callbacks", "EarlyStopping, ReduceLROnPlateau"),
        ("model.fit", "epochs + batch size"),
    ]
    right_y = np.linspace(0.72, 0.30, len(right_steps))
    for i, ((title, sub), y) in enumerate(zip(right_steps, right_y)):
        bg_col = C_HEAD if i < 3 else C_TRAIN
        draw_box(ax, 0.77, y, 0.24, 0.055, title, sub,
                 bg=bg_col, fg="#111827", font_size=8.4,
                 radius=0.014, border="#94A3B8", lw=1.2)
        if i < len(right_steps) - 1:
            arrow(ax, 0.77, y - 0.034, 0.77, right_y[i + 1] + 0.034,
                  color=C_ARROW, lw=1.2)

    # Connect backbone to head
    arrow(ax, 0.55, 0.40, 0.65, 0.72, color=C_ARROW, lw=1.4)

    # ── Footer note ──────────────────────────────────────────────────────────
    draw_box(ax, 0.50, 0.12, 0.82, 0.055,
             "Output Models",
             "Model 1: Color Classifier (Green / Yellow / Brown)   |   Model 2: TalisayGuard (Talisay / Non-Talisay)",
             bg="#DBEAFE", fg="#0F172A", font_size=8.4,
             radius=0.02, border="#93C5FD", lw=1.2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "methodology_B_transfer_learning.png")
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=C_BG)
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
