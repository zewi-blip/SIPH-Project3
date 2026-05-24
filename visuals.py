import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_llm_distribution(
        benign_csv="benign_results_llamaPURE.csv",
        malignant_csv="malignant_results_llamaPURE.csv",
        output_image="llm_distribution_PURE.png",
):
    # All required BI-RADS categories to ensure completeness
    categories = ["0", "1", "2", "3", "4", "5"]

    # Use subplots instead of plt.figure() to follow best practices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- 1. Process Benign Results ---
    if os.path.exists(benign_csv):
        df_b = pd.read_csv(benign_csv)
        df_b["Predicted_Category"] = df_b["Predicted_Category"].astype(str)

        # Sum occurrences and ensure all classes from 0 to 5 are included in order
        counts_b = (
            df_b["Predicted_Category"].value_counts().reindex(categories, fill_value=0)
        )

        # Color rules: 2 or 3 are expected, higher (>3) is red, lower is warning/neutral
        colors_b = []
        for cat in categories:
            val = int(cat)
            if val > 3:
                colors_b.append("#d9534f")  # Red for over-classified benign cases
            elif val in [2, 3]:
                colors_b.append("#5cb85c")  # Green for expected benign categories
            else:
                colors_b.append("#aaaaaa")  # Gray for categories 0 and 1

        axes[0].bar(counts_b.index, counts_b.values, color=colors_b, edgecolor="black")
        axes[0].set_title("Benign Dataset - LLM Distribution", fontsize=14, pad=12)
        axes[0].set_xlabel("Predicted BI-RADS Category", fontsize=12)
        axes[0].set_ylabel("Image Count", fontsize=12)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Place count values cleanly on top of each bar
        for i, v in enumerate(counts_b.values):
            axes[0].text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=10, weight="bold")
    else:
        axes[0].text(0.5, 0.5, f"Missing:\n{benign_csv}", ha="center", va="center", color="red", fontsize=12)
        axes[0].set_title("Benign Dataset - Not Found")

    # --- 2. Process Malignant Results ---
    if os.path.exists(malignant_csv):
        df_m = pd.read_csv(malignant_csv)
        df_m["Predicted_Category"] = df_m["Predicted_Category"].astype(str)

        # Sum occurrences and ensure all classes from 0 to 5 are included in order
        counts_m = (
            df_m["Predicted_Category"].value_counts().reindex(categories, fill_value=0)
        )

        # Color rules: 4 or 5 are all good (green), otherwise red
        colors_m = []
        for cat in categories:
            val = int(cat)
            if val in [4, 5]:
                colors_m.append("#5cb85c")  # Green for expected malignant categories
            else:
                colors_m.append("#d9534f")  # Red for under-classified malignant cases

        axes[1].bar(counts_m.index, counts_m.values, color=colors_m, edgecolor="black")
        axes[1].set_title("Malignant Dataset - LLM Distribution", fontsize=14, pad=12)
        axes[1].set_xlabel("Predicted BI-RADS Category", fontsize=12)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        # Place count values cleanly on top of each bar
        for i, v in enumerate(counts_m.values):
            axes[1].text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=10, weight="bold")
    else:
        axes[1].text(0.5, 0.5, f"Missing:\n{malignant_csv}", ha="center", va="center", color="red", fontsize=12)
        axes[1].set_title("Malignant Dataset - Not Found")

    # Tight layout ensures text labels and titles never overlap or get cut off
    plt.tight_layout()

    # Save chart safely to a file instead of using show()
    plt.savefig(output_image, dpi=300)
    print(f"Success! Distribution chart saved as '{output_image}'.")


if __name__ == "__main__":
    plot_llm_distribution()