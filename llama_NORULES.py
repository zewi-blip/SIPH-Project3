import os
import re
import ollama
import pandas as pd


def classify_ultrasound_pure_knowledge(
    csv_path, label_type, output_csv="classification_results_pure.csv"
):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)
    all_results = []

    print(f"--- Starting Pure-Knowledge AI Agent for {label_type} Data ---")

    for index, row in df.iterrows():
        sphericity = row["original_shape2D_Sphericity"]
        entropy = row["original_firstorder_Entropy"]
        color_mean = row["original_firstorder_Mean"]

        if sphericity > 0.90:
            shape_type = "Round"
        elif sphericity > 0.82:
            shape_type = "Oval"
        else:
            shape_type = "Irregular"

        elongation = row.get("original_shape2d_Elongation", 1.0)
        orientation = "Parallel" if elongation > 0.5 else "Non-parallel"

        if color_mean < 50:
            color_grade = "Hypoechoic (Dark)"
        elif entropy > 4.5:
            color_grade = "Complex/ Mixed (Heterogenous colors)"
        else:
            color_grade = "Isoechoic (Neutral/Mixed)"

        margin_type = (
            "Circumscribed"
            if sphericity > 0.88
            else "Not circumscribed (Indistinct/Angular)"
        )
        echo_pattern = "Homogeneous" if entropy < 3.8 else "Heterogeneous"

        shadow_conf = row.get("shadow_conf", 0.0)
        enhance_conf = row.get("enhancement_conf", 0.0)

        if shadow_conf > 0.5 and shadow_conf > enhance_conf:
            posterior_features = (
                "Acoustic shadowing (Typical of malignant/dense structures)"
            )
        elif enhance_conf > 0.5 and enhance_conf > shadow_conf:
            posterior_features = (
                "Acoustic enhancement (Typical of benign/fluid-filled cysts)"
            )
        else:
            posterior_features = (
                "No posterior features / No significant shadowing or enhancement"
            )

        # prompt with no logical rules, only testing clinical domain knowledge
        prompt = f"""
        System: You are an expert Radiologist AI. You must evaluate ultrasound features strictly using your internal knowledge of the ACR BI-RADS Atlas 5th Edition.
        Patient ID: {row['ID']}
        
        Morphological Findings:
        - Shape: {shape_type}
        - Margin: {margin_type}
        - Echo pattern: {echo_pattern}
        - Color grade: {color_grade}
        - Orientation: {orientation}
        - Posterior Features: {posterior_features}

        Task: Based on these clinical findings and standard ACR guidelines, assign the correct BI-RADS Category (0, 1, 2, 3, 4, or 5). Do not use any external hints.
        
        Response Format:
        Category: [Score]
        Reasoning: [One sentence explaining your diagnostic conclusion]
        """

        try:
            response = ollama.generate(model="llama3", prompt=prompt)
            resp_text = response["response"]

            cat_match = re.search(r"Category:\s*([0-6])", resp_text)
            assigned_cat = cat_match.group(1) if cat_match else "Unknown"

            all_results.append(
                {
                    "ID": row["ID"],
                    "Original_Label": label_type,
                    "Predicted_Category": assigned_cat,
                    "Shape": shape_type,
                    "Margin": margin_type,
                    "Echo": echo_pattern,
                    "Posterior_Features": posterior_features,
                    "Full_AI_Reasoning": resp_text.replace("\n", " "),
                }
            )

            print(f"Processed {row['ID']}: Category {assigned_cat}")

        except Exception as e:
            print(f"Error processing {row['ID']}: {e}")

    # save all results to a new CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n--- Success! Results saved to {output_csv} ---")

    # print summary of classifications
    print("\nTotal Classification Counts:")
    print(results_df["Predicted_Category"].value_counts().sort_index())


if __name__ == "__main__":
    classify_ultrasound_pure_knowledge("benign_features.csv", "Benign", "benign_results_llamaPURE.csv")
    #classify_ultrasound_pure_knowledge(
        #"malignant_features.csv", "Malignant", "malignant_results_llamaPURE.csv"
    #)