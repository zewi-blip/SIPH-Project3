import pandas as pd
import ollama
import os
import re


def classify_ultrasound(csv_path, label_type, output_csv="classification_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)
    all_results = []

    print(f"--- Starting AI Agent for {label_type} Data ---")

    for index, row in df.iterrows():
        sphericity = row['original_shape2D_Sphericity']
        entropy = row['original_firstorder_Entropy']
        color_mean = row['original_firstorder_Mean']


        # morphological logic based on BI-RADS Atlas
        if sphericity > 0.90:
            shape_type = "Round"
        elif sphericity > 0.82:
            shape_type = "Oval"
        else:
            shape_type = "Irregular"

        elongation = row.get('original_shape2d_Elongation', 1.0)
        orientation = "Parallel" if elongation > 0.5 else "Non-parallel"

        if color_mean < 50:
            color_grade = "Hypoechoic (Dark)"
        elif entropy > 4.5:
            color_grade = "Complex/ Mixed (Heterogenous colors)"
        else:
            color_grade = "Isoechoic (Neutral/Mixed)"

        margin_type = "Circumscribed" if sphericity > 0.88 else "Not circumscribed (Indistinct/Angular)"
        echo_pattern = "Homogeneous" if entropy < 3.8 else "Heterogeneous"

        shadow_conf = row.get("shadow_conf", 0.0)
        enhance_conf = row.get("enhancement_conf", 0.0)

        if shadow_conf > 0.5 and shadow_conf > enhance_conf:
            posterior_features = "Acoustic shadowing (Typical of malignant/dense structures)"
        elif enhance_conf > 0.5 and enhance_conf > shadow_conf:
            posterior_features = (
                "Acoustic enhancement (Typical of benign/fluid-filled cysts)"
            )
        else:
            posterior_features = "No posterior features / No significant shadowing or enhancement"

        prompt = f"""
        System: You are an expert Radiologist AI using the ACR BI-RADS Atlas 5th Edition.
        Patient ID: {row['ID']}
        Morphological Features:
        - Shape: {shape_type}
        - Margin: {margin_type}
        - Echo pattern:  {echo_pattern}
        - Color grade: {color_grade}
        - Orientation: {orientation}
        - Posterior Features: {posterior_features}
        

        Rules :
        - Category 1: Negative
        - Category 2: Benign (Oval/Round & Parallel & Circumscribed & Enhancement & Lighter/ Isoechoic)
        - Category 3: Probably Benign, some doubts/ not definitive
        - Category 4: Suspicious (Irregular OR Not circumscribed & Non-parallel & Shadow & Dark/Mixed colors)
        - Category 5: Highly Suggestive (Irregular AND Not circumscribed & Non-parallel & Shadow & Dark/ Mixed colors)
        - Category 0: Incomplete scan or artifacts

        Task: Assign a Category (0, 1, 2, 3, 4, or 5).
        Response Format:
        Category: [Score]
        Reasoning: [One sentence]
        """

        try:
            response = ollama.generate(model='llama3', prompt=prompt)
            resp_text = response['response']

            # extract the category number using Regex
            cat_match = re.search(r'Category:\s*([0-6])', resp_text)
            assigned_cat = cat_match.group(1) if cat_match else "Unknown"

            # store data for the new CSV
            all_results.append({
                "ID": row['ID'],
                "Original_Label": label_type,
                "Predicted_Category": assigned_cat,
                "Shape": shape_type,
                "Margin": margin_type,
                "Echo": echo_pattern,
                "Posterior_Features": posterior_features,
                "Full_AI_Reasoning": resp_text.replace('\n', ' ')
            })

            print(f"Processed {row['ID']}: Category {assigned_cat}")

        except Exception as e:
            print(f"Error processing {row['ID']}: {e}")

    # save all results to a new CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n--- Success! Results saved to {output_csv} ---")

    # print summary of classifications
    print("\nTotal Classification Counts:")
    print(results_df['Predicted_Category'].value_counts().sort_index())


if __name__ == "__main__":
    #classify_ultrasound("benign_features.csv", "Benign", "benign_results_llamaFULL.csv")
    classify_ultrasound("malignant_features.csv", "Malignant", "malignant_results_llamaFULL.csv")