from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from feature_extraction_rag import predict_single_image
from posterior_feature_ext import extract_posterior_features
import json
from streamlit_rag import get_augmented_context
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)
IMAGE_SYSTEM_PROMPT = """You are a medical assistant specialized in breast imaging and radiology.
    You will be provided with:
    - Extracted radiomics features from a breast mass
    - Posterior acoustic features (shadow/enhancement scores and histogram statistics)
    - BI-RADS guidelines as context

    Your task:
    - Predict the most appropriate BI-RADS category (0-6) for the mass
    - Ground your prediction strictly in the BI-RADS guidelines provided in the context

    Posterior acoustic feature interpretation:
    - shadow_score and enhancement_score are continuous values between 0 and 1
    - shadow_indicator is binary (0 or 1) and should never be interpreted in isolation
    - When both shadow_score and enhancement_score are below 0.3, treat posterior acoustic features 
      as inconclusive — do not interpret near-zero scores as evidence of either benignity or malignancy
    - When posterior features are inconclusive, weight the radiomics features more heavily
    - A meaningful shadow_score (> 0.3) suggests posterior acoustic shadowing per BI-RADS guidelines
    - A meaningful enhancement_score (> 0.3) suggests posterior acoustic enhancement per BI-RADS guidelines
    - Always contextualize shadow_indicator against the shadow_score magnitude — 
      a low shadow_score overrides a positive shadow_indicator and must be explicitly acknowledged
    - Histogram statistics (mean, std, skewness, energy, entropy) reflect regional brightness 
      and texture of the posterior, left, and right regions around the mass —
      beside_mean is the average brightness of the flanking regions and serves as the baseline

    You must respond with a valid JSON object and nothing else — no preamble, no explanation, no markdown fences.
    Use exactly this structure:
    {
        "birads_level": <integer 0–6>,
        "classification": "<benign/malignant>",
    }"""

CHROMA_PATH = r"./RAG_app/chroma_langchain_db"

SYSTEM_PROMPT = """You are a medical assistant specialized in breast imaging and radiology. 
    You answer questions strictly based on the BI-RADS (Breast Imaging Reporting and Data System) guidelines provided to you as context.

    Rules:
    - Only answer based on the provided context, do not use outside knowledge
    - If the answer is not found in the context, say "I cannot find this information in the BI-RADS guidelines"
    - Be precise and clinical in your language
    - Do not provide personal medical advice or diagnoses
    - If a question is ambiguous, ask for clarification before answering
    - Do not reorder or reinterpret the categories, present them exactly as described in the context"""

def rag_app():

    vector_store = load_index()
    client_LLM = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
        token="Token"
    )
    while True:
        user_question = input("User: ")
        augmented_context = get_augmented_context(user_question, vector_store)
        response = client_LLM.chat_completion(
            messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context: {augmented_context}\n\nQuestion: {user_question}"}
        ]
        )
        print("System: ", response.choices[0].message.content)
    return

def load_index():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    vector_store = Chroma(
        collection_name="bi_rads_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    return vector_store

def get_augmented_context(prompt, vector_store):
    results = vector_store.similarity_search(
        prompt,
        k=6,
    )
    context = "\n\n".join([res.page_content for res in results])
    return context


def test_model():
    preds = {
        "0": {"correct" : 0, "total" : 0},
        "1": {"correct" : 0, "total" : 0},
        "2": {"correct" : 0, "total" : 0},
        "3": {"correct" : 0, "total" : 0},
        "4": {"correct" : 0, "total" : 0},
        "5": {"correct" : 0, "total" : 0},
        "6": {"correct" : 0, "total" : 0},
    }
    ds_path = r"C:\Users\giuli\Desktop\UNI\1 anno\2o Semestre\Healthcare\img_ds\samples"
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(
        collection_name="bi_rads_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    client = InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        token="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"
    )
    user_question = "Classify and return the bi-rads level for this image"
    for filename in os.listdir(ds_path):
        img_path = os.path.join(ds_path, filename)
        if "benign" in filename:
            sample_label = "benign"
        else:
            sample_label = "malignant"
        try:
            context = get_augmented_context(user_question, vector_store)
            mass_info = predict_single_image(img_path)
            posterior_features = extract_posterior_features(img_path)
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_question}\n\nMass Info: {mass_info} \n\n Posterior Features: {json.dumps(posterior_features, indent = 2)}"}
                ]
            )
            answer = response.choices[0].message.content
            result = json.loads(answer)
            b_level = result["birads_level"]
            classification = result["classification"]
            if classification.lower().strip() == sample_label:
                preds[f"{b_level}"]["correct"] += 1
            preds[f"{b_level}"]["total"] += 1
        except Exception as e:
            print(f"Skipping {filename}: {e}")
        # compute the accuracy level per bi-rads and overall
        # print them together with how many times each bi-rads level was computed

    total_correct = sum(v["correct"] for v in preds.values())
    total_samples = sum(v["total"] for v in preds.values())

    print("\n--- Per BI-RADS Level Results ---")
    for level, stats in preds.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  BI-RADS {level}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%) — count: {stats['total']}")
        else:
            print(f"  BI-RADS {level}: no samples")

    overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    print(f"\n--- Overall Accuracy: {total_correct}/{total_samples} ({overall_acc:.1f}%) ---")

    return
if __name__ == "__main__":
    #rag_app()
    test_model()