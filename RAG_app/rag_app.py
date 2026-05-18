from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


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

if __name__ == "__main__":
    rag_app()