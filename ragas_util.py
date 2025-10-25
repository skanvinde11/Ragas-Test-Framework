# ragas_utils.py

from raga_pipeline import setup_rag_system, create_rag_chain
import pandas as pd
from datasets import Dataset

# --- RAGAS and LangChain Imports ---
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough, \
    RunnableParallel  # <---  RunnableParallel is imported
from ragas.metrics import (
    # Corrected RAGAS Metric Imports
    faithfulness, answer_relevancy, context_precision, context_recall,
    context_entity_recall, answer_correctness
)

# 1. Configure the RAGAS Judge LLM
EVAL_LLM = LangchainLLMWrapper(ChatOllama(model="tinydolphin"))
EVAL_EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

# 2. Define the metrics for easy access
ALL_METRICS = {
    "retrieval": [context_precision, context_recall, context_entity_recall],
    "generation": [faithfulness, answer_relevancy, answer_correctness]
}


# 3. Utility function to run the RAG chain and collect data for RAGAS
def run_rag_chain_for_ragas(question: str, retriever: object, rag_chain: object) -> dict:
    """
    Executes the RAG pipeline to get the final answer and the retrieved documents,
    formatting the output for the RAGAS Dataset object.
    """

    # 💥 FIX: Wrap the dictionary in RunnableParallel to make it callable
    full_pipeline = RunnableParallel({
        "context": retriever,
        "answer": rag_chain,
        "question": RunnablePassthrough()
    })

    output = full_pipeline.invoke(question)

    return {
        "question": output['question'],
        "answer": output['answer'],
        "contexts": [doc.page_content for doc in output['context']],
    }


# 4. Core evaluation function
def run_evaluation(test_data: list, metrics: list) -> pd.DataFrame:
    """
    Runs the RAG chain on the test data, converts to RAGAS Dataset,
    and performs the evaluation with specified metrics.
    """
    retriever = setup_rag_system()
    rag_chain, _ = create_rag_chain(retriever)

    predictions = []
    print(f"--- Generating {len(test_data)} predictions for evaluation ---")

    for item in test_data:
        ragas_output = run_rag_chain_for_ragas(item["question"], retriever, rag_chain)

        # Ragas requires a list for ground_truths
        ragas_output["ground_truths"] = [item["ground_truth"]]

        # ADD THE 'reference' COLUMN
        # This is mandatory for the 'context_precision' metric in modern RAGAS
        ragas_output["reference"] = item["ground_truth"]

        predictions.append(ragas_output)

    ragas_dataset = Dataset.from_list(predictions)

    print(f"\n--- Starting RAGAS Evaluation with {len(metrics)} metrics ---")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=EVAL_LLM,
        embeddings=EVAL_EMBEDDINGS,
        #  Limit the number of workers to 1 to prevent resource contention
        # This forces sequential processing and prevents timeouts.
        raise_exceptions=False,  # Optional: prevents the script from crashing immediately
        workers=1
    )

    print("\n--- Aggregated Scores ---")
    print(result)

    return result.to_pandas()