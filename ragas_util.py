# ragas_utils.py

from raga_pipeline import setup_rag_system, create_rag_chain
import pandas as pd
from datasets import Dataset
import time  # 👈 Import the time module

# --- RAGAS and LangChain Imports ---
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall,
    context_entity_recall, answer_correctness
)

# 1. Configure the RAGAS Judge LLM
# Retain the long timeout and fast model to ensure success
EVAL_LLM = LangchainLLMWrapper(
    ChatOllama(
        model="tinydolphin",
        request_timeout=1800  # Set timeout to 30 minutes
    )
)
EVAL_EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

# 2. Define the metrics for easy access
ALL_METRICS = {
    "retrieval": [#context_precision,
                  context_recall
                  #context_entity_recall
                  ],
    "generation": [faithfulness, answer_relevancy, answer_correctness]
}


# 3. Utility function to run the RAG chain and collect data for RAGAS
def run_rag_chain_for_ragas(question: str, retriever: object, rag_chain: object) -> dict:
    """Executes the RAG pipeline to get the final answer and the retrieved documents."""

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


# 4. Core evaluation function (TIMED)
def run_evaluation(test_data: list, metrics: list) -> pd.DataFrame:
    """
    Runs the RAG chain on the test data, converts to RAGAS Dataset,
    and performs the evaluation with specified metrics.
    """
    retriever = setup_rag_system()
    rag_chain, _ = create_rag_chain(retriever)

    predictions = []

    # ----------------------------------------------------
    # PHASE 1: Data Collection Timing (Running the RAG Chain)
    # ----------------------------------------------------
    start_data_collection = time.time()  #  START TIMER

    print(f"--- Generating {len(test_data)} predictions for evaluation ---")

    for item in test_data:
        ragas_output = run_rag_chain_for_ragas(item["question"], retriever, rag_chain)
        ragas_output["ground_truths"] = [item["ground_truth"]]
        ragas_output["reference"] = item["ground_truth"]
        predictions.append(ragas_output)

    ragas_dataset = Dataset.from_list(predictions)

    end_data_collection = time.time()  # END TIMER
    duration_data_collection = end_data_collection - start_data_collection
    print(f"Data Collection Time: {duration_data_collection:.2f} seconds")

    # ----------------------------------------------------
    # PHASE 2: RAGAS Evaluation Timing (Running the Judge LLM)
    # ----------------------------------------------------
    start_evaluation = time.time()  #  START TIMER

    print(f"\n--- Starting RAGAS Evaluation with {len(metrics)} metrics ---")

    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=EVAL_LLM,
        embeddings=EVAL_EMBEDDINGS
    )

    end_evaluation = time.time()  #  END TIMER
    duration_evaluation = end_evaluation - start_evaluation

    print("\n--- Aggregated Scores ---")
    print(result)
    print(f" **RAGAS Evaluation Time:** {duration_evaluation:.2f} seconds")

    return result.to_pandas()