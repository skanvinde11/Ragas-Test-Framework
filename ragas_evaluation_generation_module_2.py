# ragas_eval_module_2_generation.py

from ragas_util import run_evaluation, ALL_METRICS
import pandas as pd

# --- 1. Define Generation Test Data ---
TEST_DATA_GENERATION = [
    {
        "question": "What was the Net interest income for the quarter ended 30.6.25, and what was the corresponding figure for 30.6.24?",
        "ground_truth": "The net interest income for the quarter ended June 30, 2025, was $1,629 million, compared to $1,535 million for the same quarter in 2024.",
    },
    {
        "question": "What are the five business divisions that UBS reports as operating segments?",
        "ground_truth": "The five business divisions are Global Wealth Management, Personal & Corporate Banking, Asset Management, the Investment Bank, and Non-core and Legacy.",
    },
    {
        "question": "How did the underlying loss before tax for the Non-core and Legacy segment in Q2 2025 compare to Q2 2024?",
        "ground_truth": "The underlying loss before tax for the Non-core and Legacy segment was $477 million in Q2 2025, which was higher than the underlying loss of $371 million in Q2 2024.",
    },
]

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("      Module 2: GENERATION EVALUATION (Answer Focus)")
    print("=" * 70)

    # Use the 'generation' metrics (Faithfulness, Answer Relevancy, Answer Correctness)
    df = run_evaluation(TEST_DATA_GENERATION, ALL_METRICS["generation"])

    print("\n--- Detailed Answer Metrics Results ---")
    # Display the correct metrics that were evaluated
    print(df[['question', 'faithfulness', 'answer_relevancy', 'answer_correctness']].to_string())
    print(
        "\n**Note on Answer Correctness:** This metric effectively replaces Aspect Critic for objective correctness checks.")