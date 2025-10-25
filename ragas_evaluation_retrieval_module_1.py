# ragas_eval_module_1_retrieval.py

# Import everything needed from the utility file
from ragas_util import run_evaluation, ALL_METRICS
import pandas as pd

# --- 1. Define Retrieval Test Data ---
TEST_DATA_RETRIEVAL = [
    {
        "question": "Identify all specific legal proceedings related to the Credit Suisse merger mentioned, including the court and appeal status.",
        "ground_truth": "The report mentions class action complaints in the SDNY on behalf of Credit Suisse shareholders and AT1 noteholders. The court granted motions to dismiss the civil RICO claims and conditionally dismissed the Swiss law claims. Plaintiffs in two of these cases have appealed the dismissal as of September 2024.",
    },
    {
        "question": "What are the two components of the Group Items residual amount, and what is its primary purpose?",
        "ground_truth": "Group Items is the residual amount after allocating costs from Group functions (support and control) to business divisions. The two main components are the net income from Group hedging/own debt and the difference between reported PBT and underlying PBT for Non-core and Legacy.",
    },
    {
        "question": "What was the year-to-date Net interest income for UBS Group through June 30, 2025, and how did it compare to the same period in 2024?",
        "ground_truth": "The year-to-date Net interest income for the period ended June 30, 2025, was $3,595 million, compared to $3,475 million for the same period in 2024.",
    },
]

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("      Module 1: RETRIEVAL EVALUATION (Context Focus)")
    print("=" * 70)

    # Call the utility function to run the entire evaluation
    df = run_evaluation(TEST_DATA_RETRIEVAL, ALL_METRICS["retrieval"])

    print("\n--- Detailed Context Metrics Results ---")
    print(df[['question', 'context_precision', 'context_recall', 'context_entity_recall']].to_string())