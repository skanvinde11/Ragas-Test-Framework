# ragas_eval_module_3_advanced.py

# Import specific components needed for advanced features
from ragas_util import (
    setup_rag_system, create_rag_chain, run_rag_chain_for_ragas,
    EVAL_LLM
)
from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from ragas.metrics import answer_correctness, answer_relevancy
from ragas.messages import HumanMessage, AIMessage

# --- 1. Single-Turn Sample Configuration ---
# Use AnswerCorrectness as the objective
ASPECT_CHECK_QUESTION = "What is the primary contact phone number for UBS Investor Relations in Zurich?"
ASPECT_CHECK_GROUND_TRUTH = "The primary contact for Investor Relations in Zurich is +41-44-234-4100."

# --- 2. Multi-Turn Configuration ---
# Multi-turn evaluation works on pre-generated samples to assess consistency
multi_turn_sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="What are the five main business divisions of UBS?"),
        AIMessage(
            content="The five divisions are Global Wealth Management, Personal & Corporate Banking, Asset Management, the Investment Bank, and Non-core and Legacy."),
        HumanMessage(content="Of those, which one handles positions not aligned with the bank's main strategy?"),
        # Follow-up
        AIMessage(
            content="That would be the Non-core and Legacy division, which consists of positions and businesses not aligned with the UBS strategy and policies.")
    ],
    # The ground truth for multi-turn is defined as the expected context/outcome
    ground_truth="The five business divisions are Global Wealth Management, Personal & Corporate Banking, Asset Management, the Investment Bank, and Non-core and Legacy. Non-core and Legacy handles non-strategic positions.",
    contexts=[
        "We report five business divisions: Global Wealth Management, Personal & Corporate Banking, Asset Management, the Investment Bank, and Non-core and Legacy.",
        "Non-core and Legacy consists of positions and businesses not aligned with our strategy and policies."
    ]
)

if __name__ == "__main__":
    retriever = setup_rag_system()
    rag_chain, _ = create_rag_chain(retriever)

    print("\n" + "=" * 70)
    print("      Module 3: ADVANCED & MULTI-TURN EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # PART A: SINGLE-TURN CHECK (Using AnswerCorrectness)
    # ------------------------------------------------------------------
    print("\n--- PART A: Single-Turn Correctness Check ---")
    critique_output = run_rag_chain_for_ragas(ASPECT_CHECK_QUESTION, retriever, rag_chain)

    critique_data = [
        {
            "question": ASPECT_CHECK_QUESTION,
            "answer": critique_output['answer'],
            "contexts": critique_output['contexts'],
            "ground_truths": [ASPECT_CHECK_GROUND_TRUTH]
        }
    ]
    critique_dataset = Dataset.from_list(critique_data)

    critique_result = evaluate(
        dataset=critique_dataset,
        metrics=[answer_correctness, answer_relevancy],  # Use core metrics
        llm=EVAL_LLM
    )

    print("\n--- Single-Turn Correctness Results ---")
    print(critique_result.to_pandas())

    # ------------------------------------------------------------------
    # PART B: MULTI-TURN SAMPLE EVALUATION
    # ------------------------------------------------------------------
    print("\n\n--- PART B: Multi-Turn Sample Evaluation (Evaluates pre-generated sample) ---")

    multi_turn_dataset = EvaluationDataset(samples=[multi_turn_sample])

    # We use AnswerCorrectness/Relevancy, which treats the entire history/ground_truth as a single complex sample
    multi_turn_result = evaluate(
        dataset=multi_turn_dataset,
        metrics=[answer_correctness, answer_relevancy],
        llm=EVAL_LLM
    )

    print("\n--- Multi-Turn Results (Consistency Check) ---")
    print(multi_turn_result.to_pandas())
    print("\n" + "=" * 70)