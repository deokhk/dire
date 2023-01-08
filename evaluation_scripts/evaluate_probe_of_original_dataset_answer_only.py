"""
Script to evaluate predictions on the probe of the original dataset (HotpotQA).
"""

from collections import defaultdict
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from metrics import (
    BaseMetric,
    ProbeAnswerMetric,
    ProbeSupportingFactsMetric
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to evaluate predictions on"
                                     " the probe of the original dataset (HotpotQA).")
    parser.add_argument('probe_input_file_path', type=str)
    parser.add_argument('probe_prediction_file_path', type=str)
    parser.add_argument('original_prediction_file_path', type=str)

    args = parser.parse_args()

    with open(args.probe_input_file_path, "r") as file:
        probe_inputs = json.load(file)

    with open(args.probe_prediction_file_path, "r") as file:
        probe_predictions = [json.loads(line)
                             for line in file.readlines() if line.strip()]

    with open(args.original_prediction_file_path, "r") as file:
        original_predictions = [json.loads(line)
                                for line in file.readlines() if line.strip()]

    answer_metric = ProbeAnswerMetric()

    probe_predictions_ = defaultdict(list)
    for prediction in probe_predictions:
        probe_predictions_[prediction["question_id"]].append(prediction)
    probe_predictions = probe_predictions_

    question_ids = set()
    for input_instance in probe_inputs:
        question_id = input_instance["_id"]
        question_ids.add(question_id)

        label_answer = input_instance["answer"]

        probe_prediction = probe_predictions[question_id].pop()
        predicted_answer = probe_prediction["answer"]
        predicted_confidence = probe_prediction["answer_confidence"]

        answer_metric.store_prediction(
            predicted_answer=predicted_answer,
            predicted_confidence=predicted_confidence,
            label_answer=label_answer,
            question_id=question_id,
            is_probe=True
        )


    original_predictions = {prediction["question_id"]: prediction
                            for prediction in original_predictions}
    for question_id in question_ids:

        original_prediction = original_predictions[question_id]
        predicted_answer = original_prediction["answer"]

        answer_metric.store_prediction(
            predicted_answer=predicted_answer,
            question_id=question_id,
            is_probe=False
        )


    answer_scores = answer_metric.compute_dataset_scores()

    print(f"Answer Scores")
    print(json.dumps(answer_scores, indent=4))

