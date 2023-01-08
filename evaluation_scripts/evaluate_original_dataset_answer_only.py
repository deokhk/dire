"""
Script to evaluate predictions on the original dataset (HotpotQA).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from metrics import (
    BaseMetric,
    AnswerMetric,
    SupportingFactsMetric
)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script to evaluate predictions"
                                     " on the original dataset (HotpotQA).")
    parser.add_argument('input_file_path', type=str)
    parser.add_argument('prediction_file_path', type=str)

    args = parser.parse_args()

    with open(args.input_file_path, "r") as file:
        inputs = json.load(file)

    with open(args.prediction_file_path, "r") as file:
        predictions = [json.loads(line) for line in file.readlines() if line.strip()]

    answer_metric = AnswerMetric()

    predictions = {prediction["question_id"]: prediction for prediction in predictions}
    for input_instance in inputs:
        question_id = input_instance["_id"]

        label_answer = input_instance["answer"]

        prediction = predictions[question_id]
        predicted_answer = prediction["answer"]

        answer_metric.store_prediction(
            predicted_answer=predicted_answer,
            label_answer=label_answer,
            question_id=question_id
        )


    answer_scores = answer_metric.compute_dataset_scores()

    print(f"Answer Scores")
    print(json.dumps(answer_scores, indent=4))
