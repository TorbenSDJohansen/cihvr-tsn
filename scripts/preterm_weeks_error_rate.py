# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Preterm birth number of weeks performance is very poor. It seems this is a
result of images often containing a range, such as "5-6", whereas the label
only contains 1 number and not a range, even if present on the image. Further,
it is not consistent whether the first or the second number in a range was
used as the label, and thus a lot of incorrect predictions take the form:
    On image: 5-6
    Label: 6
    Prediction: 6
Therefore, calculate the error rate on the evaluation set allowing for +/- 1.

"""


import argparse
import os

from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--files', type=str, nargs='+')
    parser.add_argument('--fn-out', type=str, default=None)

    args = parser.parse_args()

    for file in args.files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f'requested file {file} does not exist')

    if args.fn_out is not None:
        if os.path.isfile(args.fn_out):
            raise FileExistsError(f'--fn-out {args.fn_out} already exists')

    return args


def calculate_metrics(pred: pd.DataFrame) -> Tuple[float, float]:
    seq_acc = (pred['pred'] == pred['label']).mean()
    correct_within_range = []

    for label, transcription in pred[['label', 'pred']].values:
        if label == transcription:
            correct_within_range.append(True)
            continue

        if label == '0=Mangler' or transcription == '0=Mangler':
            # We know they are not *both* equal to '0=Mangler'
            correct_within_range.append(False)
            continue

        if label == 'bad cpd' or transcription == 'bad cpd':
            # We know they are not *both* equal to 'bad cpd'
            correct_within_range.append(False)
            continue

        label = int(label)
        transcription = int(transcription)

        if abs(label - transcription) <= 1:
            correct_within_range.append(True)
        else:
            correct_within_range.append(False)

    seq_acc_within_range = sum(correct_within_range) / len(correct_within_range)

    return seq_acc, seq_acc_within_range


def main():
    args = parse_args()

    preds = {}

    for file in args.files:
        pred = pd.read_csv(file)
        name = os.path.basename(os.path.dirname(file))

        preds[name] = pred

    results = []

    for name, pred in preds.items():
        results.append([name, 'full', *calculate_metrics(pred)])

        sub = pred[pred['label'] != '0=Mangler']
        results.append([name, 'non-empty', *calculate_metrics(sub)])

    results = pd.DataFrame(results, columns=['Model', 'Sample', 'Acc', 'Acc within +/- 1 range'])

    if args.fn_out is None:
        print(results)
        return

    # Cast percentage and round
    results[results.columns[2:]] = (100 * results[results.columns[2:]]).round(1)

    # Write .tex

    with pd.option_context("max_colwidth", 1000):
        results_str = results.to_latex(
            index=False,
            escape=False,
            )

    results_str = '\n'.join(results_str.split('\n')[2:-3])

    with open(args.fn_out, 'w', encoding='utf-8') as file:
        print(results_str, file=file)


if __name__ == '__main__':
    main()
