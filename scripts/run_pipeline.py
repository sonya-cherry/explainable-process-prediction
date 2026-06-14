"""Command-line entry point for the complete prediction pipeline.

This script contains no data-processing, model-training, or evaluation logic.
It only parses command-line arguments and calls the reusable functions from
``src.pipeline``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# When running:
#
#     python scripts/run_pipeline.py
#
# Python adds ``scripts/`` to the import path, but not necessarily the
# repository root. We add the root explicitly so imports from ``src`` work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline import run_evaluation_pipeline, run_training_pipeline


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete explainable process outcome prediction "
            "pipeline on an event log."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the input event log in XES or CSV format.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help=(
            "Directory for generated reports and figures. "
            "Default: <repository>/outputs."
        ),
    )

    parser.add_argument(
        "--min-prefix",
        type=int,
        default=1,
        help=(
            "Minimum prefix length used when generating running-case "
            "examples. Default: 1."
        ),
    )

    parser.add_argument(
        "--positive-final-activities",
        nargs="+",
        default=["Closed", "Resolved"],
        metavar="ACTIVITY",
        help=(
            "Final lifecycle transitions treated as positive outcomes. "
            "Default: Closed Resolved."
        ),
    )

    parser.add_argument(
        "--rf-scoring",
        default="f1",
        help=(
            "Validation metric used to select the Random Forest. "
            "Default: f1."
        ),
    )

    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP generation for a faster run.",
    )

    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=300,
        help=(
            "Maximum number of test rows used for global SHAP calculations. "
            "Default: 300."
        ),
    )

    return parser


def print_run_summary(
    training_result,
    evaluation_result,
) -> None:
    """Print the main results and generated file paths."""
    print("\nDataset summary")
    print("---------------")

    for name, value in training_result.dataset_summary.items():
        readable_name = name.replace("_", " ").capitalize()
        print(f"{readable_name}: {value}")

    print("\nSelected Random Forest parameters")
    print("---------------------------------")

    for name, value in (
        training_result.best_random_forest_parameters.items()
    ):
        print(f"{name}: {value}")

    print("\nValidation-set comparison")
    print("-------------------------")
    print(
        training_result.validation_comparison.to_string(
            index=False
        )
    )

    print("\nTest-set comparison")
    print("-------------------")
    print(
        evaluation_result.test_comparison.to_string(
            index=False
        )
    )

    print("\nGenerated reports")
    print("-----------------")

    for name, path in evaluation_result.report_paths.items():
        print(f"{name}: {path}")

    print("\nGenerated figures")
    print("-----------------")

    for name, path in evaluation_result.figure_paths.items():
        print(f"{name}: {path}")


def main(argv: list[str] | None = None) -> int:
    """Run training and evaluation from command-line arguments."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.min_prefix < 1:
        parser.error("--min-prefix must be at least 1.")

    if args.shap_max_samples < 1:
        parser.error("--shap-max-samples must be at least 1.")

    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    print(f"Input event log: {data_path}")
    print(f"Output directory: {output_dir}")

    try:
        print("\nRunning training pipeline...")

        training_result = run_training_pipeline(
            event_log_path=data_path,
            positive_final_activities=(
                args.positive_final_activities
            ),
            min_prefix=args.min_prefix,
            random_forest_scoring=args.rf_scoring,
        )

        print("Training completed.")
        print("\nRunning evaluation pipeline...")

        evaluation_result = run_evaluation_pipeline(
            training_result=training_result,
            output_dir=output_dir,
            generate_shap=not args.no_shap,
            shap_max_samples=args.shap_max_samples,
        )

    except (FileNotFoundError, ValueError) as error:
        print(
            f"\nPipeline failed: {error}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "\nPipeline interrupted by the user.",
            file=sys.stderr,
        )
        return 130

    print("Evaluation completed.")

    print_run_summary(
        training_result,
        evaluation_result,
    )

    print("\nPipeline completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
