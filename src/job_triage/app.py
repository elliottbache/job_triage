"""Command-line entrypoint for the job triage workflow."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from job_triage.job_apply.app import apply_to_jobs
from job_triage.job_assess.app import assess_jobs
from job_triage.job_assess.stack_skills import (
    StackSkillAppendResult,
    append_missing_job_score_skills_to_my_stack,
)
from job_triage.job_search.providers.ashbyhq import extract_ashby_listings
from job_triage.logging_utils import configure_logging


def main(argv: Sequence[str] | None = None) -> int:
    """Run one job triage workflow command from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level)
    args.handler(args)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job_triage",
        description="Run job search, assessment, application, and stack workflows.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level to use for the command. Defaults to INFO.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search",
        help="Discover Ashby jobs and persist matching raw jobs.",
    )
    search_parser.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        help="Keyword to match in job title or description. May be repeated.",
    )
    search_parser.set_defaults(handler=_run_search)

    assess_parser = subparsers.add_parser(
        "assess",
        help="Assess active, unapplied raw jobs and persist fit scores.",
    )
    assess_parser.add_argument(
        "--ai-model",
        help="Claude model name to use for assessment. Defaults to workflow setting.",
    )
    assess_parser.set_defaults(handler=_run_assess)

    apply_parser = subparsers.add_parser(
        "apply",
        help="Generate application packets for eligible scored jobs.",
    )
    apply_parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum final score cutoff. Jobs must be strictly above this value.",
    )
    apply_parser.add_argument(
        "--output-folder",
        type=Path,
        help="Folder where application packets should be written.",
    )
    apply_parser.set_defaults(handler=_run_apply)

    update_stack_parser = subparsers.add_parser(
        "update-stack",
        help="Append missing high-priority job-score skills to the stack CSV.",
    )
    update_stack_parser.add_argument(
        "--stack-path",
        type=Path,
        help="Path to the stack CSV. Defaults to private/my_stack.csv.",
    )
    update_stack_parser.set_defaults(handler=_run_update_stack)

    return parser


def _run_search(args: argparse.Namespace) -> None:
    if args.keywords:
        extract_ashby_listings(keywords=set(args.keywords))
        return

    extract_ashby_listings()


def _run_assess(args: argparse.Namespace) -> None:
    if args.ai_model:
        assess_jobs(ai_model=args.ai_model)
        return

    assess_jobs()


def _run_apply(args: argparse.Namespace) -> None:
    if args.output_folder is not None:
        apply_to_jobs(min_score=args.min_score, output_folder=args.output_folder)
        return

    apply_to_jobs(min_score=args.min_score)


def _run_update_stack(args: argparse.Namespace) -> None:
    if args.stack_path is not None:
        result = append_missing_job_score_skills_to_my_stack(
            stack_path=args.stack_path,
        )
    else:
        result = append_missing_job_score_skills_to_my_stack()

    _print_stack_update_result(result)


def _print_stack_update_result(result: StackSkillAppendResult) -> None:
    if not result.added_skills:
        print(
            "No missing high-priority job-score skills to add "
            f"from {result.assessed_score_count} assessed job score(s)."
        )
        return

    print(f"Added {len(result.added_skills)} missing high-priority job-score skill(s):")
    for skill in result.added_skills:
        print(f"- {skill}")


if __name__ == "__main__":
    raise SystemExit(main())
