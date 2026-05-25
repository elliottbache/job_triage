import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from job_triage.job_assess.schemas import (
    JobPostAssessment,
    JobPostExtraction,
    LocationConstraint,
    RoleFamily,
    SeniorityLevel,
    StackMention,
    WorkArrangement,
)

_REQUIRED_LEVEL_RANGE = {
    "Novice": (0, 0),
    "Basic": (0, 30),
    "Intermediate": (30, 60),
    "Advanced": (60, 80),
    "Expert": (80, 100),
}
_REQUIRED_YEARS_RANGE = {
    0: (0, 0),
    1: (0, 31),
    2: (31, 54),
    3: (54, 70),
    4: (70, 81),
    5: (81, 89),
    6: (89, 95),
    7: (95, 100),
}
_DEFAULT_MY_STACK_PATH = Path("private") / "my_stack.csv"
_DEFAULT_SALARY_MATRIX_PATH = Path("expected_gross_salary_matrix_eur.csv")
_DEFAULT_MINIMUM_SALARY = 55000

logger = logging.getLogger(__name__)


def evaluate_job_fit(
    job_post_extraction: JobPostExtraction,
    job_post_assessment: JobPostAssessment,
) -> int:
    """Compute an application-level fit score for one job post.

    The current pipeline combines three steps:
    1. score the user's stack against the extracted required skills
    2. estimate salary from either the explicit range or the fallback matrix
    3. reject the role if seniority, location, or salary constraints fail

    If the validation step fails, the function returns ``0``. Otherwise it
    converts the stack-fit and salary estimate into a single integer score.

    Args:
        job_post_extraction: Structured extraction of the job post.
        job_post_assessment: Structured assessment produced from the extraction.

    Returns:
        An application-level fit score as an integer that scales with the estimated
        salary.
    """

    stack_fit = _compare_my_stack_to_theirs(
        stack_mentions=job_post_extraction.stack_mentions,
    )
    salary = _estimate_salary(
        job_post_assessment=job_post_assessment, job_fit=stack_fit
    )
    if not _validate_seniority_location_salary(
        seniority=job_post_assessment.seniority,
        role=job_post_assessment.role_family,
        location=job_post_assessment.location_constraint,
        work_arrangement=job_post_assessment.work_arrangement,
        salary=salary,
    ):
        return 0

    # make triple the salary double the fit score
    salary_multiplier = (salary - _DEFAULT_MINIMUM_SALARY) / (
        _DEFAULT_MINIMUM_SALARY
    ) / 2.0 + 1

    return int(stack_fit * salary_multiplier)


def _estimate_salary(
    *,
    job_post_assessment: JobPostAssessment,
    job_fit: int,
    salary_matrix_path: Path = _DEFAULT_SALARY_MATRIX_PATH,
) -> int:
    """Estimate gross salary in euros for a job assessment.

    Uses the explicit salary range from the assessment when available.
    Otherwise, falls back to the salary matrix keyed by role family,
    seniority, and location constraint.

    Args:
        job_post_assessment: Normalized assessment data for the job post.
        job_fit: Overall fit score from 0 to 100.
        salary_matrix_path: Path to the fallback salary matrix CSV.

    Returns:
        The estimated gross annual salary in euros.
    """
    if job_post_assessment.salary_range is None:
        salary = _retrieve_salary_from_matrix(
            job_post_assessment=job_post_assessment,
            salary_matrix_path=salary_matrix_path,
        )
    else:
        salary = _estimate_salary_from_range(job_post_assessment.salary_range, job_fit)

    return salary


def _estimate_salary_from_range(salaries: list[int], job_fit: int) -> int:
    """Estimate salary from an explicit lower and upper bound.

    The input range is sorted defensively. Scores below 50 map to the lower
    bound, while scores from 50 to 100 interpolate linearly up to the upper
    bound.

    Args:
        salaries: Two salary bounds in euros.
        job_fit: Overall fit score from 0 to 100. Values outside this range are
            clamped before interpolation.

    Returns:
        The estimated gross annual salary in euros.

    Raises:
        ValueError: If ``salaries`` does not contain exactly two elements.
    """
    if len(salaries) != 2:
        raise ValueError(
            f"Salary range should have two elements: min salary and max salary.  We have: {salaries}"
        )

    salary_range = salaries.copy()
    salary_range.sort()

    job_fit = max(0, min(100, job_fit))
    if job_fit < 50:
        return salary_range[0]
    else:
        return int(
            (salary_range[1] - salary_range[0]) * (job_fit - 50) / 50 + salary_range[0]
        )


def _retrieve_salary_from_matrix(
    *,
    job_post_assessment: JobPostAssessment,
    salary_matrix_path: Path = _DEFAULT_SALARY_MATRIX_PATH,
) -> int:
    """Retrieve a fallback salary estimate from the matrix CSV.

    The lookup first tries the exact ``(role_family, seniority, location)``
    tuple, then relaxes location to ``Worldwide``, then relaxes seniority to
    ``Junior`` for the same role, then falls back to ``Mechanical Engineer /
    Junior / Worldwide``. If none of those keys exist, the minimum salary in
    the matrix is returned, or ``0`` when the matrix is empty.

    Args:
        job_post_assessment: Normalized assessment data for the job post.
        salary_matrix_path: Path to the salary matrix CSV.

    Returns:
        The fallback gross annual salary in euros.
    """
    salary_table = {}
    min_salary = np.inf
    with open(salary_matrix_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["role family"], row["seniority level"], row["location"])
            salary_table[key] = int(row["salary"])
            min_salary = (
                int(row["salary"]) if int(row["salary"]) < min_salary else min_salary
            )
    min_salary = int(min_salary) if min_salary < np.inf else 0

    query = (
        job_post_assessment.role_family,
        job_post_assessment.seniority,
        job_post_assessment.location_constraint,
    )
    salary = salary_table.get(query)
    if salary is None:
        query = (
            job_post_assessment.role_family,
            job_post_assessment.seniority,
            "Worldwide",
        )
        salary = salary_table.get(query)
    if salary is None:
        query = (job_post_assessment.role_family, "Junior", "Worldwide")
        salary = salary_table.get(query)
    if salary is None:
        query = ("Mechanical Engineer", "Junior", "Worldwide")
        salary = salary_table.get(query)
    if salary is None:
        salary = min_salary

    return salary


def _compare_my_stack_to_theirs(
    *,
    stack_mentions: list[StackMention],
    my_path: Path = _DEFAULT_MY_STACK_PATH,
) -> int:
    """Compare extracted required skills against the user's saved stack.

    The raw signed score is computed as achieved fit divided by the maximum
    possible fit for the same extracted skills and priority signals. That
    signed score is then remapped from ``[-100, 100]`` to ``[0, 100]``.

    Args:
        stack_mentions: Extracted required skills from the job post.
        my_path: Path to the CSV file containing the user's skill grades.

    Returns:
        A normalized fit score from 0 to 100, where 0 is the worst modeled
        fit, 50 is neutral, and 100 is the best modeled fit.

    """

    # group skills into list of list of substitutable skills
    all_skill_groups = _group_all_substitute_skills(stack_mentions)
    if not all_skill_groups:
        return 100

    max_possible_fit = 0.0
    for skill_group in all_skill_groups:
        group_skill_fit = 0.0
        for stack_mention in skill_group:
            skill_fit = 100 * _rank_priority(
                stack_mention,
                stack_mentions=stack_mentions,
            )
            if skill_fit > group_skill_fit:
                group_skill_fit = skill_fit

        max_possible_fit += group_skill_fit

    my_stack = _read_my_stack(my_path)
    not_in_my_stack = set()
    total_fit = 0.0

    for skill_group in all_skill_groups:
        group_skill_fit = -100.0
        for stack_mention in skill_group:
            skill = stack_mention.skill.lower()
            my_level = my_stack.get(skill, 0)
            skill_fit = _calculate_skill_fit(
                my_level=my_level,
                skill=stack_mention,
                stack_mentions=stack_mentions,
            )

            if skill not in my_stack:
                # keep track of the skills I am missing to add to CSV file later
                not_in_my_stack.add(skill)

            if skill_fit > group_skill_fit:
                group_skill_fit = skill_fit

        total_fit += group_skill_fit

    signed_score = total_fit / max_possible_fit * 100
    signed_score = max(-100.0, min(100.0, signed_score))

    logger.debug(f"Not in my stack: {not_in_my_stack}")

    return int((signed_score + 100) / 2)


def _group_all_substitute_skills(
    stack_mentions: list[StackMention],
) -> list[list[StackMention]]:
    """Group substitutable skills into non-overlapping skill groups.

    Each returned inner list represents one requirement unit for scoring.
    A skill and any substitutes explicitly linked by an "or" statement in the
    job post are scored as alternatives rather than double-counted.  If Skill B
    is an alternative to Skill A, Skill A has a 10 fit score (pretty bad), and Skill B
    has a 100 fit score (the best), then the fit score for that group would be 100.

    Args:
        stack_mentions: Extracted stack mentions from the job post.

    Returns:
        A list of non-overlapping substitute groups in encounter order.
    """
    grouped_skills = []
    seen_skills = set()

    for stack_mention in stack_mentions:
        if stack_mention.skill.lower() in seen_skills:
            continue

        skill_group = _group_single_substitute_skill(
            stack_mention=stack_mention,
            stack_mentions=stack_mentions,
        )
        grouped_skills.append(skill_group)

        for grouped_skill in skill_group:
            seen_skills.add(grouped_skill.skill.lower())

    return grouped_skills


def _group_single_substitute_skill(
    *, stack_mention: StackMention, stack_mentions: list[StackMention]
) -> list[StackMention]:
    """Return one substitute group for a single extracted skill.

    The skills in this group can be interchanged, and it is only expected to have
    experience in one of them.

    Args:
        stack_mention: The root extracted skill for the group.
        stack_mentions: Full extracted stack used to resolve substitute names.

    Returns:
        A list containing ``stack_mention`` and any matching substitute skills.

    Raises:
        LookupError: If a named substitute cannot be found in ``stack_mentions``.
    """

    grouped_skills = list()
    grouped_skills.append(stack_mention)

    substitutes = stack_mention.substitutes
    if not substitutes:
        return grouped_skills

    for substitute in substitutes:
        skill = _get_stack_mention(substitute, stack_mentions)
        if skill is None:
            raise LookupError(f"Skill {substitute} is not in their stack.")
        if skill not in grouped_skills:
            grouped_skills.append(skill)

    return grouped_skills


def _get_stack_mention(
    skill: str, stack_mentions: list[StackMention]
) -> StackMention | None:
    """Return the matching extracted stack mention for a skill, ignoring case.

    Args:
        skill: Skill name to look up.
        stack_mentions: Extracted stack mentions to search.

    Returns:
        The matching ``StackMention`` if found; otherwise ``None``.
    """
    for stack_mention in stack_mentions:
        if stack_mention.skill.lower() == skill.lower():
            return stack_mention

    return None


def _read_my_stack(path: Path) -> dict[str, int]:
    """Read the user's saved skill grades from CSV.

    The CSV is expected to contain ``skill`` and ``grade`` columns. Skill names
    are normalized to lowercase before returning the mapping.

    Args:
        path: Path to the CSV file.

    Returns:
        A mapping from lowercase skill name to numeric grade.
    """
    df = pd.read_csv(path)
    df["skill"] = df["skill"].str.lower()

    return df.set_index("skill")["grade"].to_dict()


def _calculate_skill_fit(
    *,
    my_level: int,
    skill: StackMention,
    stack_mentions: list[StackMention],
) -> float:
    """Compute the fit contribution for one required skill.

    The contribution combines the required-grade midpoint for the skill and its
    relative priority within its priority-signal group.

    Args:
        my_level: The user's saved grade for this skill.
        skill: Extracted required skill from the job post.
        stack_mentions: Full extracted stack, used for intra-priority ordering.

    Returns:
        The fit contribution for this skill.
    """
    grade = _grade_required_stack(skill)
    priority = _rank_priority(skill, stack_mentions=stack_mentions)
    if my_level >= grade:
        return 100 * priority

    return (my_level - grade) * priority


def _grade_required_stack(
    skill: StackMention,
    *,
    required_level_range: dict[str, tuple[int, int]] = _REQUIRED_LEVEL_RANGE,
    required_years_range: dict[int, tuple[int, int]] = _REQUIRED_YEARS_RANGE,
) -> float:
    """Estimate a 0-100 required-grade midpoint for a skill.

    Starts with the full range and narrows it using the skill's required level
    and required years. If neither required level nor required years is available,
    returns a low default requirement of ``20.0``.


    Args:
        skill: Extracted stack mention to grade.
        required_level_range: Optional override mapping required level labels to
            score ranges. Defaults to ``_REQUIRED_LEVEL_RANGE``.
        required_years_range: Optional override mapping required years to score
            ranges. Defaults to ``_REQUIRED_YEARS_RANGE``.

    Returns:
        A midpoint grade from 0 to 100 for the skill.
    """

    if skill.required_level is None and skill.required_years is None:
        return 20.0

    min_value, max_value = 0, 100
    # required level
    if skill.required_level is not None:
        this_min, this_max = required_level_range.get(skill.required_level, (0, 100))
        (min_value, max_value) = (
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_min
            ),
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_max
            ),
        )

    # required years
    if skill.required_years is not None:
        this_min, this_max = required_years_range.get(
            skill.required_years, (100, 100)
        )  # default value for years over 7
        (min_value, max_value) = (
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_min
            ),
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_max
            ),
        )

    return (min_value + max_value) / 2


def _modify_range(
    *, previous_min: int = 0, previous_max: int = 100, this_limit: int
) -> int:
    """Scale ``this_limit`` into the current range and clamp it to 0-100.

    Uses Python's ``round()``, so ties round to the nearest even integer.
    """
    limit = round((previous_max - previous_min) * this_limit / 100 + previous_min)
    return min(100, max(0, limit))


def _rank_priority(
    skill: StackMention,
    *,
    stack_mentions: list[StackMention],
) -> float:
    """Return a priority score for one extracted skill.

    Maps the skill's extracted ``priority_signal`` to a numeric weight, then
    slightly lowers that weight based on how late the skill appears among skills
    with the same signal. Earlier skills keep more of their base priority than
    later skills in the same priority-signal group.

    Args:
        skill: Extracted stack mention to score.
        stack_mentions: Extracted stack mentions in job-post order.

    Returns:
        A float priority score derived from the priority signal and the skill's
        order of appearance within that signal group.

    Raises:
        LookupError: If no extracted stack mentions belong to the matched
            priority-signal group.
        KeyError: If the priority signal is not one of ``required``,
            ``highly_preferred``, ``preferred``, ``bonus``, or ``not_required``.
    """

    # priority signal: 1-5
    priority_mapping = {
        "required": 5,
        "highly_preferred": 4,
        "preferred": 3,
        "bonus": 2,
        "not_required": 1,
    }

    priority_signal = float(priority_mapping[skill.priority_signal])

    ordered_group = [
        stack_mention
        for stack_mention in sorted(
            stack_mentions, key=lambda item: item.order_of_appearance
        )
        if stack_mention.priority_signal == skill.priority_signal
    ]

    group_size = len(ordered_group)
    if group_size == 0:
        raise LookupError(
            f"No stack mentions found for priority signal {skill.priority_signal}."
        )

    order_in_group = next(
        (
            index
            for index, stack_mention in enumerate(ordered_group, start=1)
            if stack_mention.skill.lower() == skill.skill.lower()
        ),
        None,
    )
    if order_in_group is None:
        raise LookupError(
            f"Skill {skill.skill} was not found in priority signal group "
            f"{skill.priority_signal}."
        )

    # order of appearance modifies priority within the current signal band
    priority_signal += (group_size - order_in_group + 1) / group_size - 1

    # scale priority signal from 0-3 instead of 0-5
    priority_signal *= 3 / 5.0

    return priority_signal


def _validate_seniority_location_salary(
    seniority: SeniorityLevel,
    role: RoleFamily,
    location: LocationConstraint,
    work_arrangement: WorkArrangement,
    salary: int,
) -> bool:
    """Validate coarse screening constraints before final scoring.

    The current rules reject:
    - lead/principal roles in software, backend, and data engineering
    - jobs whose normalized location is ``Other``
    - jobs that are categorized as ``Onsite`` (hybrid jobs that are far from Valencia
      are considered Onsite)
    - salaries that are below the configured minimum threshold

    Args:
        seniority: Normalized seniority for the role.
        role: Normalized role family.
        location: Normalized location constraint.
        work_arrangement: Normalized arrangement (``Remote``, ``Hybrid``, or ``Onsite``)
        salary: Estimated gross annual salary in euros.

    Returns:
        ``True`` when the role passes the coarse validation checks, otherwise
        ``False``.
    """
    # seniority fit
    if seniority in ["Lead", "Principal"] and role in [
        "Software Engineer",
        "Backend Engineer",
        "Data Engineer",
    ]:
        return False

    # location fit
    if location == "Other":
        return False

    # work arrangement fit
    if work_arrangement == "Onsite":
        return False

    # salary fit
    return salary >= _DEFAULT_MINIMUM_SALARY


if __name__ == "__main__":
    skill = StackMention.model_validate_json(
        """{
      "skill": "CFD",
      "source_text": "3+ years in CFD, thermal-fluid simulation, or related engineering analysis.",
      "order_of_appearance": 2,
      "required_level": null,
      "required_years": null,
      "priority_signal": "required",
      "substitutes": []
    }"""
    )
    stack_mentions = [skill]
    print(_grade_required_stack(skill))
    print(
        _rank_priority(
            skill,
            stack_mentions=stack_mentions,
        )
    )

    skill = StackMention.model_validate_json(
        """    {
      "skill": "CFD",
      "source_text": "3+ years in CFD, thermal-fluid simulation, or related engineering analysis.",
      "order_of_appearance": 2,
      "required_level": "Basic",
      "required_years": 3,
      "priority_signal": "required",
      "substitutes": []
    }"""
    )
    print(_grade_required_stack(skill))
