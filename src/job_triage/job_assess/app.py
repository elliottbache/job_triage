from pathlib import Path

import pandas as pd

from job_triage.job_assess.schemas import SkillPriorityItem, StackMention

_REQUIRED_LEVEL_RANGE = {
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


def _compare_my_stack_to_theirs(
    *,
    stack_mentions: list[StackMention],
    skill_priorities: list[SkillPriorityItem],
    my_path: Path = _DEFAULT_MY_STACK_PATH,
) -> float:
    """Compare extracted required skills against the user's saved stack.

    The raw signed score is computed as achieved fit divided by the maximum
    possible fit for the same extracted skills and priority assignments. That
    signed score is then remapped from ``[-100, 100]`` to ``[0, 100]``.

    Args:
        stack_mentions: Extracted required skills from the job post.
        skill_priorities: Assessed priorities for the extracted skills.
        my_path: Path to the CSV file containing the user's skill grades.

    Returns:
        A normalized fit score from 0 to 100, where 50 is neutral.

    Raises:
        LookupError: If an extracted skill has no matching priority item when
            computing the maximum possible fit.
    """

    # group skills into list of list of substitutable skills
    all_skill_groups = _group_all_substitute_skills(stack_mentions)
    if not all_skill_groups:
        return 100.0

    max_possible_fit = 0.0
    for skill_group in all_skill_groups:
        group_skill_fit = 0.0
        for stack_mention in skill_group:
            skill_priority = _get_skill_priority_item(
                stack_mention.skill, skill_priorities
            )
            if skill_priority is None:
                raise LookupError(
                    f"This skill ({stack_mention.skill}) is not in the list of skills and their priorities: {skill_priorities}"
                )
            skill_fit = 100 * _rank_priority(
                stack_mention,
                skill_priority=skill_priority,
                stack_mentions=stack_mentions,
                skill_priorities=skill_priorities,
            )
            if skill_fit > group_skill_fit:
                group_skill_fit = skill_fit

        max_possible_fit += group_skill_fit

    my_stack = _read_my_stack(my_path)
    not_in_my_stack, in_my_stack = set(), set()
    total_fit = 0.0

    for skill_group in all_skill_groups:
        group_skill_fit = -100.0
        for stack_mention in skill_group:
            skill = stack_mention.skill.lower()
            skill_priority = _get_skill_priority_item(skill, skill_priorities)
            if skill_priority is None:
                raise LookupError(f"This skill {skill} does not have a priority.")
            my_level = my_stack.get(skill, 0)
            skill_fit = _calculate_skill_fit(
                my_level=my_level,
                skill=stack_mention,
                skill_priority=skill_priority,
                stack_mentions=stack_mentions,
                skill_priorities=skill_priorities,
            )

            if skill not in my_stack:
                # keep track of the skills I am missing to add to CSV file later
                not_in_my_stack.add(skill)
            else:
                in_my_stack.add(skill)

            if skill_fit > group_skill_fit:
                group_skill_fit = skill_fit

        total_fit += group_skill_fit

    signed_score = total_fit / max_possible_fit * 100
    signed_score = max(-100.0, min(100.0, signed_score))
    return (signed_score + 100) / 2


def _group_all_substitute_skills(
    stack_mentions: list[StackMention],
) -> list[list[StackMention]]:
    """Group substitutable skills into non-overlapping skill groups."""
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
    """Return one skill group containing a stack mention and its substitutes."""

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


def _get_skill_priority_item(
    skill: str, skill_priorities: list[SkillPriorityItem]
) -> SkillPriorityItem | None:
    """Return the matching priority item for a skill, ignoring case.

    Args:
        skill: Skill name to look up.
        skill_priorities: Priority items to search.

    Returns:
        The matching ``SkillPriorityItem`` if found; otherwise ``None``.
    """

    for skill_priority_item in skill_priorities:
        if skill_priority_item.skill.lower() == skill.lower():
            return skill_priority_item

    return None


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
    skill_priority: SkillPriorityItem,
    stack_mentions: list[StackMention],
    skill_priorities: list[SkillPriorityItem],
) -> float:
    """Compute the fit contribution for one required skill.

    The contribution combines the required-grade midpoint for the skill and its
    relative priority within its priority group.

    Args:
        my_level: The user's saved grade for this skill.
        skill: Extracted required skill from the job post.
        skill_priority: Assessed priority item for this skill.
        stack_mentions: Full extracted stack, used for intra-priority ordering.
        skill_priorities: Full set of assessed priority items.

    Returns:
        The fit contribution for this skill.
    """
    grade = _grade_required_stack(skill)
    priority = _rank_priority(
        skill,
        skill_priority=skill_priority,
        stack_mentions=stack_mentions,
        skill_priorities=skill_priorities,
    )
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
    and required years. If neither required level nor required years is available, returns a low
    default requirement of ``20.0``.


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
    skill_priority: SkillPriorityItem,
    stack_mentions: list[StackMention],
    skill_priorities: list[SkillPriorityItem],
) -> float:
    """Return a priority score for one extracted skill.

    Looks up the skill's discrete priority level from ``skill_priority`` and then
    slightly lowers that score based on how late the skill appears in the stack.
    Earlier skills keep more of their base priority than later ones.

    Args:
        skill: Extracted stack mention to score.
        skill_priority: Assessed priority item for the extracted skill.
        stack_mentions: Extracted stack mentions in job-post order.
        skill_priorities: Assessed priority items for the extracted stack.

    Returns:
        A float priority score derived from the discrete priority level and the
        skill's order of appearance.

    Raises:
        LookupError: If no extracted stack mentions belong to the matched
            priority group.
        KeyError: If the matched priority value is not one of ``High``, ``Mid``,
            or ``Low``.
    """

    # priority level: 1, 2, or 3
    priority_mapping = {"High": 3, "Mid": 2, "Low": 1}

    priority_level = float(priority_mapping[skill_priority.priority])

    same_priority_skills = {
        item.skill.lower()
        for item in skill_priorities
        if item.priority == skill_priority.priority
    }
    ordered_group = [
        stack_mention
        for stack_mention in sorted(
            stack_mentions, key=lambda item: item.order_of_appearance
        )
        if stack_mention.skill.lower() in same_priority_skills
    ]

    group_size = len(ordered_group)
    if group_size == 0:
        raise LookupError(
            f"No stack mentions found for priority group {skill_priority.priority}."
        )

    order_in_group = next(
        index
        for index, stack_mention in enumerate(ordered_group, start=1)
        if stack_mention.skill.lower() == skill.skill.lower()
    )

    # order of appearance: modifies priority level within the current priority band
    priority_level += (group_size - order_in_group + 1) / group_size - 1

    return priority_level


def _calculate_final_grade() -> None:
    """Calculate the final overall job-fit grade.

    This function is a placeholder for the application-level aggregation step
    that combines stack fit with any additional scoring signals.
    """
    pass


# estimate_salary()
# compare_my_stack_to_theirs(): this gives individual_fit_scores: dict[str, Annotated[int, Field(ge=0, le=100)]]
# calculate_final_grade()

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
    skill_priorities = [
        SkillPriorityItem(skill="CFD", priority="High"),
        SkillPriorityItem(skill="ANSYS Fluent", priority="High"),
        SkillPriorityItem(skill="OpenFOAM", priority="High"),
        SkillPriorityItem(skill="Python", priority="Mid"),
        SkillPriorityItem(skill="Turbulence modeling", priority="Mid"),
        SkillPriorityItem(skill="Meshing", priority="Mid"),
        SkillPriorityItem(skill="Heat transfer", priority="Mid"),
        SkillPriorityItem(skill="Linux", priority="Low"),
        SkillPriorityItem(skill="C++", priority="Low"),
    ]
    print(
        _rank_priority(
            skill,
            skill_priority=SkillPriorityItem(skill="CFD", priority="High"),
            stack_mentions=stack_mentions,
            skill_priorities=skill_priorities,
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
