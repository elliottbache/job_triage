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


def grade_required_stack(
    skill: StackMention,
    *,
    required_level_range: dict[str, tuple[int, int]] = _REQUIRED_LEVEL_RANGE,
    required_years_range: dict[int, tuple[int, int]] = _REQUIRED_YEARS_RANGE,
) -> tuple[int, int]:
    """Estimate a 0-100 score range for a required skill.

    Starts with the full range and narrows it using the skill's required level
    and required years.

    Args:
        skill: Extracted stack mention to grade.
        required_level_range: Optional override mapping required level labels to
            score ranges. Defaults to ``_REQUIRED_LEVEL_RANGE``.
        required_years_range: Optional override mapping required years to score
            ranges. Defaults to ``_REQUIRED_YEARS_RANGE``.

    Returns:
        A ``(min_value, max_value)`` tuple for the skill.
    """

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

    return min_value, max_value


def rank_priorities(
    skill_priorities: list[SkillPriorityItem], skill: StackMention, n_skills: int
) -> float:
    """Return a priority score for one extracted skill.

    Looks up the skill's discrete priority level from ``skill_priorities`` and then
    slightly lowers that score based on how late the skill appears in the stack.
    Earlier skills keep more of their base priority than later ones.

    Args:
        skill_priorities: Assessed priority items for the extracted stack.
        skill: Extracted stack mention to score.
        n_skills: Total number of extracted stack skills.

    Returns:
        A float priority score derived from the discrete priority level and the
        skill's order of appearance.

    Raises:
        ValueError: If ``n_skills`` is less than 1 or if
            ``skill.order_of_appearance`` exceeds ``n_skills``.
        LookupError: If ``skill`` is not present in ``skill_priorities``.
        KeyError: If the matched priority value is not one of ``High``, ``Mid``,
            or ``Low``.
    """

    if n_skills <= 0:
        raise ValueError(
            f"Number of skills should be larger than 0:" f" n_skills = {n_skills}."
        )
    if skill.order_of_appearance > n_skills:
        raise ValueError(
            f"Order of appearance cannot be greater than the number of"
            f" skills in the stack: {skill.order_of_appearance} > {n_skills}"
        )

    # priority level: 1, 2, or 3
    priority_mapping = {"High": 3, "Mid": 2, "Low": 1}
    skill_priority = None
    for skill_priority_item in skill_priorities:
        if skill_priority_item.skill.lower() == skill.skill.lower():
            skill_priority = priority_mapping[skill_priority_item.priority]
            break

    if skill_priority is None:
        raise LookupError(
            f"This skill ({skill}) is not in the list of skills and their priorities: {skill_priorities}"
        )

    priority_level = float(skill_priority)
    print(f"priority_level: {priority_level}")

    # order of appearance: modifies priority level between current level and 1 below
    priority_level += (n_skills - skill.order_of_appearance + 1) / n_skills - 1
    print(f"priority_level: {priority_level}")

    return priority_level


def _modify_range(
    *, previous_min: int = 0, previous_max: int = 100, this_limit: int
) -> int:
    """Scale ``this_limit`` into the current range and clamp it to 0-100.

    Uses Python's ``round()``, so ties round to the nearest even integer.
    """
    limit = round((previous_max - previous_min) * this_limit / 100 + previous_min)
    return min(100, max(0, limit))


# estimate_salary()
# compare_my_stack_to_theirs(): this gives individual_fit_scores: dict[str, Annotated[int, Field(ge=0, le=100)]]
# calculate_final_grade()

if __name__ == "__main__":
    skill = StackMention.model_validate_json(
        """    {
      "skill": "CFD",
      "source_text": "3+ years in CFD, thermal-fluid simulation, or related engineering analysis.",
      "order_of_appearance": 2,
      "required_level": null,
      "required_years": null,
      "priority_signal": "required",
      "substitutes": []
    }"""
    )
    print(grade_required_stack(skill))
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
    print(rank_priorities(skill_priorities, skill, len(skill_priorities)))

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
    print(grade_required_stack(skill))
