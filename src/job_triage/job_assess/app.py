from job_triage.job_assess.schemas import StackMention

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


def rank_priorites(
    skill: StackMention,
    n_skills: int,
    *,
    required_level_range: dict[str, tuple[int, int]] = _REQUIRED_LEVEL_RANGE,
    required_years_range: dict[int, tuple[int, int]] = _REQUIRED_YEARS_RANGE,
) -> tuple[int, int]:
    """Estimate a 0-100 score range for a required skill.

    Starts with the full range and narrows it using the skill's required level,
    required years, and order of appearance in the stack.

    Args:
        skill: Extracted stack mention to grade.
        n_skills: Total number of skills mentioned in the stack.
        required_level_range: Mapping from required level labels to score ranges.
        required_years_range: Mapping from required years to score ranges.

    Returns:
        A ``(min_value, max_value)`` tuple for the skill.

    Raises:
        ValueError: If ``n_skills`` is less than 1 or if
            ``skill.order_of_appearance`` exceeds ``n_skills``.
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
    print(f"min_value: {min_value}, max_value: {max_value}")

    # required years
    if skill.required_years is not None:
        this_min, this_max = required_years_range.get(skill.required_years, (100, 100))
        (min_value, max_value) = (
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_min
            ),
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_max
            ),
        )
    print(f"min_value: {min_value}, max_value: {max_value}")

    # order of appearance
    this_min = round((n_skills - skill.order_of_appearance) / n_skills * 100)
    this_max = round((n_skills - skill.order_of_appearance + 1) / n_skills * 100)
    (min_value, max_value) = (
        _modify_range(
            previous_min=min_value, previous_max=max_value, this_limit=this_min
        ),
        _modify_range(
            previous_min=min_value, previous_max=max_value, this_limit=this_max
        ),
    )
    print(f"min_value: {min_value}, max_value: {max_value}")

    return min_value, max_value


def _modify_range(
    *, previous_min: int = 0, previous_max: int = 100, this_limit: int
) -> int:
    """Scale ``this_limit`` into the current range and clamp it to 0-100.

    Uses Python's ``round()``, so ties round to the nearest even integer.
    """
    limit = round((previous_max - previous_min) * this_limit / 100 + previous_min)
    return min(100, max(0, limit))

    # define range min and max as 0 and 100
    # if required_level is not None:
    #   None = 0-100
    #   Basic = 0-30
    #   Intermediate = 30-60
    #   Advanced = 60-80
    #   Expert = 80-100
    #   max = (current_max - current_min) * range_max/global_max + current_min
    #   min = (current_max - current_min) * range_min/global_max + current_min
    # if required_years is not None:
    #   1 yr = 0-30
    #   2 yr = 30-45
    #   3 yr = 45-60
    #   4 yr = 60-70
    #   5 yr = 70-80
    #   6 yr = 80-90
    #   7 yr = 90-100
    #     define range from order


# estimate_salary()
# compare_my_stack_to_theirs(): this gives individual_fit_scores: dict[str, Annotated[int, Field(ge=0, le=100)]]
# calculate_final_grade()

if __name__ == "__main__":
    skill = StackMention.model_validate_json(
        """    {
      "skill": "Thermal-fluid simulation",
      "source_text": "3+ years in CFD, thermal-fluid simulation, or related engineering analysis.",
      "order_of_appearance": 2,
      "required_level": null,
      "required_years": null,
      "priority_signal": "required",
      "substitutes": []
    }"""
    )
    print(grade_required_stack(skill))

    skill = StackMention.model_validate_json(
        """    {
      "skill": "Thermal-fluid simulation",
      "source_text": "3+ years in CFD, thermal-fluid simulation, or related engineering analysis.",
      "order_of_appearance": 2,
      "required_level": "Basic",
      "required_years": 3,
      "priority_signal": "required",
      "substitutes": []
    }"""
    )
    print(grade_required_stack(skill))
