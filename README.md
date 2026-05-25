# AI Job Triage Tool

## IN PROGRESS!!!  USE AT YOUR OWN RISK!!!

[![CI](https://github.com/elliottbache/job_triage/actions/workflows/ci.yaml/badge.svg)](https://github.com/elliottbache/job_triage/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/github/elliottbache/job_triage/graph/badge.svg?token=kNwbaexX4N)](https://codecov.io/github/elliottbache/job_triage)
[![Release](https://img.shields.io/github/v/release/elliottbache/job_triage)](https://github.com/elliottbache/job_triage/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)


## Short description

An AI-assisted job triage tool that turns raw job posts into structured data, evaluates how well a role matches your skills, and helps you prioritize which applications are worth your time.


## Grading system

The current grading system is implemented in `src/job_triage/job_assess/app.py`. The public entry point is `evaluate_job_fit()`, which returns a single integer score for a job post.

The score is calculated in three stages:

1. `evaluate_job_fit()` calls `_compare_my_stack_to_theirs()` to compute a stack-fit score from `0` to `100`.
2. `evaluate_job_fit()` calls `_estimate_salary()` to estimate gross annual salary, either from the job post salary range or from the fallback salary matrix.
3. `evaluate_job_fit()` calls `_validate_seniority_location_salary()` to reject jobs that fail hard constraints. Rejected jobs receive `0`.

Priority signal, required level, and required years are separate inputs to the stack-fit calculation. Required level and required years do not affect priority. Instead, `_grade_required_stack()` combines required level and required years into the required skill grade: the estimated level of ability needed for that skill on a `0` to `100` scale. `_rank_priority()` separately maps the extracted `priority_signal` to a priority weight and adjusts that weight by order of appearance within the same signal group. `_calculate_skill_fit()` then combines those two pieces by checking whether the user's saved grade meets the required grade and multiplying that result by the priority weight.

In other words, required level and required years answer "how good do I need to be at this skill?", while `priority_signal` answers "how much should this skill matter in the overall score?" A required skill with a large skill gap can pull the stack-fit score down more than a bonus skill with the same gap. A required skill that the user already meets gets full credit for that priority weight.

### Stack-fit score

`_compare_my_stack_to_theirs()` compares the extracted job skills against the user's saved skill grades in `private/my_stack.csv`.

The calculation uses these helper functions:

- `_group_all_substitute_skills()` groups skills that can substitute for each other.
- `_group_single_substitute_skill()` builds one substitute group from a skill and its listed substitutes.
- `_read_my_stack()` loads the user's saved skill grades.
- `_calculate_skill_fit()` calculates the fit contribution for one skill.
- `_grade_required_stack()` estimates the required skill grade from required level and required years.
- `_rank_priority()` weights a skill by extracted priority signal and order of appearance within the same signal group.

Required skill grades are estimated on a `0` to `100` scale:

| Requirement signal | Grade range |
| --- | --- |
| Novice | 0-0 |
| Basic | 0-30 |
| Intermediate | 30-60 |
| Advanced | 60-80 |
| Expert | 80-100 |

Required years are also mapped to a grade range:

| Required years | Grade range |
| --- | --- |
| 0 | 0-0 |
| 1 | 0-31 |
| 2 | 31-54 |
| 3 | 54-70 |
| 4 | 70-81 |
| 5 | 81-89 |
| 6 | 89-95 |
| 7 | 95-100 |

When both required level and required years are present, `_grade_required_stack()` applies them in order to the same required-grade range. It starts with the full `0` to `100` range, narrows that range using the required level, then narrows the result again using required years. The function returns the midpoint of the final narrowed range.

For example, a skill with `required_level="Basic"` first narrows the range from `0-100` to `0-30`. If that same skill also has `required_years=3`, the `3`-year range of `54-70` is applied inside the current `0-30` range, producing an approximate final range of `16-21` and a required grade of `18.5`. Required years are therefore relative to the current narrowed range, not an extra priority boost.

When neither required level nor required years is present, the required grade defaults to `20.0`.

Skill priority is calculated by `_rank_priority()` from the extracted `priority_signal`:

| Priority signal | Base weight before order adjustment |
| --- | --- |
| required | 3.0 |
| highly_preferred | 2.4 |
| preferred | 1.8 |
| bonus | 1.2 |
| not_required | 0.6 |

Skills with the same priority signal are adjusted by order of appearance. Earlier skills keep more of their priority weight; later skills in the same signal group receive a slightly lower weight. Skills with different priority signals do not reduce each other's priority weight.

If the user's grade for a skill is greater than or equal to the required grade, `_calculate_skill_fit()` gives that skill full credit for its priority weight. If the user's grade is below the required grade,
the skill contributes a negative value based on the gap.

The per-skill fit calculation is:

```text
if my_level >= required_grade:
    skill_fit = 100 * priority
else:
    skill_fit = (my_level - required_grade) * priority
```

The final stack-fit score is normalized from a signed range of `-100` to `100` into a public score from `0` to `100`, where `50` is neutral.

### Salary estimate

`_estimate_salary()` estimates salary in one of two ways:

- If the assessment includes `salary_range`, `_estimate_salary_from_range()` uses that range.
- If no salary range is available, `_retrieve_salary_from_matrix()` looks up a fallback salary from `expected_gross_salary_matrix_eur.csv`.

`_estimate_salary_from_range()` sorts the two salary values defensively. If the stack-fit score is below `50`, it returns the lower salary. For scores from `50` to `100`, it interpolates linearly between the
lower and upper salary.

`_retrieve_salary_from_matrix()` tries salary lookup keys in this order:

1. Exact `(role_family, seniority, location_constraint)`
2. Same role and seniority with `Worldwide` location
3. Same role with `Junior` seniority and `Worldwide` location
4. `Mechanical Engineer`, `Junior`, `Worldwide`
5. Minimum salary found in the matrix

### Hard rejection rules

`_validate_seniority_location_salary()` rejects jobs before the final score is returned.

- Seniority is `Lead` or `Principal` and the role is `Software Engineer`, `Backend Engineer`, or `Data Engineer`.
- Location constraint is `Other`.
- Work arrangement is `Onsite`.
- Estimated salary is less than `55000`.

### Final score

If the job passes validation, `evaluate_job_fit()` applies a salary multiplier to the stack-fit score:

```text
salary_multiplier = (salary - 55000) / 55000 / 2 + 1
final_score = int(stack_fit * salary_multiplier)
```

This means salary can raise the final score above the raw stack-fit score. A salary of `55000` uses a multiplier of `1.0` and passes validation because the salary rule is inclusive. A salary of `110000` uses a multiplier of `1.5`. A salary of `165000` uses a multiplier of `2.0`.

### Edge cases

| Edge case | Function involved | Result |
| --- | --- | --- |
| No extracted stack skills | `_compare_my_stack_to_theirs()` | Stack fit is `100` before salary and validation rules are applied. |
| Skill is missing from `private/my_stack.csv` | `_compare_my_stack_to_theirs()` | The user's grade is treated as `0`; the skill may contribute a negative fit value. |
| Skill has substitutes | `_group_all_substitute_skills()`, `_calculate_skill_fit()` | The best-scoring skill in the substitute group is used. |
| Priority signal is missing or unknown | `_rank_priority()` | Raises `KeyError`; no grade is returned. |
| Substitute skill is named but not extracted | `_group_single_substitute_skill()` | Raises `LookupError`; no grade is returned. |
| Skill has no required level and no required years | `_grade_required_stack()` | Required grade defaults to `20.0`. |
| Required level is unknown | `_grade_required_stack()` | Falls back to the full `0-100` range for level. |
| Required years are greater than the mapped range | `_grade_required_stack()` | Uses `100-100`, effectively requiring expert-level experience. |
| Salary range values are reversed | `_estimate_salary_from_range()` | Values are sorted before salary is estimated. |
| Salary range does not contain exactly two values | `_estimate_salary_from_range()` | Raises `ValueError`. |
| No salary range is provided | `_estimate_salary()`, `_retrieve_salary_from_matrix()` | Uses the salary matrix fallback lookup. |
| Salary matrix is empty | `_retrieve_salary_from_matrix()` | Salary becomes `0`, so validation rejects the job and returns `0`. |
| Estimated salary is exactly `55000` | `_validate_seniority_location_salary()` | Passes the salary validation rule because the minimum salary is inclusive. |
| Work arrangement is `Unclear` | `_validate_seniority_location_salary()` | Not rejected by work arrangement because only `Onsite` is rejected. |
| Seniority is `Unclear` | `_validate_seniority_location_salary()` | Not rejected by seniority. |
| Lead or Principal role is `Mechanical Engineer`, `Research Engineer`, or `Other` | `_validate_seniority_location_salary()` | Not rejected by the seniority rule. |

## TODO
Create toml file with:
- City
- Acceptable distance to city for hybrid work
