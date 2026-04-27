from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# fmt: off
LocationConstraint = Literal[
    "US", "EU", "Worldwide", "Other", "Canada", "UAE", "Thailand", "Costa Rica", 
    "Andorra", "Austria", "Belgium", "Croatia", "Czechia", "Denmark", 
    "Estonia", "Finland", "France", "Germany", "Hungary", 
    "Iceland", "Ireland", "Italy", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Monaco", 
    "Netherlands", "Norway", "Poland", "Portugal",
    "San Marino", "Slovakia", "Slovenia", "Spain", "Sweden", 
    "Switzerland", "United Kingdom"
]
SeniorityLevel = Literal["Junior", "Mid", "Senior", "Lead", "Principal", "Unclear"]
RoleFamily = Literal[
    "Software Engineer", "Backend Engineer", "Data Engineer", "Research Engineer", 
    "Mechanical Engineer", "Other"
]
BaseResume = Literal["backend", "cfd", "research"]
RequiredLevel = Literal["Expert", "Advanced", "Intermediate", "Basic"]
PriorityLevel = Literal["High", "Mid", "Low"]
# fmt: on


class StackMention(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill: str
    source_text: str
    order_of_appearance: int = Field(gt=0)
    required_level: (
        RequiredLevel | None
    )  # None refers to no mention to the level, but the skill is mentioned in the job offer.
    required_years: int | None = Field(ge=1)
    priority_signal: (
        str | None
    )  # e.g. required, a plus, nice-to-have, important are later transformed into high, mid, or low in assessment
    substitutes: list[str] = Field(
        default_factory=list
    )  # list of possible substitutes if listed as "Skill A or Skill B"


class JobPostExtraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    contact_person: str | None
    contact_data: dict[str, str] | None
    stack_mentions: list[StackMention]
    unclear_points: list[str] = Field(default_factory=list)


class SkillPriorityItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill: str
    priority: PriorityLevel


class JobPostAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_priority: list[SkillPriorityItem]
    location_constraints: LocationConstraint  # Other (e.g. LATAM) are discarded.
    seniority: (
        SeniorityLevel  # Lead positions will be discarded.  Unclear will be set as Mid.
    )
    salary_range: list[int] | None = Field(min_length=2, max_length=2)
    role_family: RoleFamily
    recommended_base_resume_name: list[BaseResume]
    fit_summary: str
    needs_human_review: list[str] = Field(default_factory=list)


class LLMRunMetadata(BaseModel):
    model_name: str
    prompt_version: str
    is_retry: bool


class ExtractionResult(BaseModel):
    extraction: JobPostExtraction
    metadata: LLMRunMetadata


class AssessmentResult(BaseModel):
    assessment: JobPostAssessment
    metadata: LLMRunMetadata
