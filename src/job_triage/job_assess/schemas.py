from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# fmt: off
LocationConstraint = Literal[
    "US", "EU", "Worldwide", "Other", "Canada", "UAE", "Thailand", "Costa Rica", 
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", 
    "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", 
    "Estonia", "Finland", "France", "Georgia", "Germany", "Greece", "Hungary", 
    "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", 
    "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", 
    "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", 
    "Switzerland", "Turkey","Ukraine", "United Kingdom", "Vatican City"
]
SkillPriority = Literal["High", "Mid", "Low"]
SeniorityLevel = Literal["Junior", "Mid", "Senior", "Lead", "Unclear"]
RoleFamily = Literal[
    "Software Engineer", "Backend Engineer", "Data Engineer", "Research Engineer", 
    "Mechanical Engineer", "Other"
]
WorkAuthorization = Literal[
    "US Work Authorization", "EU Work Authorization", "Other", "Unclear"
]
BaseResume = Literal["backend", "cfd", "research"]
# fmt: on


class StackMention(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill: str
    source_text: str
    order_of_appearance: int
    explicit_required_level: (
        Literal["Expert", "Advanced", "Intermediate", "Basic"] | None
    )  # None refers to no mention to the level, but the skill is mentioned in the job offer.
    explicit_years: int | None
    priority_signal: str | None


class JobPostExtraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    contact_person: str | None
    contact_data: dict[str, str] | None
    stack_mentions: list[StackMention]
    unclear_points: list[str] = Field(default_factory=list)


class JobPostAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_priority: dict[str, SkillPriority] = Field(default_factory=dict)
    location_constraints: LocationConstraint  # Other (e.g. LATAM) are discarded.
    required_work_authorization: WorkAuthorization  # this is based on the most explicit evidence, but can be overridden to "Unclear" if there are contradictions or lack of clarity.
    seniority: (
        SeniorityLevel  # Lead positions will be discarded.  Unclear will be set as Mid.
    )
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
