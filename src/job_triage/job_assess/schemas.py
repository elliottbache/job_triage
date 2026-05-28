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
EngagementType = Literal["Employee", "Freelance", "Contractor", "Unclear", "Other"]
EmploymentType = Literal["FullTime", "PartTime", "Contract", "Unclear", "Other"]
RequiredLevel = Literal["Expert", "Advanced", "Intermediate", "Basic", "Novice"]
WorkArrangement = Literal["Remote", "Hybrid", "Onsite", "Unclear"]
Priority = Literal["required", "highly_preferred", "preferred", "bonus", "not_required"]
# fmt: on


class StackMention(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill: str
    source_text: str
    required_level_text: str | None
    required_years: int | None = Field(ge=1)
    priority_text: str | None
    substitutes: list[str] = Field(
        default_factory=list
    )  # list of possible substitutes if listed as "Skill A or Skill B"


class JobPostExtraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    contact_person: str | None
    contact_data: dict[str, str] | None
    stack_mentions: list[StackMention]
    location_text: list[str]
    engagement_text: list[str]
    employment_text: list[str]
    work_arrangement_text: list[str]
    seniority_text: list[str]
    salary_text: list[str]


class StackAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill: str
    required_level: (
        RequiredLevel | None
    )  # None means the skill is mentioned, but no required depth is stated.
    priority: Priority = Field(
        description=(
            "Classify the skill's importance based on text signals:\n"
            "- 'required': Explicitly mandatory, a must-have, or tied to required minimum years of experience.\n"
            "- 'highly_preferred': Strongly requested or highlighted as a massive advantage (e.g., 'strongly preferred', 'highly desired').\n"
            "- 'preferred': Standard asset or desired qualification (e.g., 'preferred', 'important', 'should have').\n"
            "- 'bonus': Framed as a 'plus', 'nice-to-have', or extra advantage.\n"
            "- 'not_required': Explicitly mentioned but stated as not required (e.g., 'No prior ML knowledge needed')."
        )
    )


class JobPostAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    stack_assessments: list[StackAssessment]
    location_constraint: LocationConstraint
    engagement_type: EngagementType
    employment_type: EmploymentType
    work_arrangement: WorkArrangement
    seniority: (
        SeniorityLevel  # Lead positions will be discarded.  Unclear will be set as Mid.
    )
    salary_range: list[int] | None = Field(min_length=2, max_length=2)
    role_family: RoleFamily
    needs_human_review: list[str] = Field(default_factory=list)


class LLMRunMetadata(BaseModel):
    model_name: str
    prompt_version: str


class JobPostAnalysis(BaseModel):
    extracted: JobPostExtraction
    assessment: JobPostAssessment
    metadata: LLMRunMetadata | None = None
