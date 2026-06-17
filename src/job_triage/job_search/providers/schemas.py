from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Location Elements ---
class PostalAddress(BaseModel):
    address_locality: str | None = Field(None, alias="addressLocality")
    address_region: str | None = Field(None, alias="addressRegion")
    address_country: str | None = Field(None, alias="addressCountry")


class AddressWrapper(BaseModel):
    postal_address: PostalAddress | None = Field(None, alias="postalAddress")


class SecondaryLocation(BaseModel):
    location: str
    address: PostalAddress | None = None


# --- Compensation Sub-Structures ---
class CompensationComponent(BaseModel):
    id: str | None = None  # Summary components don't have IDs
    summary: str | None = None
    compensation_type: str = Field(..., alias="compensationType")
    interval: str
    currency_code: str | None = Field(None, alias="currencyCode")
    min_value: float | None = Field(None, alias="minValue")
    max_value: float | None = Field(None, alias="maxValue")


class CompensationTier(BaseModel):
    id: str
    tier_summary: str | None = Field(None, alias="tierSummary")
    title: str | None
    additional_information: Any | None = Field(None, alias="additionalInformation")
    components: list[CompensationComponent] = Field(default_factory=list)


class JobCompensation(BaseModel):
    compensation_tier_summary: str | None = Field(None, alias="compensationTierSummary")
    scrapeable_compensation_salary_summary: str | None = Field(
        None, alias="scrapeableCompensationSalarySummary"
    )
    compensation_tiers: list[CompensationTier] = Field(
        default_factory=list, alias="compensationTiers"
    )
    summary_components: list[CompensationComponent] = Field(
        default_factory=list, alias="summaryComponents"
    )


class AshbyJob(BaseModel):
    id: str
    title: str
    location: str
    secondary_locations: list[SecondaryLocation] = Field(
        default_factory=list, alias="secondaryLocations"
    )
    department: str | None = None
    team: str | None = None
    is_listed: bool = Field(..., alias="isListed")
    is_remote: bool | None = Field(None, alias="isRemote")
    workplace_type: str | None = Field(None, alias="workplaceType")
    description_html: str | None = Field(None, alias="descriptionHtml")
    description_plain: str | None = Field(None, alias="descriptionPlain")
    published_at: datetime | None = Field(None, alias="publishedAt")
    updated_at: datetime | None = Field(None, alias="updatedAt")
    employment_type: str | None = Field(None, alias="employmentType")
    address: AddressWrapper | None = None
    job_url: str = Field(..., alias="jobUrl")
    apply_url: str | None = Field(None, alias="applyUrl")
    compensation: JobCompensation | None = None


class ParsedAshbyJob(BaseModel):
    """Original Ashby payload paired with its validated job model."""

    raw_payload: dict[str, Any]
    job: AshbyJob
