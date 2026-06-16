from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

from job_triage._helpers import CURRENCY_EUR_RATES, SALARY_PERIOD_MULTIPLIERS


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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_yearly_salary_eur(self) -> float | None:
        """Return the maximum yearly base salary converted to EUR, when available."""
        # Return None if there is no compensation metadata present
        if not self.compensation or not self.compensation.summary_components:
            return None

        for component in self.compensation.summary_components:
            # Only perform processing if it's explicitly a base 'Salary' component
            if component.compensation_type != "Salary":
                continue

            # Return None if critical information within the salary block is missing
            if (
                component.min_value is None
                or component.currency_code is None
                or component.interval is None
            ):
                return None

            # Standardize casings for clean dictionary lookup keys
            currency_key = component.currency_code.upper().strip()
            interval_key = component.interval.lower().strip()

            currency_rate = CURRENCY_EUR_RATES.get(currency_key)
            period_multiplier = SALARY_PERIOD_MULTIPLIERS.get(interval_key)

            # Return None if currency or interval period type is unlisted
            if currency_rate is None or period_multiplier is None:
                return None

            max_value = (
                component.max_value
                if component.max_value is not None
                else component.min_value
            )
            if max_value is None:
                return None
            return round(max_value * period_multiplier / currency_rate)

        return None


class ParsedAshbyJob(BaseModel):
    """Original Ashby payload paired with its validated job model."""

    raw_payload: dict[str, Any]
    job: AshbyJob
