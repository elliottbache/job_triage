import json
import logging
import time
from collections.abc import Callable
from datetime import date, timedelta
from hashlib import sha256
from os import getenv
from typing import Any, TypedDict
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from requests import HTTPError, RequestException, Timeout
from sqlalchemy import and_, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    wait_exponential,
)
from tenacity.wait import wait_base

from job_triage._helpers import (
    DEFAULT_MINIMUM_SALARY,
    ROOT_DIR,
)
from job_triage.db.db_access import get_session
from job_triage.db.models import ATSBoard, RawJob
from job_triage.job_search.providers.schemas import AshbyJob, ParsedAshbyJob
from job_triage.schemas import JobPostSource

_DOTENV_PATH = ROOT_DIR / ".env"
_DEFAULT_TIMEOUT = 15
_DEFAULT_RATE_LIMIT_DELAY = 1.0
_DEFAULT_BACKOFF = wait_exponential(multiplier=1, min=2, max=32)
_DEFAULT_MAX_PAGES = 9
_DEFAULT_RESULTS_PER_PAGE = 20
_DEFAULT_KEYWORDS = {"python", "backend", "software engineer", "developer"}
_DEFAULT_SEARCH_PHRASE = (
    "site:jobs.ashbyhq.com " + " ".join(_DEFAULT_KEYWORDS) + " remote"
)
_DEFAULT_DELAY = 2.0
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
logger = logging.getLogger(__name__)


class _JobPostSourceDict(TypedDict):
    title: str
    company: str
    job_description: str
    date_posted: str
    source_url: str
    metadata_text: dict[str, str]


def extract_ashby_listings(
    *, keywords: set[str] = _DEFAULT_KEYWORDS
) -> list[JobPostSource]:
    """Persist discovered Ashby boards and return filtered normalized job sources."""
    # 1. Search web for Ashby board URLs and extract company slugs:
    slugs = _discover_ashby_slugs()

    # 2. Save new slugs to db
    for slug in slugs:
        insert_stmt = (
            sqlite_insert(ATSBoard)
            .values(provider="Ashby", board_slug=slug)
            .on_conflict_do_nothing(index_elements=["provider", "board_slug"])
        )
        with get_session() as session:
            try:
                session.execute(insert_stmt)
                session.commit()
            except IntegrityError:
                session.rollback()
                raise

    # 3. Read all slugs from db
    select_stmt = select(ATSBoard).where(ATSBoard.provider == "Ashby")
    with get_session() as session:
        boards = session.execute(select_stmt).scalars().all()

    board_ids_by_slug = {board.board_slug: board.id for board in boards}

    # 4. For each slug:
    #      GET https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true
    job_post_sources = list()
    for slug, ats_board_id in board_ids_by_slug.items():
        try:
            raw_jobs = _retrieve_ashby_jobs_for_company(slug)
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.warning(
                "Exception raised for %s with status code %s: %s",
                slug,
                status_code,
                exc,
            )
            continue

        # 5. For each returned job:
        #      filter title + descriptionPlain, salary, remote, publish date
        for raw_job in raw_jobs:
            job = raw_job.job
            if not _filter_ashby_job(job, keywords=keywords):
                continue

            # 6. Convert matching jobs directly to JobPostSource
            job_post_sources.append(_parse_job_post_source(job, slug=slug))

            # 7. Add job to db
            _sync_raw_job_atomic(ats_board_id=ats_board_id, parsed_job=raw_job)

    return job_post_sources


def _discover_ashby_slugs(
    query: str = _DEFAULT_SEARCH_PHRASE,
    *,
    max_pages: int = _DEFAULT_MAX_PAGES,
    results_per_page: int = _DEFAULT_RESULTS_PER_PAGE,
) -> set[str]:
    """Discover unique Ashby board slugs for the provided search query."""
    urls = _search_brave(query, max_pages=max_pages, results_per_page=results_per_page)

    slugs = set()
    for url in urls:
        slug = _extract_ashby_slug(url)
        if slug is not None:
            slugs.add(slug)

    return slugs


def _search_brave(query: str, *, max_pages: int, results_per_page: int) -> list[str]:
    """Return result URLs from Brave Search."""
    api_key = getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("No Brave search API key is defined.")
    base_url = "https://api.search.brave.com/res/v1/web/search"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    urls: list[str] = []

    with requests.Session() as client:
        for offset in range(max_pages):
            if len(urls) >= results_per_page * max_pages:
                break

            params: dict[str, str | int] = {
                "q": query,
                "count": min(
                    results_per_page, results_per_page * max_pages - len(urls)
                ),
                "offset": offset,
                "result_filter": "web",
            }
            response_data = _safe_brave_request(
                base_url, client=client, headers=headers, params=params
            )

            results = response_data.get("web", {}).get("results", [])
            if not results:
                break

            for result in results:
                url = result.get("url")
                if url and url not in urls:
                    urls.append(url)

    return urls


def _wait_for_brave_retry(retry_state: RetryCallState) -> float:
    """Return Brave's rate-limit reset delay when available, else backoff."""
    if retry_state.outcome is None:
        return _DEFAULT_RATE_LIMIT_DELAY

    exc = retry_state.outcome.exception()

    if (
        isinstance(exc, HTTPError)
        and exc.response is not None
        and exc.response.status_code == 429
    ):
        reset_after = exc.response.headers.get("X-RateLimit-Reset")
        if reset_after is not None:
            return float(reset_after)

    return _DEFAULT_BACKOFF(retry_state)


def _stop_after_attempts_by_error(retry_state: RetryCallState) -> bool:
    """Dynamically drops or extends retry limits based on the specific exception."""
    if retry_state.outcome is None:
        return retry_state.attempt_number >= 2

    exc = retry_state.outcome.exception()

    if isinstance(exc, ValueError):
        return retry_state.attempt_number >= 1

    if isinstance(exc, Timeout):
        return retry_state.attempt_number >= 6

    if isinstance(exc, RequestException):
        response = exc.response
        if response is None:
            return retry_state.attempt_number >= 6

        status_code = response.status_code

        if status_code in {408, 429} or status_code >= 500:
            return retry_state.attempt_number >= 6

        if status_code in {400, 401, 402, 403, 404, 413, 422}:
            return retry_state.attempt_number >= 1

        return retry_state.attempt_number >= 2

    # 3. Default fallback for other retryable errors
    return retry_state.attempt_number >= 1


def _request_retry(
    *, wait: wait_base | Callable[[RetryCallState], float] = _DEFAULT_BACKOFF
) -> Callable[..., Any]:
    """Build a requests retry decorator with configurable wait behavior."""
    return retry(
        stop=_stop_after_attempts_by_error,  # Dynamically change the max attempts based on the exception type
        wait=wait,
        retry=retry_if_exception_type(RequestException),
        reraise=True,  # Throw original exception if all fail
    )


@_request_retry(wait=_wait_for_brave_retry)
def _safe_brave_request(
    url: str,
    *,
    client: requests.Session,
    headers: dict[str, str],
    params: dict[str, str | int],
) -> dict[str, Any]:
    """Call Brave Search and return the decoded JSON response."""
    response = client.get(url, headers=headers, params=params, timeout=_DEFAULT_TIMEOUT)

    # Pacing to respect 50 requests per second window
    time.sleep(0.02)

    response.raise_for_status()

    return response.json()


def _extract_ashby_slug(url: str) -> str | None:
    """Extract Ashby job-board slug from a jobs.ashbyhq.com URL."""
    parsed = urlparse(url)

    if parsed.netloc != "jobs.ashbyhq.com":
        return None

    parts = [part for part in parsed.path.split("/") if part]

    if not parts:
        return None

    return parts[0]


@_request_retry()
def _retrieve_ashby_jobs_for_company(slug: str) -> list[ParsedAshbyJob]:
    """Retrieve Ashby jobs while preserving each original provider payload."""
    api_url = (
        "https://api.ashbyhq.com/posting-api/job-board/"
        f"{slug}?includeCompensation=true"
    )

    response = requests.get(api_url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()

    jobs = response.json().get("jobs", [])
    return [
        ParsedAshbyJob(raw_payload=job, job=AshbyJob.model_validate(job))
        for job in jobs
    ]


def _filter_ashby_job(
    job: AshbyJob, *, keywords: set[str] = _DEFAULT_KEYWORDS, maximum_days_ago: int = 14
) -> bool:
    """Return whether an Ashby job matches remote, salary, keyword, and date rules."""
    if not job.is_remote or job.workplace_type == "OnSite":
        return False

    full_description = (
        " ".join([job.title or "", job.description_plain or ""]).strip().casefold()
    )
    if not any(keyword in full_description for keyword in keywords):
        return False

    max_salary = job.max_yearly_salary_eur
    if max_salary is not None and max_salary < DEFAULT_MINIMUM_SALARY:
        return False

    todays_date = date.today()
    if job.updated_at:
        posted_date = job.updated_at.date()
    elif job.published_at:
        posted_date = job.published_at.date()
    else:
        posted_date = todays_date
    return not posted_date < todays_date - timedelta(days=maximum_days_ago)


def _sync_raw_job_atomic(parsed_job: ParsedAshbyJob, ats_board_id: int) -> None:
    """Insert or refresh a raw Ashby job row without rewriting unchanged content."""
    job = parsed_job.job
    source_url = job.apply_url or job.job_url
    external_id = (
        job.id or _extract_ashby_id(job.job_url) or _extract_ashby_id(source_url)
    )
    provider_payload_json = json.dumps(
        parsed_job.raw_payload, sort_keys=True, separators=(",", ":")
    )
    incoming_hash = sha256(provider_payload_json.encode("utf-8")).hexdigest()
    posted_at = job.updated_at or job.published_at
    date_posted = posted_at.date() if posted_at else date.today()
    raw_job_values = {
        "source_url": source_url,
        "ats_board_id": ats_board_id,
        "external_id": external_id,
        "title": job.title,
        "date_posted": date_posted,
        "provider_payload_json": provider_payload_json,
        "content_hash": incoming_hash,
    }
    with get_session() as session:
        # 1. Attempt an ATOMIC insert. If EITHER constraint is violated,
        # SQLite will silently ignore the insert without crashing.
        insert_stmt = (
            sqlite_insert(RawJob).values(**raw_job_values).on_conflict_do_nothing()
        )
        result = session.execute(insert_stmt)

        # 2. Check if a new row was actually inserted
        if result.rowcount > 0:
            session.commit()
            return

        # 3. If rowcount is 0, the row already exists (Constraint hit).
        # Now we run an ATOMIC update that ONLY fires if the hash has changed.
        update_stmt = (
            update(RawJob)
            .where(
                and_(
                    # Match whichever row caused the conflict
                    or_(
                        RawJob.source_url == source_url,
                        and_(
                            RawJob.ats_board_id == ats_board_id,
                            RawJob.external_id == external_id,
                        ),
                    ),
                    # CRITICAL: Only match if the stored hash is different from the incoming one
                    RawJob.content_hash != incoming_hash,
                )
            )
            .values(**raw_job_values)
        )
        session.execute(update_stmt)
        session.commit()


def _extract_ashby_id(url: str) -> str | None:
    """Extract the job identifier from a jobs.ashbyhq.com posting URL."""
    parsed = urlparse(url)

    if parsed.netloc != "jobs.ashbyhq.com":
        return None

    parts = [part for part in parsed.path.split("/") if part]

    if len(parts) < 2:
        return None

    return parts[1]


def _parse_job_post_source(job: AshbyJob, *, slug: str) -> JobPostSource:
    job_post_source_dict: _JobPostSourceDict = {
        "title": job.title,
        "company": slug,
        "job_description": job.description_plain or "",
        "date_posted": str(job.updated_at or job.published_at or date.today()),
        "source_url": job.apply_url or job.job_url or "",
        "metadata_text": dict(),
    }

    keys = [
        "location",
        "employment_type",
        "work_arrangement",
        "is_remote",
        "alternative_url",
        "max_salary",
        "compensation",
    ]
    if job.compensation and job.compensation.compensation_tier_summary:
        compensation = job.compensation.compensation_tier_summary
    else:
        compensation = None
    values = [
        job.location,
        job.employment_type,
        job.workplace_type,
        str(job.is_remote),
        job.job_url,
        job.max_yearly_salary_eur,
        compensation,
    ]
    for key, value in zip(keys, values, strict=True):
        if value:
            job_post_source_dict["metadata_text"][key] = str(value)

    return JobPostSource.model_validate(job_post_source_dict)


if __name__ == "__main__":
    """urls = _search_brave(
        "site:jobs.ashbyhq.com software engineer python remote",
        max_results=100,
    )

    for url in urls:
        print(url)"""

    # extract_ashby_listings()

    # jobs = _retrieve_ashby_jobs_for_company("scalera")

    job_listings = extract_ashby_listings()
    pass
