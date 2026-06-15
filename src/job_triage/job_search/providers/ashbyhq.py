import time
from collections.abc import Callable
from datetime import date, timedelta
from os import getenv
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from requests import HTTPError, RequestException, Timeout
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
from job_triage.job_search.providers.schemas import AshbyJob

_DOTENV_PATH = ROOT_DIR / ".env"
_DEFAULT_TIMEOUT = 15
_DEFAULT_RATE_LIMIT_DELAY = 1.0
_DEFAULT_BACKOFF = wait_exponential(multiplier=1, min=2, max=32)
_DEFAULT_MAX_PAGES = 10
_DEFAULT_KEYWORDS = {"python", "backend", "software engineer", "developer"}
_DEFAULT_SEARCH_PHRASE = (
    "site:jobs.ashbyhq.com " + " ".join(_DEFAULT_KEYWORDS) + " remote"
)
_DEFAULT_DELAY = 2.0
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


def extract_ashby_listings() -> None:
    """Discover Ashby boards and retrieve their job listings."""
    # 1. Search web for Ashby board URLs and extract company slugs:
    slugs = _discover_ashby_slugs(
        _DEFAULT_SEARCH_PHRASE, max_results=_DEFAULT_MAX_PAGES
    )
    print(slugs)

    # 2. For each slug:
    #      GET https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true
    for slug in slugs:
        _jobs = _retrieve_ashby_jobs_for_company(slug)

    # 4. For each returned job:
    #      filter title + descriptionPlain + location locally

    # 5. Convert matching jobs directly to JobPostSource
    # map_ashby_job_to_source(job, company_slug)
    #     -> JobPostSource

    # return all listings
    pass


def _discover_ashby_slugs(
    query: str,
    *,
    max_results: int = _DEFAULT_MAX_PAGES,
) -> set[str]:
    """Discover unique Ashby board slugs from Brave Search results."""
    urls = _search_brave(query, max_results=max_results)

    slugs = set()
    for url in urls:
        slug = _extract_ashby_slug(url)
        if slug is not None:
            slugs.add(slug)

    return slugs


def _search_brave(query: str, *, max_results: int) -> list[str]:
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
        for offset in range(_DEFAULT_MAX_PAGES):
            if len(urls) >= max_results:
                break

            params: dict[str, str | int] = {
                "q": query,
                "count": min(20, max_results - len(urls)),
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
def _retrieve_ashby_jobs_for_company(slug: str) -> list[AshbyJob]:
    """Retrieve and validate jobs from an Ashby company job board."""
    api_url = (
        "https://api.ashbyhq.com/posting-api/job-board/"
        f"{slug}?includeCompensation=true"
    )

    response = requests.get(api_url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()

    jobs = response.json().get("jobs", [])
    return [AshbyJob.model_validate(job) for job in jobs]


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
    published_at = job.published_at.date() if job.published_at else todays_date
    return not published_at < todays_date - timedelta(days=maximum_days_ago)


if __name__ == "__main__":
    """urls = _search_brave(
        "site:jobs.ashbyhq.com software engineer python remote",
        max_results=100,
    )

    for url in urls:
        print(url)"""

    # extract_ashby_listings()

    jobs = _retrieve_ashby_jobs_for_company("scalera")
    pass
