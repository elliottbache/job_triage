import json
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from requests import HTTPError, RequestException, Timeout
from sqlalchemy.exc import IntegrityError
from tenacity import wait_none

from job_triage._helpers import DEFAULT_MINIMUM_SALARY
from job_triage.job_search.providers import ashbyhq
from job_triage.job_search.providers.schemas import ParsedAshbyJob


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.raise_for_status = MagicMock(side_effect=self._raise_for_status)

    def json(self) -> dict[str, object]:
        return self._payload

    def _raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"Status {self.status_code}", response=self)


def _retry_state(attempt_number: int, exception: BaseException | None):
    outcome = None
    if exception is not None:
        outcome = SimpleNamespace(exception=lambda: exception)

    return SimpleNamespace(attempt_number=attempt_number, outcome=outcome)


def _http_error(status_code: int, headers: dict[str, str] | None = None) -> HTTPError:
    response = _FakeResponse(status_code=status_code, payload={}, headers=headers)
    return HTTPError(f"Status {status_code}", response=response)


def _request_exception_without_response() -> RequestException:
    return RequestException("connection failed")


def _timeout_error() -> Timeout:
    return Timeout("timed out")


def _ashby_job_payload(
    title: str = "Backend Engineer",
    **overrides: object,
) -> dict[str, object]:
    payload = {
        "id": "9a64ae0e-48c1-48b8-870d-35894530090d",
        "title": title,
        "location": "Remote",
        "isListed": True,
        "isRemote": True,
        "jobUrl": "https://jobs.ashbyhq.com/scalera/backend-engineer",
        "applyUrl": "https://jobs.ashbyhq.com/scalera/backend-engineer/application",
        "descriptionPlain": "Build Python services.",
        "employmentType": "FullTime",
    }
    payload.update(overrides)
    return payload


def _ashby_job(**overrides: object) -> ashbyhq.AshbyJob:
    return ashbyhq.AshbyJob.model_validate(_ashby_job_payload(**overrides))


def _published_at(days_ago: int = 0) -> str:
    published_date = date.today() - timedelta(days=days_ago)
    return datetime.combine(published_date, time.min).isoformat()


def _updated_at(days_ago: int = 0) -> str:
    updated_date = date.today() - timedelta(days=days_ago)
    return datetime.combine(updated_date, time.min).isoformat()


def _compensation_payload(
    *,
    min_value: float = 10_000,
    max_value: float | None = None,
    currency_code: str = "EUR",
    interval: str = "year",
) -> dict[str, object]:
    return {
        "summaryComponents": [
            {
                "compensationType": "Salary",
                "interval": interval,
                "currencyCode": currency_code,
                "minValue": min_value,
                "maxValue": max_value,
            }
        ]
    }


class TestExtractAshbyListings:
    def test_returns_filtered_job_post_sources_for_discovered_slugs(self) -> None:
        session = MagicMock()
        session.execute.return_value.scalars.return_value.all.return_value = [
            SimpleNamespace(id=42, board_slug="scalera")
        ]
        old_published_at = _published_at(days_ago=5)
        recent_updated_at = _updated_at(days_ago=1)
        raw_payload = _ashby_job_payload(
            title="Backend Engineer",
            descriptionPlain="Build Python services.",
            publishedAt=old_published_at,
            updatedAt=recent_updated_at,
            compensation=_compensation_payload(max_value=DEFAULT_MINIMUM_SALARY),
        )

        with (
            patch(
                "job_triage.job_search.providers.ashbyhq._discover_ashby_slugs",
                return_value={"scalera"},
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq.get_session",
                return_value=session,
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq._retrieve_ashby_jobs_for_company",
                return_value=[
                    ParsedAshbyJob(
                        raw_payload=raw_payload,
                        job=ashbyhq.AshbyJob.model_validate(raw_payload),
                    )
                ],
            ),
        ):
            result = ashbyhq.extract_ashby_listings(keywords={"python"})

        assert len(result) == 1
        assert result[0].title == "Backend Engineer"
        assert result[0].company == "scalera"
        assert result[0].date_posted == str(
            ashbyhq.AshbyJob.model_validate(
                _ashby_job_payload(updatedAt=recent_updated_at)
            ).updated_at
        )
        assert result[0].metadata_text["max_salary"] == str(DEFAULT_MINIMUM_SALARY)
        assert session.add.call_count == 2
        raw_job = session.add.call_args_list[1].args[0]
        assert raw_job.ats_board_id == 42
        assert json.loads(raw_job.raw_json) == raw_payload
        assert session.commit.call_count == 2

    def test_ignores_duplicate_board_insert_errors(self) -> None:
        session = MagicMock()
        session.commit.side_effect = IntegrityError(
            "insert",
            {},
            Exception(
                "UNIQUE constraint failed: ats_boards.provider, ats_boards.board_slug"
            ),
        )

        with (
            patch(
                "job_triage.job_search.providers.ashbyhq._discover_ashby_slugs",
                return_value={"scalera"},
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq.get_session",
                return_value=session,
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq._retrieve_ashby_jobs_for_company",
                return_value=[],
            ),
        ):
            result = ashbyhq.extract_ashby_listings(keywords={"python"})

        assert result == []
        session.add.assert_called_once()
        session.commit.assert_called_once_with()
        session.rollback.assert_called_once_with()

    def test_reraises_unexpected_integrity_errors(self) -> None:
        session = MagicMock()
        error = IntegrityError("insert", {}, Exception("foreign key mismatch"))
        session.commit.side_effect = error

        with (
            patch(
                "job_triage.job_search.providers.ashbyhq._discover_ashby_slugs",
                return_value={"scalera"},
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq.get_session",
                return_value=session,
            ),
            pytest.raises(IntegrityError) as exc_info,
        ):
            ashbyhq.extract_ashby_listings(keywords={"python"})

        assert exc_info.value is error
        session.rollback.assert_called_once_with()


class TestDiscoverAshbySlugs:
    def test_returns_unique_slugs_from_brave_urls(self) -> None:
        with patch(
            "job_triage.job_search.providers.ashbyhq._search_brave",
            return_value=[
                "https://jobs.ashbyhq.com/linear/software-engineer",
                "https://jobs.ashbyhq.com/ramp",
                "https://jobs.ashbyhq.com/linear/data-engineer",
                "https://example.com/not-an-ashby-board",
            ],
        ) as mock_search:
            result = ashbyhq._discover_ashby_slugs(
                "python remote",
                max_pages=1,
                results_per_page=5,
            )

        assert result == {"linear", "ramp"}
        mock_search.assert_called_once_with(
            "python remote",
            max_pages=1,
            results_per_page=5,
        )


class TestSearchBrave:
    def test_raises_when_api_key_is_missing(self) -> None:
        with (
            patch(
                "job_triage.job_search.providers.ashbyhq.getenv",
                return_value=None,
            ),
            pytest.raises(ValueError, match="No Brave search API key"),
        ):
            ashbyhq._search_brave(
                "python remote",
                max_pages=1,
                results_per_page=10,
            )

    def test_returns_unique_urls_from_paginated_results(self) -> None:
        client = MagicMock()
        client_context = MagicMock()
        client_context.__enter__.return_value = client
        client_context.__exit__.return_value = None

        with (
            patch(
                "job_triage.job_search.providers.ashbyhq.getenv",
                return_value="brave-key",
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq.requests.Session",
                return_value=client_context,
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq._safe_brave_request",
                side_effect=[
                    {
                        "web": {
                            "results": [
                                {"url": "https://jobs.ashbyhq.com/linear"},
                                {"url": "https://example.com/search-result"},
                            ]
                        }
                    },
                    {
                        "web": {
                            "results": [
                                {"url": "https://jobs.ashbyhq.com/linear"},
                                {"url": "https://jobs.ashbyhq.com/ramp"},
                            ]
                        }
                    },
                ],
            ) as mock_request,
        ):
            result = ashbyhq._search_brave(
                "python remote",
                max_pages=2,
                results_per_page=2,
            )

        assert result == [
            "https://jobs.ashbyhq.com/linear",
            "https://example.com/search-result",
            "https://jobs.ashbyhq.com/ramp",
        ]
        assert mock_request.call_count == 2
        assert mock_request.call_args_list[0].kwargs["params"] == {
            "q": "python remote",
            "count": 2,
            "offset": 0,
            "result_filter": "web",
        }
        assert mock_request.call_args_list[1].kwargs["params"] == {
            "q": "python remote",
            "count": 2,
            "offset": 1,
            "result_filter": "web",
        }

    def test_stops_when_page_has_no_results(self) -> None:
        client_context = MagicMock()
        client_context.__enter__.return_value = MagicMock()
        client_context.__exit__.return_value = None

        with (
            patch(
                "job_triage.job_search.providers.ashbyhq.getenv",
                return_value="brave-key",
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq.requests.Session",
                return_value=client_context,
            ),
            patch(
                "job_triage.job_search.providers.ashbyhq._safe_brave_request",
                return_value={"web": {"results": []}},
            ) as mock_request,
        ):
            result = ashbyhq._search_brave(
                "python remote",
                max_pages=9,
                results_per_page=20,
            )

        assert result == []
        mock_request.assert_called_once()


class TestSafeBraveRequest:
    def test_returns_json_after_successful_request(self) -> None:
        client = MagicMock()
        response = _FakeResponse(status_code=200, payload={"web": {"results": []}})
        client.get.return_value = response

        with patch("job_triage.job_search.providers.ashbyhq.time.sleep") as mock_sleep:
            result = ashbyhq._safe_brave_request(
                "https://api.search.brave.com/res/v1/web/search",
                client=client,
                headers={"X-Subscription-Token": "brave-key"},
                params={"q": "python remote", "count": 10},
            )

        assert result == {"web": {"results": []}}
        response.raise_for_status.assert_called_once_with()
        mock_sleep.assert_called_once_with(0.02)
        client.get.assert_called_once_with(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": "brave-key"},
            params={"q": "python remote", "count": 10},
            timeout=ashbyhq._DEFAULT_TIMEOUT,
        )

    def test_retries_after_rate_limit_reset(self) -> None:
        client = MagicMock()
        rate_limited_response = _FakeResponse(
            status_code=429,
            payload={},
            headers={"X-RateLimit-Reset": "3"},
        )
        success_response = _FakeResponse(
            status_code=200,
            payload={"web": {"results": [{"url": "https://jobs.ashbyhq.com/ramp"}]}},
        )
        client.get.side_effect = [rate_limited_response, success_response]

        with patch("job_triage.job_search.providers.ashbyhq.time.sleep") as mock_sleep:
            result = ashbyhq._safe_brave_request(
                "https://api.search.brave.com/res/v1/web/search",
                client=client,
                headers={"X-Subscription-Token": "brave-key"},
                params={"q": "python remote", "count": 10},
            )

        assert result == {
            "web": {"results": [{"url": "https://jobs.ashbyhq.com/ramp"}]}
        }
        assert client.get.call_count == 2
        assert mock_sleep.call_args_list[0].args == (0.02,)
        assert mock_sleep.call_args_list[1].args == (3.0,)
        assert mock_sleep.call_args_list[2].args == (0.02,)

    @pytest.mark.parametrize(
        ("exception_factory", "expected_attempts"),
        [
            (lambda: _http_error(400), 1),
            (lambda: _http_error(408), 6),
            (lambda: _http_error(418), 2),
            (lambda: _http_error(429), 6),
            (lambda: _http_error(503), 6),
            (_request_exception_without_response, 6),
            (_timeout_error, 6),
        ],
    )
    def test_retry_decorator_stops_at_exception_specific_attempt_limit(
        self, exception_factory, expected_attempts
    ) -> None:
        client = MagicMock()
        errors = [exception_factory() for _ in range(expected_attempts)]
        client.get.side_effect = errors

        with pytest.raises(type(errors[-1])):
            ashbyhq._safe_brave_request.retry_with(wait=wait_none())(
                "https://api.search.brave.com/res/v1/web/search",
                client=client,
                headers={"X-Subscription-Token": "brave-key"},
                params={"q": "python remote", "count": 10},
            )

        assert client.get.call_count == expected_attempts

    def test_retry_decorator_does_not_retry_unhandled_exceptions(self) -> None:
        client = MagicMock()
        client.get.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            ashbyhq._safe_brave_request.retry_with(wait=wait_none())(
                "https://api.search.brave.com/res/v1/web/search",
                client=client,
                headers={"X-Subscription-Token": "brave-key"},
                params={"q": "python remote", "count": 10},
            )

        client.get.assert_called_once()


class TestParseRawJob:
    def test_preserves_original_provider_payload_as_raw_json(self) -> None:
        raw_payload = _ashby_job_payload(
            publishedAt=_published_at(days_ago=2),
            providerOnlyField={"nested": ["kept"]},
        )
        parsed_job = ParsedAshbyJob(
            raw_payload=raw_payload,
            job=ashbyhq.AshbyJob.model_validate(raw_payload),
        )

        result = ashbyhq._parse_raw_job(ats_board_id=42, parsed_job=parsed_job)

        assert result.ats_board_id == 42
        assert result.source_url == (
            "https://jobs.ashbyhq.com/scalera/backend-engineer/application"
        )
        assert result.external_id == "9a64ae0e-48c1-48b8-870d-35894530090d"
        assert result.date_posted == date.today() - timedelta(days=2)
        assert json.loads(result.raw_json) == raw_payload
        assert (
            result.content_hash == sha256(result.raw_json.encode("utf-8")).hexdigest()
        )

    def test_uses_updated_at_for_date_posted_when_available(self) -> None:
        raw_payload = _ashby_job_payload(
            publishedAt=_published_at(days_ago=10),
            updatedAt=_updated_at(days_ago=1),
        )
        parsed_job = ParsedAshbyJob(
            raw_payload=raw_payload,
            job=ashbyhq.AshbyJob.model_validate(raw_payload),
        )

        result = ashbyhq._parse_raw_job(ats_board_id=42, parsed_job=parsed_job)

        assert result.date_posted == date.today() - timedelta(days=1)


class TestRetrieveAshbyJobsForCompany:
    def test_returns_validated_jobs_for_company_slug(self) -> None:
        response = _FakeResponse(
            status_code=200,
            payload={"jobs": [_ashby_job_payload()]},
        )

        with patch(
            "job_triage.job_search.providers.ashbyhq.requests.get",
            return_value=response,
        ) as mock_get:
            result = ashbyhq._retrieve_ashby_jobs_for_company("scalera")

        assert len(result) == 1
        assert result[0].job.title == "Backend Engineer"
        assert result[0].job.is_remote is True
        assert result[0].job.job_url == (
            "https://jobs.ashbyhq.com/scalera/backend-engineer"
        )
        assert result[0].raw_payload == _ashby_job_payload()
        response.raise_for_status.assert_called_once_with()
        mock_get.assert_called_once_with(
            "https://api.ashbyhq.com/posting-api/job-board/"
            "scalera?includeCompensation=true",
            timeout=ashbyhq._DEFAULT_TIMEOUT,
        )

    def test_returns_empty_list_when_response_has_no_jobs(self) -> None:
        response = _FakeResponse(status_code=200, payload={})

        with patch(
            "job_triage.job_search.providers.ashbyhq.requests.get",
            return_value=response,
        ):
            result = ashbyhq._retrieve_ashby_jobs_for_company("scalera")

        assert result == []

    @pytest.mark.parametrize(
        ("exception_factory", "expected_attempts"),
        [
            (lambda: _http_error(400), 1),
            (lambda: _http_error(408), 6),
            (lambda: _http_error(418), 2),
            (lambda: _http_error(429), 6),
            (lambda: _http_error(503), 6),
            (_request_exception_without_response, 6),
            (_timeout_error, 6),
        ],
    )
    def test_retry_decorator_stops_at_exception_specific_attempt_limit(
        self, exception_factory, expected_attempts
    ) -> None:
        errors = [exception_factory() for _ in range(expected_attempts)]

        with (
            patch(
                "job_triage.job_search.providers.ashbyhq.requests.get",
                side_effect=errors,
            ) as mock_get,
            pytest.raises(type(errors[-1])),
        ):
            ashbyhq._retrieve_ashby_jobs_for_company.retry_with(wait=wait_none())(
                "scalera"
            )

        assert mock_get.call_count == expected_attempts


class TestFilterAshbyJob:
    def test_accepts_remote_recent_job_with_keyword_in_description(self) -> None:
        job = _ashby_job(
            title="Platform Engineer",
            descriptionPlain="Build backend services with Python.",
            publishedAt=_published_at(),
        )

        assert ashbyhq._filter_ashby_job(job, keywords={"python"}) is True

    def test_matches_keywords_case_insensitively_in_title(self) -> None:
        job = _ashby_job(
            title="Senior Backend Engineer",
            descriptionPlain=None,
            publishedAt=_published_at(),
        )

        assert ashbyhq._filter_ashby_job(job, keywords={"backend"}) is True

    @pytest.mark.parametrize(
        "overrides",
        [
            {"isRemote": False, "workplaceType": "Remote"},
            {"isRemote": True, "workplaceType": "OnSite"},
        ],
    )
    def test_rejects_non_remote_or_onsite_jobs(self, overrides) -> None:
        job = _ashby_job(
            descriptionPlain="Python backend services.",
            publishedAt=_published_at(),
            **overrides,
        )

        assert ashbyhq._filter_ashby_job(job, keywords={"python"}) is False

    def test_rejects_jobs_without_matching_keywords(self) -> None:
        job = _ashby_job(
            title="Product Manager",
            descriptionPlain="Own customer discovery and roadmap planning.",
            publishedAt=_published_at(),
        )

        assert ashbyhq._filter_ashby_job(job, keywords={"python"}) is False

    def test_rejects_jobs_older_than_maximum_age(self) -> None:
        job = _ashby_job(
            descriptionPlain="Build Python services.",
            publishedAt=_published_at(days_ago=15),
        )

        assert (
            ashbyhq._filter_ashby_job(
                job,
                keywords={"python"},
                maximum_days_ago=14,
            )
            is False
        )

    def test_accepts_old_jobs_when_updated_recently(self) -> None:
        job = _ashby_job(
            descriptionPlain="Build Python services.",
            publishedAt=_published_at(days_ago=15),
            updatedAt=_updated_at(days_ago=1),
        )

        assert (
            ashbyhq._filter_ashby_job(
                job,
                keywords={"python"},
                maximum_days_ago=14,
            )
            is True
        )

    def test_allows_jobs_without_published_dates(self) -> None:
        job = _ashby_job(
            descriptionPlain="Build Python services.",
            publishedAt=None,
        )

        assert ashbyhq._filter_ashby_job(job, keywords={"python"}) is True

    @pytest.mark.parametrize(
        "max_salary",
        [
            DEFAULT_MINIMUM_SALARY,
            DEFAULT_MINIMUM_SALARY + 1,
        ],
    )
    def test_accepts_jobs_when_company_max_salary_meets_minimum(
        self, max_salary
    ) -> None:
        job = _ashby_job(
            descriptionPlain="Build Python services.",
            compensation=_compensation_payload(max_value=max_salary),
            publishedAt=_published_at(),
        )

        assert ashbyhq._filter_ashby_job(job, keywords={"python"}) is True

    @pytest.mark.parametrize(
        "max_salary",
        [
            0,
            DEFAULT_MINIMUM_SALARY - 1,
        ],
    )
    def test_rejects_jobs_when_company_max_salary_is_below_minimum(
        self, max_salary
    ) -> None:
        job = _ashby_job(
            descriptionPlain="Build Python services.",
            compensation=_compensation_payload(max_value=max_salary),
            publishedAt=_published_at(),
        )

        assert ashbyhq._filter_ashby_job(job, keywords={"python"}) is False

    def test_accepts_jobs_without_salary_metadata(self) -> None:
        job = _ashby_job(
            descriptionPlain="Build Python services.",
            compensation=None,
            publishedAt=_published_at(),
        )

        assert ashbyhq._filter_ashby_job(job, keywords={"python"}) is True


class TestWaitForBraveRetry:
    def test_returns_default_delay_when_outcome_is_missing(self) -> None:
        assert (
            ashbyhq._wait_for_brave_retry(_retry_state(1, None))
            == ashbyhq._DEFAULT_RATE_LIMIT_DELAY
        )

    def test_returns_rate_limit_reset_header_for_429_response(self) -> None:
        error = _http_error(429, headers={"X-RateLimit-Reset": "3.5"})

        assert ashbyhq._wait_for_brave_retry(_retry_state(1, error)) == 3.5

    @pytest.mark.parametrize(
        "error",
        [
            _http_error(429),
            _http_error(503),
            _request_exception_without_response(),
        ],
    )
    def test_uses_default_backoff_when_rate_limit_reset_is_unavailable(
        self, error
    ) -> None:
        retry_state = _retry_state(2, error)

        with patch(
            "job_triage.job_search.providers.ashbyhq._DEFAULT_BACKOFF",
            return_value=2.5,
        ) as mock_backoff:
            result = ashbyhq._wait_for_brave_retry(retry_state)

        assert result == 2.5
        mock_backoff.assert_called_once_with(retry_state)


class TestDynamicStopByError:
    def test_stops_after_two_attempts_when_outcome_is_missing(self) -> None:
        assert ashbyhq._stop_after_attempts_by_error(_retry_state(1, None)) is False
        assert ashbyhq._stop_after_attempts_by_error(_retry_state(2, None)) is True

    def test_value_errors_stop_after_one_attempt(self) -> None:
        error = ValueError("bad configuration")

        assert ashbyhq._stop_after_attempts_by_error(_retry_state(1, error)) is True

    @pytest.mark.parametrize("status_code", [408, 429, 500, 504])
    def test_transient_http_errors_stop_after_six_attempts(self, status_code) -> None:
        error = _http_error(status_code)

        assert ashbyhq._stop_after_attempts_by_error(_retry_state(5, error)) is False
        assert ashbyhq._stop_after_attempts_by_error(_retry_state(6, error)) is True

    @pytest.mark.parametrize("status_code", [400, 401, 402, 403, 404, 413, 422])
    def test_non_retryable_http_errors_stop_after_one_attempt(
        self, status_code
    ) -> None:
        error = _http_error(status_code)

        assert ashbyhq._stop_after_attempts_by_error(_retry_state(1, error)) is True

    @pytest.mark.parametrize("status_code", [409, 418, 499])
    def test_other_http_errors_stop_after_two_attempts(self, status_code) -> None:
        error = _http_error(status_code)

        assert ashbyhq._stop_after_attempts_by_error(_retry_state(1, error)) is False
        assert ashbyhq._stop_after_attempts_by_error(_retry_state(2, error)) is True

    @pytest.mark.parametrize(
        "exception_factory",
        [
            _request_exception_without_response,
            _timeout_error,
        ],
    )
    def test_connection_errors_stop_after_six_attempts(self, exception_factory) -> None:
        error = exception_factory()

        assert ashbyhq._stop_after_attempts_by_error(_retry_state(5, error)) is False
        assert ashbyhq._stop_after_attempts_by_error(_retry_state(6, error)) is True

    def test_other_retryable_errors_stop_after_one_attempt(self) -> None:
        error = RuntimeError("boom")

        assert ashbyhq._stop_after_attempts_by_error(_retry_state(1, error)) is True


class TestExtractAshbySlug:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://jobs.ashbyhq.com/linear", "linear"),
            ("https://jobs.ashbyhq.com/ramp/software-engineer", "ramp"),
            ("https://jobs.ashbyhq.com/cursor?utm_source=brave", "cursor"),
        ],
    )
    def test_extracts_company_slug_from_ashby_url(
        self, url: str, expected: str
    ) -> None:
        assert ashbyhq._extract_ashby_slug(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/linear",
            "https://boards.greenhouse.io/linear",
            "https://jobs.ashbyhq.com",
        ],
    )
    def test_returns_none_for_non_board_urls(self, url: str) -> None:
        assert ashbyhq._extract_ashby_slug(url) is None
