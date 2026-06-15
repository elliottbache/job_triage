from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from requests import HTTPError, RequestException, Timeout
from tenacity import wait_none

from job_triage.job_search import ashbyhq


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


class TestDiscoverAshbySlugs:
    def test_returns_unique_slugs_from_brave_urls(self) -> None:
        with patch(
            "job_triage.job_search.ashbyhq._search_brave",
            return_value=[
                "https://jobs.ashbyhq.com/linear/software-engineer",
                "https://jobs.ashbyhq.com/ramp",
                "https://jobs.ashbyhq.com/linear/data-engineer",
                "https://example.com/not-an-ashby-board",
            ],
        ) as mock_search:
            result = ashbyhq._discover_ashby_slugs("python remote", max_results=5)

        assert result == {"linear", "ramp"}
        mock_search.assert_called_once_with("python remote", max_results=5)


class TestSearchBrave:
    def test_raises_when_api_key_is_missing(self) -> None:
        with (
            patch("job_triage.job_search.ashbyhq.getenv", return_value=None),
            pytest.raises(ValueError, match="No Brave search API key"),
        ):
            ashbyhq._search_brave("python remote", max_results=10)

    def test_returns_unique_urls_from_paginated_results(self) -> None:
        client = MagicMock()
        client_context = MagicMock()
        client_context.__enter__.return_value = client
        client_context.__exit__.return_value = None

        with (
            patch("job_triage.job_search.ashbyhq.getenv", return_value="brave-key"),
            patch(
                "job_triage.job_search.ashbyhq.requests.Session",
                return_value=client_context,
            ),
            patch(
                "job_triage.job_search.ashbyhq._safe_brave_request",
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
            result = ashbyhq._search_brave("python remote", max_results=3)

        assert result == [
            "https://jobs.ashbyhq.com/linear",
            "https://example.com/search-result",
            "https://jobs.ashbyhq.com/ramp",
        ]
        assert mock_request.call_count == 2
        assert mock_request.call_args_list[0].kwargs["params"] == {
            "q": "python remote",
            "count": 3,
            "offset": 0,
            "result_filter": "web",
        }
        assert mock_request.call_args_list[1].kwargs["params"] == {
            "q": "python remote",
            "count": 1,
            "offset": 1,
            "result_filter": "web",
        }

    def test_stops_when_page_has_no_results(self) -> None:
        client_context = MagicMock()
        client_context.__enter__.return_value = MagicMock()
        client_context.__exit__.return_value = None

        with (
            patch("job_triage.job_search.ashbyhq.getenv", return_value="brave-key"),
            patch(
                "job_triage.job_search.ashbyhq.requests.Session",
                return_value=client_context,
            ),
            patch(
                "job_triage.job_search.ashbyhq._safe_brave_request",
                return_value={"web": {"results": []}},
            ) as mock_request,
        ):
            result = ashbyhq._search_brave("python remote", max_results=10)

        assert result == []
        mock_request.assert_called_once()


class TestSafeBraveRequest:
    def test_returns_json_after_successful_request(self) -> None:
        client = MagicMock()
        response = _FakeResponse(status_code=200, payload={"web": {"results": []}})
        client.get.return_value = response

        with patch("job_triage.job_search.ashbyhq.time.sleep") as mock_sleep:
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
            timeout=ashbyhq._DEFAULT_BRAVE_TIMEOUT,
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

        with patch("job_triage.job_search.ashbyhq.time.sleep") as mock_sleep:
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
            "job_triage.job_search.ashbyhq._DEFAULT_BACKOFF",
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
