from unittest.mock import MagicMock, patch

import pytest

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
        self.raise_for_status = MagicMock()

    def json(self) -> dict[str, object]:
        return self._payload


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
        assert mock_sleep.call_args_list[0].args == (3,)
        assert mock_sleep.call_args_list[1].args == (0.02,)


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
