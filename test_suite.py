import sys
import io
import os
import json
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import pytest
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# MOCK STREAMLIT — must happen BEFORE importing app.py
# ---------------------------------------------------------------------------
st_mock = MagicMock()
sys.modules["streamlit"] = st_mock

import app  # noqa: E402 — import after mocking

# Configure session state as a plain dict for testing
app.st.session_state = {}

load_dotenv()


# ===========================================================================
# UNIT TESTS
# ===========================================================================

class TestCalculateReopens:
    """Tests for the calculate_reopens() changelog parser."""

    def test_done_to_todo_counts_as_reopen(self):
        """Classic Done → To Do backward transition."""
        changelog = {
            "histories": [
                {"items": [{"field": "status", "fromString": "Done", "toString": "To Do"}]}
            ]
        }
        assert app.calculate_reopens(changelog) == 1

    def test_explicit_reopened_status_counts(self):
        """A transition whose target status name contains 'reopen'."""
        changelog = {
            "histories": [
                {"items": [{"field": "status", "fromString": "Closed", "toString": "Reopened"}]}
            ]
        }
        assert app.calculate_reopens(changelog) == 1

    def test_forward_transition_does_not_count(self):
        """To Do → In Progress should NOT increment the counter."""
        changelog = {
            "histories": [
                {"items": [{"field": "status", "fromString": "To Do", "toString": "In Progress"}]}
            ]
        }
        assert app.calculate_reopens(changelog) == 0

    def test_done_to_done_substatus_does_not_count(self):
        """Done → Closed (both DONE variants) should not count."""
        changelog = {
            "histories": [
                {"items": [{"field": "status", "fromString": "Done", "toString": "Closed"}]}
            ]
        }
        assert app.calculate_reopens(changelog) == 0

    def test_none_changelog_returns_zero(self):
        """None changelog is a valid API edge case — must not crash."""
        assert app.calculate_reopens(None) == 0

    def test_empty_dict_changelog_returns_zero(self):
        """Empty changelog dict — no histories key."""
        assert app.calculate_reopens({}) == 0

    def test_multiple_reopens_counted_correctly(self):
        """Two separate reopen events in history."""
        changelog = {
            "histories": [
                {"items": [{"field": "status", "fromString": "Done", "toString": "To Do"}]},
                {"items": [{"field": "status", "fromString": "Done", "toString": "Backlog"}]},
            ]
        }
        assert app.calculate_reopens(changelog) == 2

    def test_non_status_field_is_ignored(self):
        """Changes to non-status fields must be ignored."""
        changelog = {
            "histories": [
                {"items": [{"field": "assignee", "fromString": "Alice", "toString": "Bob"}]}
            ]
        }
        assert app.calculate_reopens(changelog) == 0


class TestNormalizeJiraData:
    """Tests for normalize_jira_data() — raw JSON → DataFrame."""

    def _full_issue(self, key="TEST-1"):
        return {
            "key": key,
            "fields": {
                "summary": "Fix Bug",
                "status": {"name": "In Progress"},
                "assignee": {"displayName": "Dev User"},
                "created": "2023-01-01T10:00:00.000+0000",
                "updated": "2023-01-02T10:00:00.000+0000",
                "issuetype": {"name": "Bug"},
                "priority": {"name": "High"},
            },
            "changelog": {},
        }

    def test_required_columns_present(self):
        """All expected columns must exist in the output DataFrame."""
        df = app.normalize_jira_data([self._full_issue()])
        expected = [
            "Issue key", "Summary", "Status", "Assignee",
            "Created", "Updated", "Issue Type", "Priority",
            "Reopen Count", "Team_Source", "Transitions",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_values_mapped_correctly(self):
        """Spot-check key field values."""
        df = app.normalize_jira_data([self._full_issue("TEST-1")])
        assert df.iloc[0]["Issue key"] == "TEST-1"
        assert df.iloc[0]["Status"] == "In Progress"
        assert df.iloc[0]["Assignee"] == "Dev User"

    def test_empty_fields_use_defaults(self):
        """Issue with empty `fields` should use sensible defaults, not crash."""
        raw = [{"key": "TEST-2", "fields": {}, "changelog": None}]
        df = app.normalize_jira_data(raw)
        assert df.iloc[0]["Assignee"] == "Unassigned"
        assert df.iloc[0]["Status"] == "Unknown"

    def test_empty_list_returns_empty_dataframe(self):
        """No issues → empty DataFrame."""
        df = app.normalize_jira_data([])
        assert df.empty


class TestCalculateMetricsZombies:
    """Tests for zombie & orphan detection inside calculate_metrics()."""

    def _make_df(self):
        now = pd.Timestamp.now(tz="UTC")
        old_date = now - pd.Timedelta(days=30)
        recent_date = now - pd.Timedelta(days=1)
        return pd.DataFrame({
            "Issue key": ["Z-1", "A-1", "O-1"],
            "Summary":   ["Old Task", "Active Task", "Orphan Task"],
            # Status names intentionally NOT pre-mapped — mapping does it
            "Status":    ["To Do", "In Progress", "To Do"],
            "Assignee":  ["User A", "User A", None],
            "Updated":   [old_date, recent_date, recent_date],
            "Issue Type": ["Task", "Task", "Task"],
            "Transitions": [[], [], []],
        })

    def test_old_inactive_task_is_zombie(self):
        mapping = {"To Do": "TODO", "In Progress": "IN_PROGRESS"}
        results = app.calculate_metrics(self._make_df(), 14, mapping)
        assert "Z-1" in results["zombies"]["Issue key"].values

    def test_recently_updated_task_not_zombie(self):
        mapping = {"To Do": "TODO", "In Progress": "IN_PROGRESS"}
        results = app.calculate_metrics(self._make_df(), 14, mapping)
        assert "A-1" not in results["zombies"]["Issue key"].values

    def test_unassigned_task_is_orphan(self):
        mapping = {"To Do": "TODO", "In Progress": "IN_PROGRESS"}
        results = app.calculate_metrics(self._make_df(), 14, mapping)
        assert "O-1" in results["orphans"]["Issue key"].values


class TestCalculateMetricsWipOverload:
    """Tests for WIP overload detection (threshold = 2)."""

    def _make_df(self):
        now = pd.Timestamp.now(tz="UTC")
        rows = []
        # User A — 3 active tasks → OVERLOADED
        for i in range(3):
            rows.append({
                "Assignee": "User A", "Issue key": f"A-{i}",
                "Status": "In Progress", "Updated": now,
                "Issue Type": "Task", "Transitions": [],
            })
        # User B — exactly 2 active tasks → NOT overloaded
        for i in range(2):
            rows.append({
                "Assignee": "User B", "Issue key": f"B-{i}",
                "Status": "To Do", "Updated": now,
                "Issue Type": "Task", "Transitions": [],
            })
        return pd.DataFrame(rows)

    def test_user_above_threshold_is_overloaded(self):
        mapping = {"To Do": "TODO", "In Progress": "IN_PROGRESS"}
        results = app.calculate_metrics(self._make_df(), 30, mapping)
        assert "User A" in results["overloaded_assignees"], \
            "User A with 3 tasks should be overloaded (threshold is 2)"

    def test_user_at_threshold_is_not_overloaded(self):
        mapping = {"To Do": "TODO", "In Progress": "IN_PROGRESS"}
        results = app.calculate_metrics(self._make_df(), 30, mapping)
        assert "User B" not in results["overloaded_assignees"], \
            "User B with exactly 2 tasks should NOT be overloaded"

    def test_done_tasks_excluded_from_wip(self):
        """DONE tasks must not contribute to WIP count."""
        now = pd.Timestamp.now(tz="UTC")
        df = pd.DataFrame([
            {"Assignee": "User C", "Issue key": "C-1", "Status": "Done",
             "Updated": now, "Issue Type": "Task", "Transitions": []},
            {"Assignee": "User C", "Issue key": "C-2", "Status": "Done",
             "Updated": now, "Issue Type": "Task", "Transitions": []},
            {"Assignee": "User C", "Issue key": "C-3", "Status": "Done",
             "Updated": now, "Issue Type": "Task", "Transitions": []},
        ])
        mapping = {"Done": "DONE"}
        results = app.calculate_metrics(df, 30, mapping)
        assert "User C" not in results["overloaded_assignees"], \
            "User C's tasks are all DONE and should not count as WIP"


class TestCalculateMetricsDeadEpics:
    """Tests for dead epic detection."""

    def test_old_todo_epic_is_dead(self):
        now = pd.Timestamp.now(tz="UTC")
        df = pd.DataFrame({
            "Issue key":  ["E-1", "E-2", "E-3"],
            "Issue Type": ["Epic", "Epic", "Task"],
            "Status":     ["To Do", "Done", "To Do"],
            "Updated":    [now - pd.Timedelta(days=31)] * 3,
            "Assignee":   ["User A"] * 3,
            "Transitions": [[], [], []],
        })
        mapping = {"To Do": "TODO", "Done": "DONE"}
        results = app.calculate_metrics(df, 30, mapping)
        dead = results["dead_epics"]["Issue key"].values

        assert "E-1" in dead,    "E-1: Epic, TODO, old → Dead"
        assert "E-2" not in dead, "E-2: DONE → not dead"
        assert "E-3" not in dead, "E-3: Task, not Epic → not dead"


class TestCalculateMetricsNoiseRatio:
    """Tests for noise ratio (reopened / done)."""

    def test_noise_ratio_50_percent(self):
        """1 noisy task out of 2 DONE → 50%."""
        df = pd.DataFrame({
            "Issue key":  ["N-1", "C-1"],
            "Status":     ["Done", "Done"],
            "Transitions": [
                [{"to": "Reopened", "from": "Done", "date": "2023-01-01"}],
                [],
            ],
            "Updated":   [pd.Timestamp.now(tz="UTC")] * 2,
            "Assignee":  ["User A", "User A"],
            "Issue Type": ["Task", "Task"],
        })
        mapping = {"Done": "DONE", "Reopened": "TODO"}
        results = app.calculate_metrics(df, 30, mapping)
        assert results["noise_ratio"] == 50.0

    def test_noise_ratio_zero_when_no_done(self):
        """No DONE tasks → ratio must be 0, not a division error."""
        df = pd.DataFrame({
            "Issue key":  ["T-1"],
            "Status":     ["To Do"],
            "Transitions": [[]],
            "Updated":   [pd.Timestamp.now(tz="UTC")],
            "Assignee":  ["User A"],
            "Issue Type": ["Task"],
        })
        mapping = {"To Do": "TODO"}
        results = app.calculate_metrics(df, 30, mapping)
        assert results["noise_ratio"] == 0


class TestEmptyDataframeSafety:
    """Metrics calculation must not crash on an empty DataFrame."""

    def test_all_metrics_safe_on_empty_input(self):
        df = pd.DataFrame(columns=[
            "Issue key", "Status", "Updated", "Assignee",
            "Issue Type", "Transitions",
        ])
        results = app.calculate_metrics(df, 30, {})
        assert results["total_issues"] == 0
        assert results["noise_ratio"] == 0
        assert results["zombies"].empty
        assert results["orphans"].empty
        assert results["dead_epics"].empty
        assert results["overloaded_assignees"] == []


class TestHealthScoreCalculation:
    """Health score formula: 100 - (penalty / active_issues * 10), clamped to [0, 100]."""

    def _score(self, zombies=0, orphans=0, dead_epics=0, active=10):
        """Replicates the formula from app.main()."""
        penalty = (zombies * 2) + (orphans * 1) + (dead_epics * 5)
        return max(0, 100 - (penalty / max(1, active) * 10))

    def test_perfect_backlog_scores_100(self):
        assert self._score() == 100.0

    def test_score_decreases_with_zombies(self):
        assert self._score(zombies=5, active=10) < 100

    def test_dead_epic_has_highest_penalty(self):
        """Each dead epic costs 5 vs 2 for zombies."""
        s_epic = self._score(dead_epics=1, active=10)
        s_zombie = self._score(zombies=1, active=10)
        assert s_epic < s_zombie

    def test_score_does_not_go_below_zero(self):
        """With many issues, score must clamp to 0."""
        assert self._score(zombies=1000, orphans=1000, dead_epics=1000, active=1) == 0

    def test_score_does_not_exceed_100(self):
        assert self._score() <= 100


class TestClassifyStatusesWithAiMock:
    """Tests for classify_statuses_with_ai() using mocked OpenAI client."""

    def test_returns_parsed_mapping_dict(self):
        """With a valid API response, function should return a dict."""
        mock_result = {"To Do": "TODO", "In Progress": "IN_PROGRESS", "Done": "DONE"}

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(mock_result)

        with patch("app.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = mock_response

            result = app.classify_statuses_with_ai(["To Do", "In Progress", "Done"], "fake-key")

        assert result == mock_result

    def test_returns_none_when_no_api_key(self):
        """No API key → should return None immediately without calling OpenAI."""
        with patch("app.OpenAI") as MockOpenAI:
            result = app.classify_statuses_with_ai(["To Do"], api_key=None)
            MockOpenAI.assert_not_called()
        assert result is None

    def test_returns_none_on_openai_exception(self):
        """If OpenAI raises an exception, function should return None gracefully."""
        with patch("app.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.side_effect = Exception("API error")
            result = app.classify_statuses_with_ai(["To Do"], api_key="fake-key")
        assert result is None


# ===========================================================================
# INTEGRATION TESTS (skipped when env vars not set)
# ===========================================================================

@pytest.mark.skipif(
    not os.getenv("JIRA_URL"),
    reason="Skipped: no JIRA_URL in .env — set JIRA_URL, JIRA_EMAIL, JIRA_TOKEN, JIRA_PROJECT_KEY"
)
def test_real_jira_connection_and_fetch():
    """Connect to a real Jira instance and verify the returned DataFrame.

    Requires .env with:
        JIRA_URL, JIRA_EMAIL, JIRA_TOKEN, JIRA_PROJECT_KEY
    """
    url     = os.getenv("JIRA_URL")
    email   = os.getenv("JIRA_EMAIL")
    token   = os.getenv("JIRA_TOKEN")
    project = os.getenv("JIRA_PROJECT_KEY")

    df, error = app.fetch_jira_data(url, email, token, project, days_back=30)

    assert error is None, f"Jira fetch returned an error: {error}"
    assert df is not None, "fetch_jira_data returned None"
    assert not df.empty, "Fetched DataFrame is empty"

    required_cols = ["Issue key", "Summary", "Status", "Assignee", "Issue Type"]
    for col in required_cols:
        assert col in df.columns, f"Missing column after Jira fetch: {col}"


def test_csv_loading_if_exists():
    """Verify CSV loading works with any .csv file found in the project directory."""
    files = [f for f in os.listdir(".") if f.endswith(".csv")]
    if not files:
        pytest.skip("No CSV files found in project directory — skipping integration test")

    class MockFile(io.BytesIO):
        def __init__(self, name, content):
            super().__init__(content)
            self.name = name

    file_path = files[0]
    with open(file_path, "rb") as f:
        content = f.read()

    uploaded_file = MockFile(file_path, content)
    df, errors = app.load_data([uploaded_file])

    assert not df.empty, "CSV loaded but resulting DataFrame is empty"
    assert "Issue key" in df.columns, "Expected 'Issue key' column missing"
    assert len(errors) == 0, f"Unexpected load errors: {errors}"
