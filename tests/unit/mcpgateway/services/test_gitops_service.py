# -*- coding: utf-8 -*-
"""Location: ./tests/unit/mcpgateway/services/test_gitops_service.py
Copyright 2026
SPDX-License-Identifier: Apache-2.0
Authors: Ioannis Ioannou

Unit tests for the Policy GitOps service.

Examples:
    >>> True
    True
"""
# Standard
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call
import uuid

# Third-Party
import pytest

# First-Party
from mcpgateway.services.gitops_service import (
    GitOpsService,
    PolicyNotFoundError,
    PolicyRollbackError,
    InvalidPromotionPathError,
    PolicyApprovalError,
    PROMOTION_PATH,
)


def _make_version(**kwargs):
    """Create a mock PolicyVersion with sensible defaults."""
    v = MagicMock()
    v.id = kwargs.get("id", str(uuid.uuid4()))
    v.policy_name = kwargs.get("policy_name", "test-policy")
    v.version = kwargs.get("version", "v1")
    v.content = kwargs.get("content", "permit all;")
    v.content_hash = kwargs.get("content_hash", "abc123")
    v.engine = kwargs.get("engine", "cedar")
    v.author = kwargs.get("author", "dev@example.com")
    v.environment = kwargs.get("environment", "dev")
    v.is_active = kwargs.get("is_active", True)
    v.commit_sha = kwargs.get("commit_sha", None)
    v.commit_message = kwargs.get("commit_message", None)
    v.change_summary = kwargs.get("change_summary", None)
    v.created_at = kwargs.get("created_at", datetime.now(timezone.utc))
    return v


def _make_db():
    """Create a mock SQLAlchemy session."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.count.return_value = 0
    db.query.return_value.filter.return_value.update.return_value = 0
    db.query.return_value.distinct.return_value.all.return_value = []
    db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
    db.query.return_value.order_by.return_value.all.return_value = []
    db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
    return db


# ---------------------------------------------------------------------------
# TestStoreVersion
# ---------------------------------------------------------------------------

class TestStoreVersion:
    """Tests for GitOpsService.store_version."""

    def test_store_version_new_content(self):
        """New content is stored and marked active."""
        db = _make_db()
        db.query.return_value.filter.return_value.first.return_value = None
        db.query.return_value.filter.return_value.count.return_value = 0
        svc = GitOpsService(db)
        result = svc.store_version("my-policy", "permit all;", "cedar", "dev@x.com", "dev")
        db.add.assert_called_once()
        db.commit.assert_called_once()
        assert result.is_active is True

    def test_store_version_deduplication_returns_existing(self):
        """Identical content returns existing version without writing."""
        db = _make_db()
        existing = _make_version(content_hash="same_hash")
        db.query.return_value.filter.return_value.first.return_value = existing
        svc = GitOpsService(db)
        result = svc.store_version("my-policy", "permit all;", "cedar", "dev@x.com", "dev")
        db.add.assert_not_called()
        assert result is existing

    def test_store_version_increments_version_tag(self):
        """Version tag increments based on existing count."""
        db = _make_db()
        db.query.return_value.filter.return_value.first.return_value = None
        db.query.return_value.filter.return_value.count.return_value = 3
        svc = GitOpsService(db)
        result = svc.store_version("my-policy", "permit all;", "cedar", "dev@x.com", "dev")
        assert result.version == "v4"

    def test_store_version_with_commit_metadata(self):
        """Commit SHA and message are stored on the version."""
        db = _make_db()
        db.query.return_value.filter.return_value.first.return_value = None
        db.query.return_value.filter.return_value.count.return_value = 0
        svc = GitOpsService(db)
        result = svc.store_version(
            "my-policy", "permit all;", "cedar", "dev@x.com", "dev",
            commit_sha="abc123def456", commit_message="Add tool policy",
        )
        assert result.commit_sha == "abc123def456"
        assert result.commit_message == "Add tool policy"


# ---------------------------------------------------------------------------
# TestGetVersion
# ---------------------------------------------------------------------------

class TestGetVersion:
    """Tests for GitOpsService.get_version."""

    def test_get_version_found(self):
        """Returns version when it exists."""
        db = _make_db()
        v = _make_version()
        db.query.return_value.filter.return_value.first.return_value = v
        svc = GitOpsService(db)
        assert svc.get_version(v.id) is v

    def test_get_version_not_found_raises(self):
        """Raises PolicyNotFoundError when version does not exist."""
        db = _make_db()
        db.query.return_value.filter.return_value.first.return_value = None
        svc = GitOpsService(db)
        with pytest.raises(PolicyNotFoundError):
            svc.get_version("nonexistent-id")


# ---------------------------------------------------------------------------
# TestGetHistory
# ---------------------------------------------------------------------------

class TestGetHistory:
    """Tests for GitOpsService.get_history."""

    def test_get_history_returns_versions(self):
        """Returns list of versions for a policy."""
        db = _make_db()
        versions = [_make_version(version=f"v{i}") for i in range(3)]
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = versions
        svc = GitOpsService(db)
        result = svc.get_history("test-policy")
        assert len(result) == 3

    def test_get_history_empty(self):
        """Returns empty list when no versions exist."""
        db = _make_db()
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        svc = GitOpsService(db)
        assert svc.get_history("nonexistent") == []


# ---------------------------------------------------------------------------
# TestDiff
# ---------------------------------------------------------------------------

class TestDiff:
    """Tests for GitOpsService.diff."""

    def test_diff_detects_added_lines(self):
        """Added lines are detected in the diff."""
        db = _make_db()
        v1 = _make_version(id="id-a", version="v1", content="line1\n")
        v2 = _make_version(id="id-b", version="v2", content="line1\nline2\n")
        def get_version_mock(vid):
            return v1 if vid == "id-a" else v2
        svc = GitOpsService(db)
        svc.get_version = get_version_mock
        result = svc.diff("id-a", "id-b")
        assert result["added_lines"] == 1
        assert result["removed_lines"] == 0

    def test_diff_detects_removed_lines(self):
        """Removed lines are detected in the diff."""
        db = _make_db()
        v1 = _make_version(id="id-a", version="v1", content="line1\nline2\n")
        v2 = _make_version(id="id-b", version="v2", content="line1\n")
        svc = GitOpsService(db)
        svc.get_version = lambda vid: v1 if vid == "id-a" else v2
        result = svc.diff("id-a", "id-b")
        assert result["removed_lines"] == 1

    def test_diff_no_changes(self):
        """Identical content produces empty diff."""
        db = _make_db()
        v1 = _make_version(id="id-a", version="v1", content="same\n")
        v2 = _make_version(id="id-b", version="v2", content="same\n")
        svc = GitOpsService(db)
        svc.get_version = lambda vid: v1 if vid == "id-a" else v2
        result = svc.diff("id-a", "id-b")
        assert result["added_lines"] == 0
        assert result["removed_lines"] == 0


# ---------------------------------------------------------------------------
# TestRollback
# ---------------------------------------------------------------------------

class TestRollback:
    """Tests for GitOpsService.rollback."""

    def test_rollback_success(self):
        """Successfully rolls back to a previous inactive version."""
        db = _make_db()
        target = _make_version(policy_name="my-policy", version="v1", is_active=False, environment="dev")
        svc = GitOpsService(db)
        svc.get_version = MagicMock(return_value=target)
        result = svc.rollback("my-policy", target.id, "Emergency rollback", "admin@x.com")
        assert target.is_active is True
        db.add.assert_called_once()
        db.commit.assert_called_once()

    def test_rollback_already_active_raises(self):
        """Raises PolicyRollbackError if target is already active."""
        db = _make_db()
        target = _make_version(policy_name="my-policy", is_active=True)
        svc = GitOpsService(db)
        svc.get_version = MagicMock(return_value=target)
        with pytest.raises(PolicyRollbackError, match="already active"):
            svc.rollback("my-policy", target.id, "reason", "admin@x.com")

    def test_rollback_wrong_policy_raises(self):
        """Raises PolicyRollbackError if version belongs to different policy."""
        db = _make_db()
        target = _make_version(policy_name="other-policy", is_active=False)
        svc = GitOpsService(db)
        svc.get_version = MagicMock(return_value=target)
        with pytest.raises(PolicyRollbackError, match="does not belong"):
            svc.rollback("my-policy", target.id, "reason", "admin@x.com")


# ---------------------------------------------------------------------------
# TestPromote
# ---------------------------------------------------------------------------

class TestPromote:
    """Tests for GitOpsService.promote."""

    def test_promote_dev_to_staging(self):
        """Successfully promotes from dev to staging."""
        db = _make_db()
        source = _make_version(policy_name="my-policy", environment="dev", is_active=True)
        db.query.return_value.filter.return_value.first.return_value = source
        svc = GitOpsService(db)
        svc.store_version = MagicMock(return_value=_make_version(environment="staging"))
        result = svc.promote("my-policy", "dev", "staging", "admin@x.com")
        svc.store_version.assert_called_once()
        db.add.assert_called_once()
        db.commit.assert_called_once()

    def test_promote_invalid_path_raises(self):
        """Raises InvalidPromotionPathError for invalid promotion path."""
        db = _make_db()
        svc = GitOpsService(db)
        with pytest.raises(InvalidPromotionPathError):
            svc.promote("my-policy", "prod", "dev", "admin@x.com")

    def test_promote_skipping_env_raises(self):
        """Raises InvalidPromotionPathError when skipping an environment."""
        db = _make_db()
        svc = GitOpsService(db)
        with pytest.raises(InvalidPromotionPathError):
            svc.promote("my-policy", "dev", "prod", "admin@x.com")

    def test_promote_no_active_source_raises(self):
        """Raises PolicyNotFoundError when no active version in source env."""
        db = _make_db()
        db.query.return_value.filter.return_value.first.return_value = None
        svc = GitOpsService(db)
        with pytest.raises(PolicyNotFoundError):
            svc.promote("my-policy", "dev", "staging", "admin@x.com")


# ---------------------------------------------------------------------------
# TestWebhook
# ---------------------------------------------------------------------------

class TestWebhook:
    """Tests for GitOpsService.handle_webhook."""

    def test_webhook_unknown_ref_skipped(self):
        """Unknown ref returns skipped status."""
        db = _make_db()
        svc = GitOpsService(db)
        result = svc.handle_webhook({"ref": "refs/heads/unknown", "commits": []}, None)
        assert result["status"] == "skipped"

    def test_webhook_main_maps_to_prod(self):
        """refs/heads/main maps to prod environment."""
        db = _make_db()
        svc = GitOpsService(db)
        result = svc.handle_webhook({"ref": "refs/heads/main", "commits": [], "pusher": {}}, None)
        assert result["environment"] == "prod"

    def test_webhook_stores_cedar_file(self):
        """Cedar policy files are stored from webhook payload."""
        db = _make_db()
        db.query.return_value.filter.return_value.first.return_value = None
        db.query.return_value.filter.return_value.count.return_value = 0
        svc = GitOpsService(db)
        svc.store_version = MagicMock(return_value=_make_version())
        payload = {
            "ref": "refs/heads/develop",
            "pusher": {"email": "ci@example.com"},
            "commits": [{
                "id": "abc123",
                "message": "Add tool policy",
                "added": ["policies/tools/read-only.cedar"],
                "modified": [],
                "_file_contents": {"policies/tools/read-only.cedar": "permit all;"},
            }],
        }
        result = svc.handle_webhook(payload, None)
        assert result["stored"] == 1
        assert result["skipped"] == 0

    def test_webhook_skips_non_policy_files(self):
        """Non-.cedar and non-.rego files are skipped."""
        db = _make_db()
        svc = GitOpsService(db)
        svc.store_version = MagicMock()
        payload = {
            "ref": "refs/heads/develop",
            "pusher": {"email": "ci@example.com"},
            "commits": [{"id": "abc", "message": "update readme", "added": ["README.md"], "modified": []}],
        }
        result = svc.handle_webhook(payload, None)
        assert result["skipped"] == 1
        svc.store_version.assert_not_called()


# ---------------------------------------------------------------------------
# TestApprovals
# ---------------------------------------------------------------------------

class TestApprovals:
    """Tests for approval workflow."""

    def test_request_approval_creates_record(self):
        """Creates a pending approval record."""
        db = _make_db()
        svc = GitOpsService(db)
        svc.get_version = MagicMock(return_value=_make_version())
        result = svc.request_approval("version-id", "dev@x.com", "Please approve")
        db.add.assert_called_once()
        db.commit.assert_called_once()
        assert result.status == "pending"

    def test_resolve_approval_approved(self):
        """Resolves a pending approval as approved."""
        db = _make_db()
        approval = MagicMock()
        approval.status = "pending"
        db.query.return_value.filter.return_value.first.return_value = approval
        svc = GitOpsService(db)
        result = svc.resolve_approval("approval-id", "admin@x.com", "approved")
        assert approval.status == "approved"
        assert approval.approved_by == "admin@x.com"
        db.commit.assert_called_once()

    def test_resolve_approval_rejected(self):
        """Resolves a pending approval as rejected."""
        db = _make_db()
        approval = MagicMock()
        approval.status = "pending"
        db.query.return_value.filter.return_value.first.return_value = approval
        svc = GitOpsService(db)
        svc.resolve_approval("approval-id", "admin@x.com", "rejected")
        assert approval.status == "rejected"

    def test_resolve_approval_invalid_decision_raises(self):
        """Raises PolicyApprovalError for invalid decision."""
        db = _make_db()
        svc = GitOpsService(db)
        with pytest.raises(PolicyApprovalError, match="Invalid decision"):
            svc.resolve_approval("id", "admin@x.com", "maybe")

    def test_resolve_approval_already_resolved_raises(self):
        """Raises PolicyApprovalError if approval is already resolved."""
        db = _make_db()
        approval = MagicMock()
        approval.status = "approved"
        db.query.return_value.filter.return_value.first.return_value = approval
        svc = GitOpsService(db)
        with pytest.raises(PolicyApprovalError, match="already"):
            svc.resolve_approval("id", "admin@x.com", "approved")

    def test_resolve_approval_not_found_raises(self):
        """Raises PolicyApprovalError when approval does not exist."""
        db = _make_db()
        db.query.return_value.filter.return_value.first.return_value = None
        svc = GitOpsService(db)
        with pytest.raises(PolicyApprovalError, match="not found"):
            svc.resolve_approval("missing-id", "admin@x.com", "approved")


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for internal helper methods."""

    def test_valid_promotion_dev_to_staging(self):
        svc = GitOpsService(MagicMock())
        assert svc._valid_promotion("dev", "staging") is True

    def test_valid_promotion_staging_to_prod(self):
        svc = GitOpsService(MagicMock())
        assert svc._valid_promotion("staging", "prod") is True

    def test_invalid_promotion_skip(self):
        svc = GitOpsService(MagicMock())
        assert svc._valid_promotion("dev", "prod") is False

    def test_invalid_promotion_backwards(self):
        svc = GitOpsService(MagicMock())
        assert svc._valid_promotion("prod", "staging") is False

    def test_detect_engine_cedar(self):
        svc = GitOpsService(MagicMock())
        assert svc._detect_engine("policies/tools/read-only.cedar") == "cedar"

    def test_detect_engine_opa(self):
        svc = GitOpsService(MagicMock())
        assert svc._detect_engine("policies/rego/authz.rego") == "opa"

    def test_detect_engine_unknown(self):
        svc = GitOpsService(MagicMock())
        assert svc._detect_engine("README.md") is None

    def test_path_to_policy_name(self):
        svc = GitOpsService(MagicMock())
        assert svc._path_to_policy_name("policies/tools/read_only.cedar") == "read-only"

    def test_ref_to_environment_main(self):
        svc = GitOpsService(MagicMock())
        assert svc._ref_to_environment("refs/heads/main") == "prod"

    def test_ref_to_environment_develop(self):
        svc = GitOpsService(MagicMock())
        assert svc._ref_to_environment("refs/heads/develop") == "dev"

    def test_ref_to_environment_unknown(self):
        svc = GitOpsService(MagicMock())
        assert svc._ref_to_environment("refs/heads/feature/foo") is None
