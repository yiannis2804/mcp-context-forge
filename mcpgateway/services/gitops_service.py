# -*- coding: utf-8 -*-
"""Location: ./mcpgateway/services/gitops_service.py
Copyright 2026
SPDX-License-Identifier: Apache-2.0
Authors: Ioannis Ioannou

Policy GitOps Service — version control, webhook handling, promotion and rollback.

Related to Issue #2238: Policy GitOps and version control

Examples:
    >>> True
    True
"""
# Future
from __future__ import annotations

# Standard
import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

# Third-Party
from sqlalchemy.orm import Session

# First-Party
from mcpgateway.config import settings
from mcpgateway.db import PolicyVersion, PolicyDeployment, PolicyApproval

logger = logging.getLogger(__name__)

PROMOTION_PATH = ["dev", "staging", "prod"]


class PolicyNotFoundError(Exception):
    """Raised when a policy version cannot be found."""


class PolicyRollbackError(Exception):
    """Raised when a rollback is unsafe or not possible."""


class InvalidPromotionPathError(Exception):
    """Raised when an invalid promotion path is requested."""


class PolicyApprovalError(Exception):
    """Raised when an approval operation is invalid."""


class GitOpsService:
    """Manage policy versions, Git webhook events, deployments, and promotions.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> db = MagicMock()
        >>> svc = GitOpsService(db)
        >>> isinstance(svc, GitOpsService)
        True
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def store_version(self, policy_name: str, content: str, engine: str, author: str,
                      environment: str, commit_sha: Optional[str] = None,
                      commit_message: Optional[str] = None, change_summary: Optional[str] = None) -> PolicyVersion:
        """Store a new policy version, deduplicating by content hash.

        Args:
            policy_name: Name of the policy.
            content: Raw policy content (Cedar or Rego).
            engine: Policy engine ('cedar' or 'opa').
            author: Email of the author.
            environment: Target environment ('dev', 'staging', 'prod').
            commit_sha: Optional Git commit SHA.
            commit_message: Optional Git commit message.
            change_summary: Optional human-readable summary.

        Returns:
            The stored PolicyVersion ORM object.

        Examples:
            >>> from unittest.mock import MagicMock
            >>> db = MagicMock()
            >>> db.query.return_value.filter.return_value.first.return_value = None
            >>> svc = GitOpsService(db)
            >>> isinstance(svc, GitOpsService)
            True
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = (self.db.query(PolicyVersion)
                    .filter(PolicyVersion.policy_name == policy_name,
                            PolicyVersion.content_hash == content_hash,
                            PolicyVersion.environment == environment).first())
        if existing:
            logger.info("Policy %s/%s unchanged, skipping store", policy_name, environment)
            return existing

        version_tag = self._next_version_tag(policy_name, environment)
        pv = PolicyVersion(
            id=str(uuid.uuid4()), policy_name=policy_name, version=version_tag,
            content=content, content_hash=content_hash, engine=engine, author=author,
            commit_sha=commit_sha, commit_message=commit_message, change_summary=change_summary,
            environment=environment, is_active=False,
        )
        self.db.add(pv)
        self.db.query(PolicyVersion).filter(
            PolicyVersion.policy_name == policy_name,
            PolicyVersion.environment == environment,
            PolicyVersion.is_active == True,  # noqa: E712
        ).update({"is_active": False})
        pv.is_active = True
        self.db.commit()
        self.db.refresh(pv)
        logger.info("Stored policy version %s for %s/%s", version_tag, policy_name, environment)
        return pv

    def get_history(self, policy_name: str, environment: Optional[str] = None, limit: int = 50) -> List[PolicyVersion]:
        """Return version history for a policy, newest first.

        Args:
            policy_name: Name of the policy.
            environment: Optional environment filter.
            limit: Maximum number of versions to return.

        Returns:
            List of PolicyVersion objects ordered by created_at descending.
        """
        q = self.db.query(PolicyVersion).filter(PolicyVersion.policy_name == policy_name)
        if environment:
            q = q.filter(PolicyVersion.environment == environment)
        return q.order_by(PolicyVersion.created_at.desc()).limit(limit).all()

    def get_version(self, version_id: str) -> PolicyVersion:
        """Fetch a single policy version by ID.

        Args:
            version_id: UUID of the policy version.

        Returns:
            PolicyVersion ORM object.

        Raises:
            PolicyNotFoundError: If version does not exist.
        """
        pv = self.db.query(PolicyVersion).filter(PolicyVersion.id == version_id).first()
        if not pv:
            raise PolicyNotFoundError(f"Policy version not found: {version_id}")
        return pv

    def list_policies(self, environment: Optional[str] = None) -> List[str]:
        """Return distinct policy names.

        Args:
            environment: Optional environment filter.

        Returns:
            Sorted list of unique policy names.
        """
        q = self.db.query(PolicyVersion.policy_name).distinct()
        if environment:
            q = q.filter(PolicyVersion.environment == environment)
        return sorted([row[0] for row in q.all()])

    def diff(self, version_id_a: str, version_id_b: str) -> Dict[str, Any]:
        """Compute a line-level diff between two policy versions.

        Args:
            version_id_a: ID of the first (older) version.
            version_id_b: ID of the second (newer) version.

        Returns:
            Dict with keys: policy_name, version_a, version_b, added, removed.
        """
        import difflib
        a = self.get_version(version_id_a)
        b = self.get_version(version_id_b)
        lines_a = a.content.splitlines(keepends=True)
        lines_b = b.content.splitlines(keepends=True)
        added, removed = [], []
        for line in difflib.unified_diff(lines_a, lines_b, lineterm=""):
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:].rstrip())
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:].rstrip())
        return {"policy_name": a.policy_name, "version_a": a.version, "version_b": b.version,
                "environment": a.environment, "added_lines": len(added), "removed_lines": len(removed),
                "added": added, "removed": removed}

    def rollback(self, policy_name: str, target_version_id: str, reason: str, approver: str) -> PolicyDeployment:
        """Roll back a policy to a specific previous version.

        Args:
            policy_name: Name of the policy to roll back.
            target_version_id: ID of the version to restore.
            reason: Reason for the rollback.
            approver: Email of the approver performing the rollback.

        Returns:
            PolicyDeployment record for this rollback.

        Raises:
            PolicyNotFoundError: If the target version does not exist.
            PolicyRollbackError: If the target is already active or belongs to different policy.
        """
        target = self.get_version(target_version_id)
        if target.policy_name != policy_name:
            raise PolicyRollbackError(f"Version {target_version_id} does not belong to policy '{policy_name}'")
        if target.is_active:
            raise PolicyRollbackError(f"Version {target.version} is already active")
        self.db.query(PolicyVersion).filter(
            PolicyVersion.policy_name == policy_name,
            PolicyVersion.environment == target.environment,
            PolicyVersion.is_active == True,  # noqa: E712
        ).update({"is_active": False})
        target.is_active = True
        deployment = PolicyDeployment(
            id=str(uuid.uuid4()), policy_version_id=target.id, environment=target.environment,
            deployed_by=approver, deployment_type="rollback", status="success", notes=reason,
        )
        self.db.add(deployment)
        self.db.commit()
        self.db.refresh(deployment)
        logger.info("Rolled back %s/%s to %s by %s", policy_name, target.environment, target.version, approver)
        return deployment

    def promote(self, policy_name: str, from_env: str, to_env: str, approver: str) -> PolicyDeployment:
        """Promote the active policy from one environment to the next.

        Args:
            policy_name: Name of the policy to promote.
            from_env: Source environment.
            to_env: Target environment.
            approver: Email of the approver.

        Returns:
            PolicyDeployment record for this promotion.

        Raises:
            InvalidPromotionPathError: If the promotion path is invalid.
            PolicyNotFoundError: If no active version exists in source env.
        """
        if not self._valid_promotion(from_env, to_env):
            raise InvalidPromotionPathError(
                f"Cannot promote from '{from_env}' to '{to_env}'. Valid path: {' -> '.join(PROMOTION_PATH)}")
        source = (self.db.query(PolicyVersion)
                  .filter(PolicyVersion.policy_name == policy_name,
                          PolicyVersion.environment == from_env,
                          PolicyVersion.is_active == True)  # noqa: E712
                  .first())
        if not source:
            raise PolicyNotFoundError(f"No active version of '{policy_name}' in '{from_env}'")
        promoted = self.store_version(
            policy_name=policy_name, content=source.content, engine=source.engine,
            author=approver, environment=to_env, commit_sha=source.commit_sha,
            commit_message=f"Promoted from {from_env} by {approver}",
            change_summary=f"Promoted from {from_env} (version {source.version})",
        )
        deployment = PolicyDeployment(
            id=str(uuid.uuid4()), policy_version_id=promoted.id, environment=to_env,
            deployed_by=approver, deployment_type="promotion", status="success",
            notes=f"Promoted from {from_env}",
        )
        self.db.add(deployment)
        self.db.commit()
        self.db.refresh(deployment)
        logger.info("Promoted %s from %s to %s by %s", policy_name, from_env, to_env, approver)
        return deployment

    def handle_webhook(self, payload: Dict[str, Any], signature: Optional[str]) -> Dict[str, Any]:
        """Process a Git push webhook event.

        Args:
            payload: Parsed JSON webhook payload.
            signature: Optional HMAC-SHA256 signature header value.

        Returns:
            Dict with processed counts and any errors.

        Raises:
            ValueError: If signature validation fails.
        """
        webhook_secret = getattr(settings, "gitops_webhook_secret", None)
        if webhook_secret and signature:
            self._verify_signature(payload, signature, webhook_secret)
        ref = payload.get("ref", "")
        environment = self._ref_to_environment(ref)
        if not environment:
            return {"status": "skipped", "reason": f"No environment mapping for ref '{ref}'"}
        commits = payload.get("commits", [])
        author = payload.get("pusher", {}).get("email", "webhook@gitops")
        stored, skipped, errors = 0, 0, []
        for commit in commits:
            commit_sha = commit.get("id", "")[:40]
            commit_message = commit.get("message", "")
            for file_path in commit.get("added", []) + commit.get("modified", []):
                try:
                    engine = self._detect_engine(file_path)
                    if not engine:
                        skipped += 1
                        continue
                    content = commit.get("_file_contents", {}).get(file_path, "")
                    if not content:
                        skipped += 1
                        continue
                    policy_name = self._path_to_policy_name(file_path)
                    self.store_version(policy_name=policy_name, content=content, engine=engine,
                                       author=author, environment=environment, commit_sha=commit_sha,
                                       commit_message=commit_message)
                    stored += 1
                except Exception as exc:
                    logger.exception("Failed to process %s", file_path)
                    errors.append({"file": file_path, "error": str(exc)})
        return {"status": "processed", "environment": environment, "stored": stored,
                "skipped": skipped, "errors": errors}

    def request_approval(self, policy_version_id: str, requested_by: str,
                         comments: Optional[str] = None) -> PolicyApproval:
        """Create a new approval request for a policy version.

        Args:
            policy_version_id: ID of the version requiring approval.
            requested_by: Email of the requester.
            comments: Optional comments.

        Returns:
            PolicyApproval ORM object.
        """
        self.get_version(policy_version_id)
        approval = PolicyApproval(
            id=str(uuid.uuid4()), policy_version_id=policy_version_id,
            requested_by=requested_by, status="pending", comments=comments,
        )
        self.db.add(approval)
        self.db.commit()
        self.db.refresh(approval)
        return approval

    def resolve_approval(self, approval_id: str, approved_by: str, decision: str,
                         comments: Optional[str] = None) -> PolicyApproval:
        """Approve or reject a pending approval request.

        Args:
            approval_id: ID of the approval.
            approved_by: Email of the admin resolving it.
            decision: 'approved' or 'rejected'.
            comments: Optional resolution comments.

        Returns:
            Updated PolicyApproval ORM object.

        Raises:
            PolicyApprovalError: If approval not found or already resolved.
        """
        if decision not in ("approved", "rejected"):
            raise PolicyApprovalError(f"Invalid decision '{decision}'")
        approval = self.db.query(PolicyApproval).filter(PolicyApproval.id == approval_id).first()
        if not approval:
            raise PolicyApprovalError(f"Approval not found: {approval_id}")
        if approval.status != "pending":
            raise PolicyApprovalError(f"Approval {approval_id} is already '{approval.status}'")
        approval.approved_by = approved_by
        approval.status = decision
        approval.resolved_at = datetime.now(timezone.utc)
        if comments:
            approval.comments = comments
        self.db.commit()
        self.db.refresh(approval)
        return approval

    def list_approvals(self, status: Optional[str] = None) -> List[PolicyApproval]:
        """List approval requests, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            List of PolicyApproval objects.
        """
        q = self.db.query(PolicyApproval)
        if status:
            q = q.filter(PolicyApproval.status == status)
        return q.order_by(PolicyApproval.requested_at.desc()).all()

    def list_deployments(self, environment: Optional[str] = None) -> List[PolicyDeployment]:
        """List deployment records.

        Args:
            environment: Optional filter by environment.

        Returns:
            List of PolicyDeployment objects.
        """
        q = self.db.query(PolicyDeployment)
        if environment:
            q = q.filter(PolicyDeployment.environment == environment)
        return q.order_by(PolicyDeployment.deployed_at.desc()).limit(200).all()

    def _next_version_tag(self, policy_name: str, environment: str) -> str:
        count = (self.db.query(PolicyVersion)
                 .filter(PolicyVersion.policy_name == policy_name,
                         PolicyVersion.environment == environment).count())
        return f"v{count + 1}"

    def _valid_promotion(self, from_env: str, to_env: str) -> bool:
        if from_env not in PROMOTION_PATH or to_env not in PROMOTION_PATH:
            return False
        return PROMOTION_PATH.index(to_env) == PROMOTION_PATH.index(from_env) + 1

    def _ref_to_environment(self, ref: str) -> Optional[str]:
        mapping = {"refs/heads/main": "prod", "refs/heads/master": "prod",
                   "refs/heads/staging": "staging", "refs/heads/develop": "dev", "refs/heads/dev": "dev"}
        return mapping.get(ref)

    def _detect_engine(self, file_path: str) -> Optional[str]:
        if file_path.endswith(".cedar"):
            return "cedar"
        if file_path.endswith(".rego"):
            return "opa"
        return None

    def _path_to_policy_name(self, file_path: str) -> str:
        import os
        base = os.path.basename(file_path)
        name, _ = os.path.splitext(base)
        return name.replace("_", "-").lower()

    def _verify_signature(self, payload: Dict[str, Any], signature: str, secret: str) -> None:
        import json
        key = secret.encode()
        body = json.dumps(payload, separators=(",", ":")).encode()
        expected = "sha256=" + hmac.new(key, body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("Invalid webhook signature")


def get_gitops_service(db: Session) -> "GitOpsService":
    """Dependency injection factory for GitOpsService.

    Args:
        db: SQLAlchemy session.

    Returns:
        GitOpsService instance.
    """
    return GitOpsService(db)
