# -*- coding: utf-8 -*-
"""Location: ./mcpgateway/routers/gitops.py
Copyright 2026
SPDX-License-Identifier: Apache-2.0
Authors: Ioannis Ioannou

Policy GitOps REST API router.

Related to Issue #2238: Policy GitOps and version control

Examples:
    >>> True
    True
"""
# Future
from __future__ import annotations

# Standard
import logging
from typing import Optional

# Third-Party
from fastapi import APIRouter, Depends, HTTPException, Header, Query, status

# First-Party
from mcpgateway.auth import get_current_user
from mcpgateway.db import get_db
from mcpgateway.schemas import (
    PolicyVersionResponse,
    PolicyDeploymentResponse,
    PolicyApprovalResponse,
    PolicyDiffResponse,
    StoreVersionRequest,
    RollbackRequest,
    PromoteRequest,
    WebhookPayload,
    ApprovalRequest,
    ResolveApprovalRequest,
)
from mcpgateway.services.gitops_service import (
    get_gitops_service,
    GitOpsService,
    PolicyNotFoundError,
    PolicyRollbackError,
    InvalidPromotionPathError,
    PolicyApprovalError,
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gitops", tags=["Policy GitOps"])


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@router.get("/policies", response_model=list[str], summary="List all policy names")
def list_policies(
    environment: Optional[str] = Query(None, description="Filter by environment"),
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """List all distinct policy names stored in the system."""
    svc: GitOpsService = get_gitops_service(db)
    return svc.list_policies(environment=environment)


@router.post("/policies", response_model=PolicyVersionResponse, status_code=status.HTTP_201_CREATED,
             summary="Store a new policy version")
def store_version(
    body: StoreVersionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Store a new policy version. Deduplicates by content hash."""
    svc: GitOpsService = get_gitops_service(db)
    pv = svc.store_version(
        policy_name=body.policy_name,
        content=body.content,
        engine=body.engine,
        author=current_user.email,
        environment=body.environment,
        commit_sha=body.commit_sha,
        commit_message=body.commit_message,
        change_summary=body.change_summary,
    )
    return pv


@router.get("/policies/{policy_name}/versions", response_model=list[PolicyVersionResponse],
            summary="Get version history for a policy")
def get_history(
    policy_name: str,
    environment: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Return version history for a named policy, newest first."""
    svc: GitOpsService = get_gitops_service(db)
    return svc.get_history(policy_name=policy_name, environment=environment, limit=limit)


@router.get("/policies/versions/{version_id}", response_model=PolicyVersionResponse,
            summary="Get a single policy version by ID")
def get_version(
    version_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Fetch a specific policy version by its UUID."""
    svc: GitOpsService = get_gitops_service(db)
    try:
        return svc.get_version(version_id)
    except PolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/policies/versions/{version_id_a}/diff/{version_id_b}",
            response_model=PolicyDiffResponse, summary="Diff two policy versions")
def diff_versions(
    version_id_a: str,
    version_id_b: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Return a line-level diff between two policy versions."""
    svc: GitOpsService = get_gitops_service(db)
    try:
        return svc.diff(version_id_a, version_id_b)
    except PolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

@router.post("/policies/{policy_name}/rollback", response_model=PolicyDeploymentResponse,
             summary="Rollback a policy to a previous version")
def rollback(
    policy_name: str,
    body: RollbackRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Roll back a policy to a specific previous version."""
    svc: GitOpsService = get_gitops_service(db)
    try:
        return svc.rollback(
            policy_name=policy_name,
            target_version_id=body.target_version_id,
            reason=body.reason,
            approver=current_user.email,
        )
    except PolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PolicyRollbackError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

@router.post("/policies/{policy_name}/promote", response_model=PolicyDeploymentResponse,
             summary="Promote a policy to the next environment")
def promote(
    policy_name: str,
    body: PromoteRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Promote the active policy version from one environment to the next."""
    svc: GitOpsService = get_gitops_service(db)
    try:
        return svc.promote(
            policy_name=policy_name,
            from_env=body.from_env,
            to_env=body.to_env,
            approver=current_user.email,
        )
    except (PolicyNotFoundError,) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidPromotionPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

@router.post("/webhook", summary="Receive a Git push webhook event")
def webhook(
    payload: WebhookPayload,
    x_hub_signature_256: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Process a Git push event. No auth required — validated by HMAC signature."""
    svc: GitOpsService = get_gitops_service(db)
    try:
        result = svc.handle_webhook(
            payload=payload.model_dump(),
            signature=x_hub_signature_256,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        logger.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

@router.get("/approvals", response_model=list[PolicyApprovalResponse], summary="List approval requests")
def list_approvals(
    status: Optional[str] = Query(None, description="Filter by status: pending, approved, rejected"),
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """List all policy approval requests."""
    svc: GitOpsService = get_gitops_service(db)
    return svc.list_approvals(status=status)


@router.post("/approvals", response_model=PolicyApprovalResponse, status_code=status.HTTP_201_CREATED,
             summary="Request approval for a policy version")
def request_approval(
    body: ApprovalRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create an approval request for a policy version before deployment."""
    svc: GitOpsService = get_gitops_service(db)
    try:
        return svc.request_approval(
            policy_version_id=body.policy_version_id,
            requested_by=current_user.email,
            comments=body.comments,
        )
    except PolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/approvals/{approval_id}/resolve", response_model=PolicyApprovalResponse,
             summary="Approve or reject a pending approval")
def resolve_approval(
    approval_id: str,
    body: ResolveApprovalRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Approve or reject a pending policy approval request."""
    svc: GitOpsService = get_gitops_service(db)
    try:
        return svc.resolve_approval(
            approval_id=approval_id,
            approved_by=current_user.email,
            decision=body.decision,
            comments=body.comments,
        )
    except PolicyApprovalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------

@router.get("/deployments", response_model=list[PolicyDeploymentResponse], summary="List deployments")
def list_deployments(
    environment: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """List recent policy deployment records."""
    svc: GitOpsService = get_gitops_service(db)
    return svc.list_deployments(environment=environment)
