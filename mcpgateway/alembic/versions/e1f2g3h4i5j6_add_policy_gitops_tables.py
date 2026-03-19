# -*- coding: utf-8 -*-
"""Add policy GitOps tables.

Revision ID: e1f2g3h4i5j6
Revises: d2d3d4d5d6d7
Create Date: 2026-03-19

"""
# Third-Party
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1f2g3h4i5j6"
down_revision: str = "abf8ac3b6008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create policy_versions, policy_deployments, and policy_approvals tables."""
    inspector = sa.inspect(op.get_bind())
    existing = inspector.get_table_names()

    if "policy_versions" not in existing:
        op.create_table(
            "policy_versions",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("policy_name", sa.String(255), nullable=False, index=True),
            sa.Column("version", sa.String(50), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("content_hash", sa.String(64), nullable=False, index=True),
            sa.Column("engine", sa.String(50), nullable=False),
            sa.Column("author", sa.String(255), nullable=False),
            sa.Column("commit_sha", sa.String(40), nullable=True),
            sa.Column("commit_message", sa.Text(), nullable=True),
            sa.Column("change_summary", sa.Text(), nullable=True),
            sa.Column("environment", sa.String(50), nullable=False, index=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, default=False, index=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("policy_name", "version", "environment", name="uq_policy_version_env"),
        )

    if "policy_deployments" not in existing:
        op.create_table(
            "policy_deployments",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("policy_version_id", sa.String(36), sa.ForeignKey("policy_versions.id"), nullable=False),
            sa.Column("environment", sa.String(50), nullable=False, index=True),
            sa.Column("deployed_by", sa.String(255), nullable=False),
            sa.Column("deployment_type", sa.String(50), nullable=False),
            sa.Column("status", sa.String(50), nullable=False, default="success"),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )

    if "policy_approvals" not in existing:
        op.create_table(
            "policy_approvals",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("policy_version_id", sa.String(36), sa.ForeignKey("policy_versions.id"), nullable=False),
            sa.Column("requested_by", sa.String(255), nullable=False),
            sa.Column("approved_by", sa.String(255), nullable=True),
            sa.Column("status", sa.String(50), nullable=False, default="pending", index=True),
            sa.Column("comments", sa.Text(), nullable=True),
            sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    """Drop policy GitOps tables."""
    inspector = sa.inspect(op.get_bind())
    existing = inspector.get_table_names()

    if "policy_approvals" in existing:
        op.drop_table("policy_approvals")
    if "policy_deployments" in existing:
        op.drop_table("policy_deployments")
    if "policy_versions" in existing:
        op.drop_table("policy_versions")
