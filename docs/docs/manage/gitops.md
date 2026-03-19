# Policy GitOps and Version Control

The Policy GitOps feature provides version-controlled policy management with full audit trails, rollback capabilities, and Git-native workflows for Cedar and OPA policies.

---

## Overview

Policy changes follow a structured pipeline:
```
Git Push → Webhook → Validate → Store Version → Approval Gate → Deploy
                                                       ↓
                                               Promote: dev → staging → prod
                                                       ↓
                                               Rollback to any previous version
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GITOPS_ENABLED` | `true` | Enable or disable the GitOps feature |
| `GITOPS_WEBHOOK_SECRET` | `""` | HMAC-SHA256 secret for validating Git webhook signatures |
| `GITOPS_REQUIRE_APPROVAL_FOR_PROD` | `true` | Require approval before promoting to prod |
| `GITOPS_MIN_APPROVALS` | `1` | Minimum approvals required for prod deployment |
| `GITOPS_RETENTION_DAYS` | `90` | Days to retain policy version history |

---

## API Endpoints

All endpoints are under `/api/gitops` and require authentication unless noted.

### Policies

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/gitops/policies` | List all policy names |
| `POST` | `/api/gitops/policies` | Store a new policy version |
| `GET` | `/api/gitops/policies/{name}/versions` | Get version history |
| `GET` | `/api/gitops/policies/versions/{id}` | Get a specific version |
| `GET` | `/api/gitops/policies/versions/{a}/diff/{b}` | Diff two versions |
| `POST` | `/api/gitops/policies/{name}/rollback` | Rollback to a previous version |
| `POST` | `/api/gitops/policies/{name}/promote` | Promote to next environment |

### Webhook

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/gitops/webhook` | Receive Git push events (no auth, HMAC validated) |

### Approvals

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/gitops/approvals` | List approval requests |
| `POST` | `/api/gitops/approvals` | Request approval for a version |
| `POST` | `/api/gitops/approvals/{id}/resolve` | Approve or reject |

### Deployments

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/gitops/deployments` | List deployment records |

---

## Supported Policy Engines

| Engine | File Extension |
|---|---|
| Cedar | `.cedar` |
| OPA (Rego) | `.rego` |

---

## Environment Promotion

Policies can only be promoted along the following path:
```
dev → staging → prod
```

Skipping environments or promoting backwards is not permitted.

---

## Git Webhook

Configure your Git provider to send push events to `/api/gitops/webhook`.

**Branch to environment mapping:**

| Branch | Environment |
|---|---|
| `main` / `master` | `prod` |
| `staging` | `staging` |
| `develop` / `dev` | `dev` |

**Payload format** (GitHub-compatible):
```json
{
  "ref": "refs/heads/develop",
  "pusher": { "email": "ci@example.com" },
  "commits": [
    {
      "id": "abc123",
      "message": "Add tool access policy",
      "added": ["policies/tools/read-only.cedar"],
      "modified": [],
      "_file_contents": {
        "policies/tools/read-only.cedar": "permit(principal, action, resource);"
      }
    }
  ]
}
```

To validate webhook signatures set `GITOPS_WEBHOOK_SECRET` and send an `X-Hub-Signature-256` header.

---

## Example Workflow

**1. Store a new policy version:**
```bash
curl -X POST https://your-gateway/api/gitops/policies \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_name": "tool-access-control",
    "content": "permit(principal, action, resource);",
    "engine": "cedar",
    "environment": "dev",
    "change_summary": "Allow all actions for initial draft"
  }'
```

**2. View version history:**
```bash
curl https://your-gateway/api/gitops/policies/tool-access-control/versions \
  -H "Authorization: Bearer $TOKEN"
```

**3. Diff two versions:**
```bash
curl "https://your-gateway/api/gitops/policies/versions/{id_a}/diff/{id_b}" \
  -H "Authorization: Bearer $TOKEN"
```

**4. Request approval before promoting to prod:**
```bash
curl -X POST https://your-gateway/api/gitops/approvals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"policy_version_id": "{version_id}", "comments": "Ready for prod"}'
```

**5. Admin approves:**
```bash
curl -X POST https://your-gateway/api/gitops/approvals/{approval_id}/resolve \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"decision": "approved"}'
```

**6. Promote from dev to staging:**
```bash
curl -X POST https://your-gateway/api/gitops/policies/tool-access-control/promote \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"from_env": "dev", "to_env": "staging"}'
```

**7. Emergency rollback:**
```bash
curl -X POST https://your-gateway/api/gitops/policies/tool-access-control/rollback \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"target_version_id": "{version_id}", "reason": "Reverting due to regression"}'
```

---

## Security Considerations

- Webhook endpoints validate HMAC-SHA256 signatures when `GITOPS_WEBHOOK_SECRET` is set.
- All policy operations are authenticated and logged.
- Production promotions require explicit approval when `GITOPS_REQUIRE_APPROVAL_FOR_PROD=true`.
- Version history is retained for audit purposes and never deleted within the retention window.
