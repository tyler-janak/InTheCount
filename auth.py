"""
auth.py — Supabase-backed authentication for TheBullpenBet.

Responsibilities
================
* Holds the two Supabase clients we need:
    - `supabase_anon` for user-facing auth calls (signup, login, etc.)
      — uses the public anon key, talks to the Supabase Auth REST API.
    - `supabase_service` for server-side reads/writes of `subscriptions`
      — uses the service role key, bypasses Row-Level Security.
* Exposes a `current_user` FastAPI dependency that:
    1. Reads the JWT from the `Authorization: Bearer <token>` header OR
       from a `sb_access_token` cookie set on login.
    2. Asks Supabase to verify the token and return the user record.
    3. Joins the `subscriptions` row so callers know `is_subscribed`,
       `plan`, and `current_period_end` without an extra round-trip.
* Provides convenience helpers to upsert and read subscription rows
  from the Stripe webhook handler.

Env vars required (see SETUP_ACCOUNTS.md):
    SUPABASE_URL
    SUPABASE_ANON_KEY
    SUPABASE_SERVICE_ROLE_KEY
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Cookie, Header, HTTPException, status
from supabase import Client, create_client


# ───── Config / client construction ────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# If env vars aren't set yet (local dev pre-Supabase), keep the import
# usable so the rest of the app can boot. Auth endpoints will 503.
_clients_ready = bool(SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_SERVICE_ROLE_KEY)

supabase_anon: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_ANON_KEY) if _clients_ready else None
)
supabase_service: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if _clients_ready else None
)


def auth_configured() -> bool:
    """Return True when the Supabase env vars are present and clients ready."""
    return _clients_ready


def _require_clients() -> None:
    if not _clients_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth is not configured on this deployment yet. "
                   "See SETUP_ACCOUNTS.md.",
        )


# ───── Subscription helpers ────────────────────────────────────
def get_subscription(user_id: str) -> dict[str, Any] | None:
    """Return the subscriptions row for a user, or None if there isn't one."""
    if not _clients_ready:
        return None
    res = (
        supabase_service.table("subscriptions")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None


def is_user_subscribed(sub_row: dict[str, Any] | None) -> bool:
    """A subscription is 'live' if status is one of active/trialing AND the
    period hasn't ended yet (small grace included to avoid clock skew)."""
    if not sub_row:
        return False
    status_ = (sub_row.get("status") or "").lower()
    if status_ not in {"active", "trialing"}:
        return False
    period_end = sub_row.get("current_period_end")
    if not period_end:
        return True   # status active but no end — trust it (e.g. lifetime)
    try:
        if isinstance(period_end, str):
            period_end = datetime.fromisoformat(period_end.replace("Z", "+00:00"))
        return period_end > datetime.now(timezone.utc)
    except (ValueError, TypeError):
        return True   # parsing fail → don't accidentally lock out a paying user


def upsert_subscription(*, user_id: str, **fields: Any) -> None:
    """Insert or update a subscriptions row for the given user."""
    _require_clients()
    payload = {"user_id": user_id, "updated_at": datetime.now(timezone.utc).isoformat()}
    payload.update({k: v for k, v in fields.items() if v is not None})
    supabase_service.table("subscriptions").upsert(payload, on_conflict="user_id").execute()


def find_user_by_stripe_customer(customer_id: str) -> str | None:
    """Reverse lookup: given a Stripe customer_id, return the Supabase user_id."""
    if not _clients_ready or not customer_id:
        return None
    res = (
        supabase_service.table("subscriptions")
        .select("user_id")
        .eq("stripe_customer_id", customer_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0]["user_id"] if rows else None


# ───── Auth dependency ────────────────────────────────────────
def _extract_token(
    authorization: str | None,
    sb_access_token: str | None,
) -> str | None:
    """Pull the JWT out of either the Authorization header or the cookie."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return sb_access_token


async def current_user(
    authorization: str | None = Header(default=None),
    sb_access_token: str | None = Cookie(default=None),
) -> dict[str, Any]:
    """FastAPI dependency: returns the logged-in Supabase user dict with
    a `subscription` block joined on. Raises 401 if not signed in."""
    _require_clients()
    token = _extract_token(authorization, sb_access_token)
    if not token:
        raise HTTPException(status_code=401, detail="Not signed in")
    try:
        user_response = supabase_anon.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid session")
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid session: {exc}") from exc
    sub = get_subscription(user.id)
    return {
        "id": user.id,
        "email": user.email,
        "subscription": sub,
        "is_subscribed": is_user_subscribed(sub),
    }


async def current_user_optional(
    authorization: str | None = Header(default=None),
    sb_access_token: str | None = Cookie(default=None),
) -> dict[str, Any] | None:
    """Like `current_user` but returns None instead of raising for anonymous
    visitors. Useful for `/api/me` to return a "logged out" body cleanly."""
    if not _clients_ready:
        return None
    token = _extract_token(authorization, sb_access_token)
    if not token:
        return None
    try:
        user_response = supabase_anon.auth.get_user(token)
        user = user_response.user
        if not user:
            return None
    except Exception:
        return None
    sub = get_subscription(user.id)
    return {
        "id": user.id,
        "email": user.email,
        "subscription": sub,
        "is_subscribed": is_user_subscribed(sub),
    }
