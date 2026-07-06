"""
billing.py — Stripe Checkout, Customer Portal, and webhook handler.

Flow
====
1. Logged-in user clicks "Subscribe Monthly" / "Subscribe Annual"
   → frontend POSTs /api/billing/checkout with {"plan": "monthly"|"annual"}
   → this module creates a Stripe Checkout session keyed to the user's id
     and returns the hosted URL. Frontend window.location = url.

2. User pays on Stripe's hosted page → Stripe redirects back to
   APP_BASE_URL/?subscription=success and fires `checkout.session.completed`
   to our webhook. Our webhook upserts the subscriptions row.

3. Future renewals / cancellations / payment failures all fire
   customer.subscription.{created,updated,deleted} events — handled by the
   same webhook so the subscriptions table is always Stripe's source of truth.

4. "Manage billing" → /api/billing/portal → creates a Stripe Customer Portal
   session → frontend redirects there. The Portal is Stripe's hosted UI for
   cancelling, swapping payment method, and viewing invoices.

Env vars required (see SETUP_ACCOUNTS.md):
    STRIPE_SECRET_KEY
    STRIPE_WEBHOOK_SECRET
    STRIPE_PRICE_MONTHLY
    STRIPE_PRICE_ANNUAL
    APP_BASE_URL
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import stripe

from auth import find_user_by_stripe_customer, upsert_subscription


STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_MONTHLY = os.environ.get("STRIPE_PRICE_MONTHLY", "")
STRIPE_PRICE_ANNUAL = os.environ.get("STRIPE_PRICE_ANNUAL", "")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://thebullpenbet.onrender.com")

# Configure the Stripe SDK (no-op if key is empty; calls will fail with
# a clear AuthenticationError that the API layer surfaces as a 503).
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

_PLAN_TO_PRICE = {
    "monthly": STRIPE_PRICE_MONTHLY,
    "annual":  STRIPE_PRICE_ANNUAL,
}


def billing_configured() -> bool:
    return bool(STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET
                and STRIPE_PRICE_MONTHLY and STRIPE_PRICE_ANNUAL)


# ───── Checkout session ────────────────────────────────────────
def create_checkout_session(*, user_id: str, email: str, plan: str) -> str:
    """Create a Stripe Checkout session for the given user + plan.

    Returns the hosted Checkout URL the frontend should redirect to.
    Raises ValueError for an unknown plan.
    """
    if not billing_configured():
        raise RuntimeError("Stripe is not configured on this deployment.")
    price_id = _PLAN_TO_PRICE.get(plan)
    if not price_id:
        raise ValueError(f"Unknown plan: {plan!r}")

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        customer_email=email,
        # client_reference_id lets the webhook tie the session back to the
        # Supabase user even before a Stripe customer exists.
        client_reference_id=user_id,
        success_url=f"{APP_BASE_URL}/?subscription=success",
        cancel_url=f"{APP_BASE_URL}/?subscription=canceled",
        allow_promotion_codes=True,
        metadata={"supabase_user_id": user_id, "plan": plan},
        subscription_data={
            "metadata": {"supabase_user_id": user_id, "plan": plan},
        },
    )
    return session.url


# ───── Customer Portal session ─────────────────────────────────
def create_portal_session(*, stripe_customer_id: str) -> str:
    """Return the URL of a Stripe Customer Portal session for the user."""
    if not billing_configured():
        raise RuntimeError("Stripe is not configured on this deployment.")
    if not stripe_customer_id:
        raise ValueError("This user has no Stripe customer record yet.")
    session = stripe.billing_portal.Session.create(
        customer=stripe_customer_id,
        return_url=f"{APP_BASE_URL}/?from=portal",
    )
    return session.url


# ───── Webhook handler ─────────────────────────────────────────
def verify_webhook(payload: bytes, sig_header: str) -> stripe.Event:
    """Verify the Stripe-Signature header. Raises stripe.error.SignatureVerificationError
    on tampering; raises ValueError on a malformed payload."""
    return stripe.Webhook.construct_event(
        payload=payload,
        sig_header=sig_header,
        secret=STRIPE_WEBHOOK_SECRET,
    )


def _plan_from_subscription(sub: Any) -> str | None:
    """Pull the human-readable plan name (monthly|annual) out of metadata,
    falling back to the price id if the metadata is missing."""
    md = (sub.get("metadata") if isinstance(sub, dict) else sub.metadata) or {}
    if md.get("plan"):
        return md["plan"]
    try:
        price_id = sub["items"]["data"][0]["price"]["id"]
    except (KeyError, IndexError, TypeError):
        return None
    if price_id == STRIPE_PRICE_MONTHLY:
        return "monthly"
    if price_id == STRIPE_PRICE_ANNUAL:
        return "annual"
    return None


def _user_id_from_subscription(sub: Any) -> str | None:
    """Pull our Supabase user_id out of the subscription's metadata, falling
    back to a lookup by stripe_customer_id."""
    md = (sub.get("metadata") if isinstance(sub, dict) else sub.metadata) or {}
    if md.get("supabase_user_id"):
        return md["supabase_user_id"]
    customer_id = sub.get("customer") if isinstance(sub, dict) else sub.customer
    return find_user_by_stripe_customer(customer_id) if customer_id else None


def handle_event(event: stripe.Event) -> dict[str, Any]:
    """Dispatch a verified Stripe event to the right handler. Returns a small
    debug-friendly summary that the API layer logs."""
    etype = event["type"]
    data = event["data"]["object"]
    result = {"event_type": etype, "handled": False}

    if etype == "checkout.session.completed":
        user_id = data.get("client_reference_id") or (
            data.get("metadata") or {}).get("supabase_user_id")
        if user_id:
            upsert_subscription(
                user_id=user_id,
                stripe_customer_id=data.get("customer"),
                stripe_subscription_id=data.get("subscription"),
                status="active",
                plan=(data.get("metadata") or {}).get("plan"),
            )
            result["handled"] = True
            result["user_id"] = user_id

    elif etype in ("customer.subscription.created",
                   "customer.subscription.updated"):
        user_id = _user_id_from_subscription(data)
        if user_id:
            cpe_ts = data.get("current_period_end")
            cpe_iso = (
                datetime.fromtimestamp(cpe_ts, tz=timezone.utc).isoformat()
                if cpe_ts else None
            )
            upsert_subscription(
                user_id=user_id,
                stripe_customer_id=data.get("customer"),
                stripe_subscription_id=data.get("id"),
                status=data.get("status"),
                plan=_plan_from_subscription(data),
                current_period_end=cpe_iso,
            )
            result["handled"] = True
            result["user_id"] = user_id

    elif etype == "customer.subscription.deleted":
        user_id = _user_id_from_subscription(data)
        if user_id:
            upsert_subscription(
                user_id=user_id,
                status="canceled",
            )
            result["handled"] = True
            result["user_id"] = user_id

    elif etype == "invoice.payment_failed":
        # Mark the subscription past_due so the paywall comes back; Stripe will
        # retry the charge automatically per the dunning settings.
        customer_id = data.get("customer")
        user_id = find_user_by_stripe_customer(customer_id) if customer_id else None
        if user_id:
            upsert_subscription(user_id=user_id, status="past_due")
            result["handled"] = True
            result["user_id"] = user_id

    return result
