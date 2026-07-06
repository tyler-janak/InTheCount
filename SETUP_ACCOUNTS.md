# Account & Subscription Setup (Supabase + Stripe)

This walks you through every manual step needed before the auth + paywall code
will work on the live site. **Estimated time: 30–45 minutes.** Do these in
roughly this order; the code already lives in `auth.py`, `billing.py`, and
`server.py` — you're just creating the accounts it needs to talk to.

---

## 1) Supabase project (auth + database)

### 1a. Create the project

1. Go to **https://supabase.com** → Sign up (free tier).
2. **New project** → name it `thebullpenbet` (or whatever), pick a region close
   to your Render service (Oregon if Render is US-West, etc.), set a strong
   database password — **save it somewhere**, you'll rarely need it.
3. Wait ~2 min for provisioning.

### 1b. Get the API keys

In the dashboard → **Project Settings → API**:

- **Project URL** → save as `SUPABASE_URL`
- **anon public** key → save as `SUPABASE_ANON_KEY`
- **service_role secret** key → save as `SUPABASE_SERVICE_ROLE_KEY` *(never
  expose this to the browser — server-only)*

### 1c. Create the `subscriptions` table

In the dashboard → **SQL Editor** → New query → paste this and Run:

```sql
create table public.subscriptions (
  id                    uuid primary key default gen_random_uuid(),
  user_id               uuid not null references auth.users(id) on delete cascade,
  stripe_customer_id    text unique,
  stripe_subscription_id text unique,
  status                text not null default 'inactive',
  plan                  text,                          -- 'monthly' or 'annual'
  current_period_end    timestamptz,
  created_at            timestamptz not null default now(),
  updated_at            timestamptz not null default now()
);

create unique index subscriptions_user_id_key on public.subscriptions(user_id);

-- Row-level security: users can only read their own row.
alter table public.subscriptions enable row level security;

create policy "users can read own subscription"
  on public.subscriptions for select
  using (auth.uid() = user_id);

-- The server (service_role key) bypasses RLS automatically for writes
-- coming from the Stripe webhook.
```

### 1d. Email auth settings

Dashboard → **Authentication → Providers → Email**:
- Enable **Email** provider (it's on by default).
- Toggle **Confirm email** ON if you want users to verify their email before
  signing in. OFF for fastest onboarding.

Dashboard → **Authentication → URL Configuration**:
- **Site URL**: `https://thebullpenbet.onrender.com` (swap to your custom
  domain once you have one).

---

## 2) Stripe account (payments)

### 2a. Create the account

1. Go to **https://stripe.com** → Sign up.
2. Fill in business details. For "type of business" pick whatever applies —
   "Software" or "Information Services" is fine. **Don't** pick anything
   gambling-related; you're selling research/analytics.
3. You can run in **Test mode** indefinitely while building. Switch to
   **Live mode** when ready to take real money.

### 2b. Create the two products

Dashboard → **Products → + Add product**.

**Product 1: Monthly**
- Name: `TheBullpenBet — Monthly`
- Description: `Pitcher projections, hitter projections, NRFI picks, and Fantasy rankings.`
- Pricing model: **Recurring**
- Price: `$10.00 USD` / `monthly`
- Save it. Copy the **Price ID** (looks like `price_1Abc...`) → save as
  `STRIPE_PRICE_MONTHLY`.

**Product 2: Annual**
- Name: `TheBullpenBet — Annual`
- Pricing model: **Recurring**
- Price: `$75.00 USD` / `yearly`
- Save → copy Price ID → save as `STRIPE_PRICE_ANNUAL`.

### 2c. Get the API keys

Dashboard → **Developers → API keys**:
- **Publishable key** (`pk_test_...` or `pk_live_...`) → save as
  `STRIPE_PUBLISHABLE_KEY`.
- **Secret key** (`sk_test_...` or `sk_live_...`) → save as `STRIPE_SECRET_KEY`.

### 2d. Create the webhook endpoint

Dashboard → **Developers → Webhooks → + Add endpoint**:
- **Endpoint URL**: `https://thebullpenbet.onrender.com/api/billing/webhook`
- **Events to send** (pick these specifically):
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.payment_failed`
- Save → reveal the **Signing secret** (`whsec_...`) → save as
  `STRIPE_WEBHOOK_SECRET`.

### 2e. Enable the Customer Portal

Dashboard → **Settings → Billing → Customer portal**:
- Click **Activate test link** (Test mode) / **Save changes** (Live mode).
- Under "Functionality" enable: cancel subscriptions, update payment method,
  view invoice history.
- Save.

---

## 3) Add all the env vars to Render

Render dashboard → your web service → **Environment** tab → **Add Environment
Variable**. Add each of these:

| Key                            | Value                                    |
|--------------------------------|------------------------------------------|
| `SUPABASE_URL`                 | (from 1b)                                |
| `SUPABASE_ANON_KEY`            | (from 1b)                                |
| `SUPABASE_SERVICE_ROLE_KEY`    | (from 1b — secret, server-only)          |
| `STRIPE_SECRET_KEY`            | (from 2c — secret)                       |
| `STRIPE_PUBLISHABLE_KEY`       | (from 2c — safe to expose)               |
| `STRIPE_WEBHOOK_SECRET`        | (from 2d)                                |
| `STRIPE_PRICE_MONTHLY`         | (from 2b)                                |
| `STRIPE_PRICE_ANNUAL`          | (from 2b)                                |
| `APP_BASE_URL`                 | `https://thebullpenbet.onrender.com`     |
| `PAYWALL_ENABLED`              | `true`                                   |

Save → Render auto-redeploys with the new env vars.

---

## 4) Local development (optional)

If you want to test locally, create a `.env` file at the project root with the
same keys (it's already gitignored). Then:

```bash
pip install -r requirements-server.txt
uvicorn server:app --reload --port 8000
```

For local Stripe webhook testing, install the Stripe CLI and run:

```bash
stripe listen --forward-to localhost:8000/api/billing/webhook
```

That command prints a `whsec_...` value — use **that** as your local
`STRIPE_WEBHOOK_SECRET` (different from the one Render uses).

---

## 5) Test the flow end-to-end

Once everything's deployed:

1. Visit your site → click **Sign up** → create an account with a real email
   (check inbox if confirmation is on).
2. Visit `/?preview=subscriber` — paywall should lift (sanity check the
   client-side blur removal works).
3. Log in → click **Subscribe Monthly** on the Pricing page → land on Stripe
   Checkout. In test mode, use card `4242 4242 4242 4242`, any future expiry,
   any CVC.
4. Pay → redirect back to site → reload → paywall should be lifted because the
   webhook fired and wrote `status='active'` to your `subscriptions` table.
5. Click **Account → Manage billing** → cancel the subscription on the Stripe
   Portal page → webhook fires again, sets `status='canceled'`, paywall comes
   back on the next reload.

If any of those steps fails, check Render logs and Stripe webhook delivery
logs (Stripe dashboard → Developers → Webhooks → your endpoint → "Recent
deliveries").

---

## What you've created when this is done

- A Supabase project holding your users + subscriptions
- A Stripe account billing in either test or live mode
- A live signup → checkout → webhook → unlock flow on your site
- A "Manage billing" link that hands users off to Stripe's hosted portal so
  you never touch credit card data

Go to live mode (Stripe dashboard → toggle top-right) only after you've
tested the full flow in test mode and reviewed the Stripe ToS for paid
sports-analytics services.
