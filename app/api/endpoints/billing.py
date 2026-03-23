"""Stripe billing & webhook endpoints."""
import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User

router = APIRouter(prefix="/billing", tags=["Billing"])
stripe.api_key = settings.STRIPE_SECRET_KEY


class CheckoutInput(BaseModel):
    token: str  # JWT from login
    plan: str   # "pro" or "business"


@router.post("/checkout")
async def create_checkout(body: CheckoutInput, db: AsyncSession = Depends(get_db)):
    """Create a Stripe Checkout session for upgrading."""
    from jose import jwt, JWTError

    try:
        payload = jwt.decode(body.token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(401, "Invalid token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    price_id = settings.STRIPE_PRO_PRICE_ID if body.plan == "pro" else settings.STRIPE_BUSINESS_PRICE_ID
    if not price_id:
        raise HTTPException(500, "Stripe price not configured")

    # Create or reuse Stripe customer
    if not user.stripe_customer_id:
        customer = stripe.Customer.create(email=user.email, metadata={"user_id": str(user.id)})
        user.stripe_customer_id = customer.id
        await db.commit()

    session = stripe.checkout.Session.create(
        customer=user.stripe_customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=f"{settings.BASE_URL}/dashboard?upgraded=true",
        cancel_url=f"{settings.BASE_URL}/pricing",
        metadata={"user_id": str(user.id), "plan": body.plan},
    )

    return {"checkout_url": session.url}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Stripe webhooks for subscription events."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig, settings.STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(400, "Invalid webhook signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = int(session["metadata"]["user_id"])
        plan = session["metadata"]["plan"]
        subscription_id = session.get("subscription")

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user:
            user.plan = plan
            user.stripe_subscription_id = subscription_id
            await db.commit()

    elif event["type"] in ("customer.subscription.deleted", "customer.subscription.paused"):
        subscription_id = event["data"]["object"]["id"]
        result = await db.execute(select(User).where(User.stripe_subscription_id == subscription_id))
        user = result.scalar_one_or_none()
        if user:
            user.plan = "free"
            user.stripe_subscription_id = None
            await db.commit()

    return {"status": "ok"}
