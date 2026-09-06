"""Tracks the health/status of the OpenRouter model, and sends a one-time
email alert if the model is deprecated/removed or persistently degraded.
Combines two cheap signals:
1. Passive results from real chat completions (no extra quota used).
2. A periodic check against OpenRouter's free /models listing endpoint,
   which costs no generation tokens, just to confirm the configured
   model is still available.
"""
import asyncio
import logging
import smtplib
import time
from collections import deque
from email.message import EmailMessage
from typing import Deque, Optional, Tuple
import httpx
from config import settings
logger = logging.getLogger(__name__)
_CALL_WINDOW_SECONDS = 15 * 60
_call_history: Deque[Tuple[float, bool]] = deque(maxlen=50)
# Whether the last *confirmed* /models listing check found the configured
# model. None = not yet confirmed either way (e.g. only network errors so far).
_model_listed: Optional[bool] = None
_last_check_ts: float = 0.0
# When the model was first confirmed missing from the /models list.
_model_unavailable_since: Optional[float] = None
# When the overall status first became non-green (yellow/red); reset to
# None whenever status returns to green.
_degraded_since: Optional[float] = None
# Ensures only one alert email is sent per running process, by default.
_alert_sent: bool = False
_lock = asyncio.Lock()
DEGRADED_ALERT_THRESHOLD_SECONDS = (
    getattr(settings, "ALERT_DEGRADED_THRESHOLD_HOURS", 24.0) * 3600
)
def record_call_result(success: bool) -> None:
    """Record the outcome of a real call_openrouter invocation."""
    _call_history.append((time.time(), success))
def _recent_results(now: float) -> list:
    return [s for (t, s) in _call_history if now - t <= _CALL_WINDOW_SECONDS]
async def check_model_availability() -> Optional[bool]:
    """Ping OpenRouter's free /models endpoint to verify the configured
    model is still listed. Uses no generation quota.
    Returns True/False if confirmed either way, or None if the check
    itself failed (e.g. network issue) and availability could not be
    determined — in which case we don't assume the model is gone.
    """
    global _model_listed, _last_check_ts
    url = f"{settings.OPENROUTER_URL}/models"
    headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            model_ids = {m.get("id") for m in data.get("data", [])}
            listed = settings.OPENROUTER_MODEL in model_ids
            async with _lock:
                _model_listed = listed
                _last_check_ts = time.time()
            if not listed:
                logger.warning(
                    "Configured model '%s' not found in OpenRouter /models list",
                    settings.OPENROUTER_MODEL,
                )
            return listed
    except Exception as exc:
        # A failed request doesn't necessarily mean the model is gone - it
        # could be a transient network/API issue. Don't overwrite
        # _model_listed in that case, just log it.
        logger.error("Failed to check model availability: %s", exc)
        async with _lock:
            _last_check_ts = time.time()
        return None
async def get_status() -> dict:
    """Compute overall status: 'green', 'yellow', or 'red'."""
    now = time.time()
    recent = _recent_results(now)
    async with _lock:
        model_listed = _model_listed
        last_check = _last_check_ts
    if model_listed is False:
        return {
            "status": "red",
            "reason": "model_unavailable",
            "last_checked": last_check,
        }
    if not recent:
        return {"status": "green", "reason": "idle", "last_checked": last_check}
    successes = sum(1 for s in recent if s)
    ratio = successes / len(recent)
    if ratio >= 0.9:
        status = "green"
    elif ratio >= 0.5:
        status = "yellow"
    else:
        status = "red"
    return {
        "status": status,
        "reason": "recent_calls",
        "success_ratio": ratio,
        "samples": len(recent),
        "last_checked": last_check,
    }
def _send_email_sync(subject: str, body: str) -> None:
    """Blocking email send — meant to be run in an executor thread."""
    if not getattr(settings, "EMAIL_ALERTS_ENABLED", True):
        logger.info("Email alerts disabled, skipping send: %s", subject)
        return
    host = getattr(settings, "SMTP_HOST", "")
    to_addr = getattr(settings, "ALERT_EMAIL_TO", "")
    if not host or not to_addr:
        logger.warning(
            "Email alert skipped — SMTP not configured (subject: %s)", subject
        )
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = (
        getattr(settings, "SMTP_FROM_EMAIL", "") or "noreply@holochatstats.info"
    )
    msg["To"] = to_addr
    msg.set_content(body)
    port = getattr(settings, "SMTP_PORT", 587)
    username = getattr(settings, "SMTP_USERNAME", "")
    password = getattr(settings, "SMTP_PASSWORD", "")
    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)
        logger.info("Alert email sent: %s", subject)
    except Exception:
        logger.exception("Failed to send alert email: %s", subject)
async def send_alert_email(subject: str, body: str) -> None:
    """Send an alert email without blocking the event loop."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send_email_sync, subject, body)
async def _maybe_send_alert(reason: str) -> None:
    """Send the one-time alert email for this process, if not already sent."""
    global _alert_sent
    async with _lock:
        if _alert_sent:
            return
        _alert_sent = True
    model_name = settings.OPENROUTER_MODEL
    if reason == "unavailable":
        subject = (
            f"[HoloChatStats] Model '{model_name}' is no longer available "
            "on OpenRouter"
        )
        body = (
            f"The configured OpenRouter model '{model_name}' no longer "
            "appears in OpenRouter's list of available models. It may have "
            "been deprecated or removed.\n\n"
            "Please update OPENROUTER_MODEL in the service configuration.\n\n"
            "This is a one-time notification for this running instance."
        )
    else:  # "degraded"
        hours = DEGRADED_ALERT_THRESHOLD_SECONDS / 3600
        subject = (
            f"[HoloChatStats] Model '{model_name}' has been degraded for "
            f"over {hours:.0f} hours"
        )
        body = (
            f"The configured OpenRouter model '{model_name}' has been in a "
            f"degraded or failing state continuously for more than "
            f"{hours:.0f} hours.\n\n"
            "Please check OpenRouter's status and consider switching models.\n\n"
            "This is a one-time notification for this running instance."
        )
    await send_alert_email(subject, body)
async def evaluate_and_alert() -> None:
    """Evaluate current model health and fire an alert email if warranted.
    Called periodically from the background poller loop, after
    check_model_availability() has run.
    """
    global _model_unavailable_since, _degraded_since
    now = time.time()
    status_info = await get_status()
    status = status_info["status"]
    async with _lock:
        model_listed = _model_listed
    # --- Deprecation / removal check -----------------------------------
    if model_listed is False:
        async with _lock:
            if _model_unavailable_since is None:
                _model_unavailable_since = now
        await _maybe_send_alert("unavailable")
        return  # no need to also evaluate the degraded case
    else:
        async with _lock:
            _model_unavailable_since = None
    # --- Continuous degradation check -----------------------------------
    if status == "green":
        async with _lock:
            _degraded_since = None
        return
    async with _lock:
        if _degraded_since is None:
            _degraded_since = now
        degraded_duration = now - _degraded_since
    if degraded_duration >= DEGRADED_ALERT_THRESHOLD_SECONDS:
        await _maybe_send_alert("degraded")
async def status_poller_loop(interval_seconds: int = 300) -> None:
    """Background task: periodically verify model availability and send
    alert emails if the model is deprecated or persistently degraded."""
    while True:
        try:
            await check_model_availability()
            await evaluate_and_alert()
        except Exception:
            logger.exception("Error in status poller loop")
        await asyncio.sleep(interval_seconds)