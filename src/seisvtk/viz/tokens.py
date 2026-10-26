"""Dependency-free HS256 token signing/verification for the viewer service.

The viewer service is handed short-lived tokens that a host (e.g. the QEarth
backend) mints with a shared secret; it only needs to *verify* them, and keeping
that dependency-free avoids pulling a JWT library into seisvtk. Only HS256
(HMAC-SHA256) is supported — the one algorithm the host uses. ``sign_hs256`` is
provided as its symmetric counterpart for standalone use and for tests; a host
with its own JWT stack does not need it.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time


class TokenError(Exception):
    """A token was missing, malformed, badly signed, or outside its time window."""


def _finite_seconds(value, name: str) -> float:
    """A time claim must be a finite real number (not bool, nan, or inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TokenError(f"invalid {name} claim")
    seconds = float(value)
    if not math.isfinite(seconds):
        raise TokenError(f"invalid {name} claim")
    return seconds


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def sign_hs256(claims: dict, secret: str) -> str:
    """Mint an HS256 token for *claims* (seconds-since-epoch ``exp``/``nbf``)."""
    if not secret:
        raise TokenError("missing secret")
    header = _b64url_encode(b'{"alg":"HS256","typ":"JWT"}')
    payload = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header}.{payload}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(sig)}"


def verify_hs256(token: str, secret: str, *, leeway: float = 0.0,
                 require_exp: bool = False) -> dict:
    """Return the claims of a valid HS256 *token*, or raise :class:`TokenError`.

    Verifies the signature in constant time and, when present, the ``exp`` and
    ``nbf`` time claims (allowing *leeway* seconds of clock skew). The ``alg``
    header must be ``HS256``. With ``require_exp`` a token without an ``exp`` is
    rejected (a viewer-entry token in a URL must not be valid forever). Any
    malformed input raises :class:`TokenError`, never a different exception.
    """
    if not token or not secret:
        raise TokenError("missing token or secret")
    parts = token.split(".")
    if len(parts) != 3:
        raise TokenError("not a three-part JWT")
    header_b64, payload_b64, sig_b64 = parts
    try:
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(payload_b64))
        signature = _b64url_decode(sig_b64)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad token
        raise TokenError(f"malformed token: {exc}") from exc
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise TokenError("token header and payload must be JSON objects")
    if header.get("alg") != "HS256":
        raise TokenError(f"unsupported alg {header.get('alg')!r}")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, signature):
        raise TokenError("bad signature")
    now = time.time()
    if "exp" in claims:
        if now > _finite_seconds(claims["exp"], "exp") + leeway:
            raise TokenError("token expired")
    elif require_exp:
        raise TokenError("token missing exp")
    if "nbf" in claims:
        if now < _finite_seconds(claims["nbf"], "nbf") - leeway:
            raise TokenError("token not yet valid")
    return claims
