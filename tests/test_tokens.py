"""HS256 token signing/verification (the viewer's shared-secret auth primitive)."""
import time

import pytest

from seisvtk.viz.tokens import TokenError, sign_hs256, verify_hs256

SECRET = "a-shared-secret"


def test_roundtrip_preserves_claims():
    claims = {"scene": "s3://bucket/run756/GR_R.zarr", "chart": 42,
              "exp": time.time() + 60}
    token = sign_hs256(claims, SECRET)
    assert verify_hs256(token, SECRET) == claims


def test_expired_token_rejected():
    token = sign_hs256({"scene": "x", "exp": time.time() - 1}, SECRET)
    with pytest.raises(TokenError):
        verify_hs256(token, SECRET)


def test_leeway_allows_small_clock_skew():
    token = sign_hs256({"scene": "x", "exp": time.time() - 5}, SECRET)
    assert verify_hs256(token, SECRET, leeway=30)["scene"] == "x"


def test_not_yet_valid_token_rejected():
    token = sign_hs256({"scene": "x", "nbf": time.time() + 60}, SECRET)
    with pytest.raises(TokenError):
        verify_hs256(token, SECRET)


def test_wrong_secret_rejected():
    token = sign_hs256({"scene": "x"}, SECRET)
    with pytest.raises(TokenError):
        verify_hs256(token, "other-secret")


def test_tampered_payload_rejected():
    token = sign_hs256({"scene": "x", "admin": False}, SECRET)
    header, _payload, sig = token.split(".")
    forged = sign_hs256({"scene": "x", "admin": True}, "attacker").split(".")[1]
    with pytest.raises(TokenError):
        verify_hs256(f"{header}.{forged}.{sig}", SECRET)


def test_non_hs256_alg_rejected():
    # A token with alg=none (the classic JWT downgrade attack) must not verify.
    import base64
    import json

    def b64(raw):
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = b64(json.dumps({"scene": "x"}).encode())
    with pytest.raises(TokenError):
        verify_hs256(f"{header}.{payload}.", SECRET)


def test_malformed_and_empty_tokens_rejected():
    for bad in ["", "notajwt", "only.two", "a.b.c.d"]:
        with pytest.raises(TokenError):
            verify_hs256(bad, SECRET)


def test_missing_secret_rejected():
    with pytest.raises(TokenError):
        verify_hs256(sign_hs256({"scene": "x"}, SECRET), "")


def test_require_exp_rejects_token_without_exp():
    token = sign_hs256({"scene": "x"}, SECRET)
    assert verify_hs256(token, SECRET)["scene"] == "x"      # fine without the flag
    with pytest.raises(TokenError):
        verify_hs256(token, SECRET, require_exp=True)


def test_non_numeric_or_non_finite_exp_raises_tokenerror():
    # A signed-but-garbage exp must raise TokenError, never crash the caller,
    # and nan/inf must not silently skip the expiry check.
    for bad in ["soon", float("nan"), float("inf"), True]:
        token = sign_hs256({"scene": "x", "exp": bad}, SECRET)
        with pytest.raises(TokenError):
            verify_hs256(token, SECRET)


def test_non_object_payload_raises_tokenerror():
    import base64
    import hashlib
    import hmac
    import json

    def b64(raw):
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = b64(json.dumps([1, 2, 3]).encode())            # a list, not an object
    sig = b64(hmac.new(SECRET.encode(), f"{header}.{payload}".encode(),
                       hashlib.sha256).digest())
    with pytest.raises(TokenError):
        verify_hs256(f"{header}.{payload}.{sig}", SECRET)
