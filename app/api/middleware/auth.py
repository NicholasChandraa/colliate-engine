"""
JWT auth middleware for engine.
Validates tokens signed by auth-service using the shared JWT_SECRET_KEY.
"""
import jwt as pyjwt
from fastapi import Header, HTTPException
from app.core.config import get_settings


async def get_optional_user_id(
    authorization: str | None = Header(default=None)
) -> str | None:
    """Returns user_id if a valid Bearer token is present, otherwise None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    try:
        settings = get_settings()
        payload = pyjwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
        return payload.get("user_id")
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def require_user_id(
    authorization: str | None = Header(default=None)
) -> str:
    """Returns user_id or raises 401 if token is missing/invalid."""
    user_id = await get_optional_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id
