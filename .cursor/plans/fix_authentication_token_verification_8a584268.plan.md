---
name: Fix Authentication Token Verification
overview: Fix the 401 Unauthorized errors by properly verifying Supabase JWT tokens in the backend. The current implementation uses get_user() incorrectly - we need to verify the JWT token directly.
todos: []
---

# Fix Authentication Token Verification

## Problem

Backend returns 401 Unauthorized for authenticated requests because `supabase.auth.get_user(token)` doesn't work correctly with JWT tokens from the frontend. The service role key client can't verify anon key JWTs.

## Solution

Use JWT verification library (PyJWT) to verify Supabase JWT tokens directly, or use the anon key client for token verification.

## Changes

### Backend: [`twilight_api.py`](twilight_api.py)

1. **Add JWT verification function** (after line 115):

- Import `jwt` from `jose` library (already in requirements.txt as part of supabase)
- Or use `PyJWT` if available
- Create `_verify_jwt_token()` function that:
- Gets Supabase JWT secret from environment (SUPABASE_JWT_SECRET or derive from anon key)
- Verifies the JWT token signature
- Extracts user_id and email from token claims
- Returns user info dict

2. **Update `verify_token()` function** (line 125-161):

- Replace `supabase.auth.get_user(token)` call
- Use new `_verify_jwt_token()` function instead
- Keep the user profile lookup from database
- Keep error handling the same

3. **Alternative approach** (if JWT secret not available):

- Create a separate Supabase client with anon key for token verification
- Use that client's `auth.get_user()` method
- Keep service role client for database operations

## Implementation Details

**Option 1: JWT Verification (Recommended)**

- Use `jose` library (already installed via supabase)
- Get JWT secret from Supabase project settings
- Verify token signature and extract claims

**Option 2: Anon Key Client**

- Create second Supabase client with anon key
- Use it only for `auth.get_user()` calls
- Keep service role client for database writes

## Testing

After fix:

- Anonymous chat should still work (200 OK)
- Authenticated `/me` should return user info (200 OK)
- Authenticated `/chat` should work with limits (200 OK)
- Invalid tokens should return 401