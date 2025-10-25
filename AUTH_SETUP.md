# Supabase Authentication Setup Guide

## Implementation Status: ✅ COMPLETE

All authentication components have been implemented! Follow the configuration steps below to get everything running.

## Database Setup ✅

Your Supabase database tables are already configured:
- **user_profiles** table (with RLS enabled)
  - Stores user tier (free/paid) and profile information
- **user_usage** table (with RLS enabled)
  - Tracks daily message counts per user

## Environment Configuration

### Backend Configuration

Create a `.env` file in the project root with:

```bash
# Supabase Configuration
SUPABASE_URL=https://vauomdmvwtjywdjvilsw.supabase.co
SUPABASE_SERVICE_KEY=YOUR_SERVICE_ROLE_KEY_HERE
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA

# OpenAI API Key (for chatbot)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Redis Configuration (for session storage)
# REDIS_URL=rediss://...
# SESSION_TTL_SECONDS=86400
```

**To get your SERVICE_ROLE_KEY:**
1. Go to your Supabase Dashboard
2. Navigate to Project Settings → API
3. Copy the `service_role` key (NOT the anon key)

### Frontend Configuration

Create a `frontend/.env.local` file with:

```bash
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=https://vauomdmvwtjywdjvilsw.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA

# API Backend URL
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Installation Steps ✅

### Backend
```bash
# Install Python dependencies (supabase package added to requirements.txt)
pip install -r requirements.txt
```

### Frontend
```bash
# Supabase packages already installed!
cd frontend
npm install
```

## What Was Implemented

### Backend (twilight_api.py)
- ✅ Supabase client initialization
- ✅ JWT token verification (`verify_token` function)
- ✅ Usage tracking and tier limits (free: 20/day, paid: 500/day)
- ✅ `/me` endpoint for user info and usage stats
- ✅ `/chat` endpoint protected with authentication
- ✅ Dev mode fallback when Supabase not configured

### Frontend Files Created
1. **`frontend/lib/supabase.ts`** - Supabase client configuration
2. **`frontend/contexts/AuthContext.tsx`** - Authentication state management
3. **`frontend/components/AuthModal.tsx`** - Login/signup modal with email/password and social login
4. **`frontend/components/UserProfile.tsx`** - User profile display with usage stats
5. **`frontend/app/auth/callback/page.tsx`** - OAuth callback handler for social login
6. **Updated `frontend/app/layout.tsx`** - Wrapped app with AuthProvider
7. **Updated `frontend/app/page.tsx`** - Integrated auth, sends JWT tokens, shows auth UI

### Features Implemented
- ✅ Email/password authentication
- ✅ Social login support (Google, GitHub) - requires provider configuration
- ✅ Automatic user profile creation on signup
- ✅ Session persistence (localStorage)
- ✅ JWT token sending with all API requests
- ✅ Usage tracking and display
- ✅ Daily message limits enforcement
- ✅ Sign out functionality
- ✅ Auth modal for login/signup
- ✅ User profile display with tier badge
- ✅ Usage progress bar

## Enable Authentication Providers (Optional)

For social login (Google, GitHub, etc.):

1. Go to Supabase Dashboard → Authentication → Providers
2. Enable desired providers (Google, GitHub, etc.)
3. Configure OAuth credentials
4. Add redirect URLs:
   - Development: `http://localhost:3000/auth/callback`
   - Production: `https://yourdomain.com/auth/callback`

## Testing the Setup

1. Start the backend:
   ```bash
   python twilight_api.py
   ```

2. Start the frontend:
   ```bash
   cd frontend
   npm run dev
   ```

3. Test the authentication flow:
   - Navigate to `http://localhost:3000`
   - Sign up with email/password
   - Verify user profile is created in Supabase
   - Try chatting (check usage limits work)
   - Log out and log back in
   - Verify session persistence

## Troubleshooting

### "Auth disabled (dev mode)" warning
- Backend can't find SUPABASE_URL or SUPABASE_SERVICE_KEY
- Check your `.env` file in project root

### "Invalid token" errors
- Check frontend has correct NEXT_PUBLIC_SUPABASE_ANON_KEY
- Verify user is logged in (check browser console)

### "User profile not found" error
- Profile creation might have failed during signup
- Manually create profile in Supabase dashboard

### Social login not working
- Verify OAuth provider is enabled in Supabase
- Check redirect URLs are configured correctly
- Ensure client ID/secret are set

## Security Notes

- **NEVER** commit `.env` or `.env.local` files to version control
- Use `service_role` key only in backend (has admin privileges)
- Use `anon` key in frontend (has limited privileges)
- RLS policies protect user data in Supabase

