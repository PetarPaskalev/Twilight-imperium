# Google OAuth Setup Guide

## ✅ Project Status
Your Supabase project has been **restored** and is now **ACTIVE**.

**Project URL:** `https://vauomdmvwtjywdjvilsw.supabase.co`

## Required Steps to Fix Google Sign-In

### Step 1: Verify Environment Variables

Make sure you have `frontend/.env.local` file with:

```bash
NEXT_PUBLIC_SUPABASE_URL=https://vauomdmvwtjywdjvilsw.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Important:** After creating/updating `.env.local`, **restart your Next.js dev server**:
```bash
# Stop the server (Ctrl+C) and restart:
cd frontend
npm run dev
```

### Step 2: Configure Google OAuth in Supabase Dashboard

1. **Go to Supabase Dashboard:**
   - Visit: https://supabase.com/dashboard/project/vauomdmvwtjywdjvilsw
   - Navigate to: **Authentication** → **Providers**

2. **Enable Google Provider:**
   - Find "Google" in the list
   - Toggle it **ON**

3. **Get Google OAuth Credentials:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project (or select existing)
   - Go to **APIs & Services** → **Credentials**
   - Click **Create Credentials** → **OAuth Client ID**
   - Application type: **Web application**
   - **Authorized JavaScript origins:**
     - `http://localhost:3000` (for local dev)
     - `https://vauomdmvwtjywdjvilsw.supabase.co` (Supabase domain)
   - **Authorized redirect URIs:**
     - `http://localhost:3000/auth/callback` (for local dev)
     - `https://vauomdmvwtjywdjvilsw.supabase.co/auth/v1/callback` (Supabase callback)

4. **Add Credentials to Supabase:**
   - Copy the **Client ID** and **Client Secret** from Google
   - Paste them into Supabase Dashboard → Authentication → Providers → Google
   - Click **Save**

### Step 3: Configure Redirect URLs in Supabase

1. Go to **Authentication** → **URL Configuration**
2. Add to **Redirect URLs**:
   - `http://localhost:3000/auth/callback` (for local development)
   - `https://your-production-domain.com/auth/callback` (for production)

### Step 4: Test Google Sign-In

1. **Restart your frontend server** (to load new env vars):
   ```bash
   cd frontend
   npm run dev
   ```

2. **Open your app:** `http://localhost:3000`

3. **Click "Continue with Google"**

4. **Expected flow:**
   - Redirects to Google sign-in
   - After signing in, redirects back to `/auth/callback`
   - Then redirects to home page
   - User profile is automatically created

## Troubleshooting

### "ERR_NAME_NOT_RESOLVED" Error
- ✅ **FIXED:** Project was inactive, now restored
- If you still see this, wait 1-2 minutes for DNS to propagate

### "Redirect URL mismatch"
- Check that `http://localhost:3000/auth/callback` is in:
  - Google Cloud Console → OAuth Client → Authorized redirect URIs
  - Supabase Dashboard → Authentication → URL Configuration

### "Provider not enabled"
- Make sure Google provider is toggled ON in Supabase Dashboard
- Verify Client ID and Secret are saved

### "Invalid client" error
- Double-check Client ID and Secret in Supabase Dashboard
- Ensure they match the ones from Google Cloud Console

### Environment variables not loading
- Make sure file is named `.env.local` (not `.env`)
- Restart Next.js dev server after creating/updating the file
- Check file is in `frontend/` directory (not root)

## Current Configuration

- **Project ID:** `vauomdmvwtjywdjvilsw`
- **Project Status:** ✅ ACTIVE_HEALTHY
- **Anon Key:** Already configured in code
- **Callback URL:** `http://localhost:3000/auth/callback`
- **OAuth Flow:** PKCE (secure, recommended)

## Next Steps After Setup

Once Google OAuth is working:
1. Users can sign in with their Google account
2. Profile is automatically created on first sign-in
3. User gets "free" tier by default (20 messages/day)
4. Usage is tracked in `user_usage` table

