# Fix Supabase OAuth Redirect URL Configuration

## Problem
OAuth callback is redirecting to `localhost:3000` instead of your Vercel production URL.

## Solution: Add Redirect URLs in Supabase

### Step 1: Go to Supabase URL Configuration

1. Go to: https://supabase.com/dashboard/project/vauomdmvwtjywdjvilsw/auth/url-configuration
2. Or navigate: **Authentication** → **URL Configuration** (in left sidebar)

### Step 2: Add Your Redirect URLs

In the **Redirect URLs** section, you need to add:

**For Production (Vercel):**
```
https://your-vercel-app.vercel.app/auth/callback
```

**For Local Development:**
```
http://localhost:3000/auth/callback
```

**Important:** Add BOTH URLs if you want to test locally AND in production!

### Step 3: Set Site URL

Make sure your **Site URL** is set to your production URL:
```
https://your-vercel-app.vercel.app
```

### Step 4: Save and Test

1. Click **Save**
2. Try signing in with Google again
3. It should redirect to your Vercel URL (not localhost)

## How It Works

- When you click "Sign in with Google", the code uses `window.location.origin` to determine the redirect URL
- If you're on `localhost:3000`, it redirects to `localhost:3000/auth/callback`
- If you're on your Vercel URL, it redirects to `your-vercel-app.vercel.app/auth/callback`
- Supabase only allows redirects to URLs in the allowlist

## Quick Checklist

- [ ] Added Vercel URL to Supabase redirect URLs: `https://your-app.vercel.app/auth/callback`
- [ ] Added localhost for dev: `http://localhost:3000/auth/callback` (optional)
- [ ] Set Site URL to production URL
- [ ] Saved changes in Supabase
- [ ] Tested sign-in from Vercel (not localhost)

## Find Your Vercel URL

1. Go to: https://vercel.com/dashboard
2. Open your project
3. Copy the deployment URL (e.g., `https://twilight-imperium-rose.vercel.app`)
4. Add `/auth/callback` to the end


