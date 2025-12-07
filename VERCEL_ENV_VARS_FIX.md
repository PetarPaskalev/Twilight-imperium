# Fix Vercel Environment Variables - DNS Error Fix

## 🔴 Problem

You're seeing: `placeholder.supabase.co's DNS address could not be found`

**Cause:** The frontend deployed on Vercel doesn't have the Supabase environment variables set, so it's using the placeholder fallback values.

---

## ✅ Solution: Add Environment Variables in Vercel

### Step 1: Go to Vercel Dashboard

1. Visit: https://vercel.com/dashboard
2. Find your **Twilight Imperium** project
3. Click on it to open

### Step 2: Add Environment Variables

1. Click **Settings** (in the top navigation)
2. Click **Environment Variables** (in the left sidebar)
3. Add these **3 variables**:

#### Variable 1: `NEXT_PUBLIC_SUPABASE_URL`
- **Key:** `NEXT_PUBLIC_SUPABASE_URL`
- **Value:** `https://vauomdmvwtjywdjvilsw.supabase.co`
- **Environment:** Select all (Production, Preview, Development)
- Click **Save**

#### Variable 2: `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- **Key:** `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- **Value:** `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA`
- **Environment:** Select all (Production, Preview, Development)
- Click **Save**

#### Variable 3: `NEXT_PUBLIC_API_URL`
- **Key:** `NEXT_PUBLIC_API_URL`
- **Value:** `https://twilight-imperium.onrender.com`
- **Environment:** Select all (Production, Preview, Development)
- Click **Save**

### Step 3: Redeploy

**Important:** After adding environment variables, you MUST redeploy:

1. Go to **Deployments** tab
2. Find the latest deployment
3. Click the **three dots** (⋯) menu
4. Click **Redeploy**
5. Wait for deployment to complete (~2-3 minutes)

---

## 🔍 Verify It's Working

After redeployment:

1. Visit your Vercel URL (e.g., `https://twilight-imperium-rose.vercel.app`)
2. Open browser console (F12)
3. Check for:
   - ✅ No "placeholder.supabase.co" errors
   - ✅ Should see Supabase connection working
4. Try clicking "Continue with Google"
   - Should redirect to Google sign-in (not placeholder error)

---

## 📋 Quick Checklist

- [ ] Added `NEXT_PUBLIC_SUPABASE_URL` in Vercel
- [ ] Added `NEXT_PUBLIC_SUPABASE_ANON_KEY` in Vercel
- [ ] Added `NEXT_PUBLIC_API_URL` in Vercel
- [ ] Set all variables for Production, Preview, and Development
- [ ] Redeployed the frontend
- [ ] Tested Google sign-in

---

## 🆘 Still Not Working?

### Check 1: Variable Names
- Must start with `NEXT_PUBLIC_` (required for Next.js)
- No typos or extra spaces

### Check 2: Redeployment
- Environment variables only apply after redeployment
- Check deployment logs for any errors

### Check 3: Browser Cache
- Clear browser cache or use incognito mode
- Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

### Check 4: Vercel Build Logs
- Go to Deployments → Latest → View Build Logs
- Check for any build errors

---

## 💡 Why This Happened

Next.js requires environment variables to be prefixed with `NEXT_PUBLIC_` to be available in the browser. When these aren't set in Vercel, the code falls back to placeholder values:

```typescript
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://placeholder.supabase.co';
```

Once you add the variables in Vercel and redeploy, it will use the real Supabase URL and Google OAuth will work!

