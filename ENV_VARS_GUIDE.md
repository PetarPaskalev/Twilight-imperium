# Environment Variables Guide - Complete Reference

## 🔑 Key Usage Summary

### **IMPORTANT: Same Key, Different Names**

The **anon key** is the SAME value, but used with different variable names:
- **Frontend:** `NEXT_PUBLIC_SUPABASE_ANON_KEY` 
- **Backend:** `SUPABASE_ANON_KEY`
- **Value:** `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA`

The **URL** is also the SAME value:
- **Frontend:** `NEXT_PUBLIC_SUPABASE_URL`
- **Backend:** `SUPABASE_URL`
- **Value:** `https://vauomdmvwtjywdjvilsw.supabase.co`

---

## 📋 Complete Environment Variables List

### **Backend (Render or Local `.env`)**

```bash
# Supabase Configuration
SUPABASE_URL=https://vauomdmvwtjywdjvilsw.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA

# OpenAI (for chatbot)
OPENAI_API_KEY=your_openai_key_here

# CORS (optional, for production)
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

**What each backend key does:**
- `SUPABASE_URL` - Used by both service role and anon clients
- `SUPABASE_SERVICE_KEY` - For database operations (read/write user_profiles, user_usage)
- `SUPABASE_ANON_KEY` - For JWT token verification (verifying tokens from frontend)
- `OPENAI_API_KEY` - For the chatbot LLM

### **Frontend (Vercel or `frontend/.env.local`)**

```bash
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=https://vauomdmvwtjywdjvilsw.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA

# Backend API URL
NEXT_PUBLIC_API_URL=https://twilight-imperium.onrender.com
# For local dev: http://localhost:8000
```

**What each frontend key does:**
- `NEXT_PUBLIC_SUPABASE_URL` - Supabase project URL (for auth client)
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Anon key (for auth client - same value as backend SUPABASE_ANON_KEY)
- `NEXT_PUBLIC_API_URL` - Your backend API URL (Render or localhost)

---

## 🔍 Why You Need Both Keys in Backend

### Service Role Key (`SUPABASE_SERVICE_KEY`)
- **Purpose:** Database operations with full permissions
- **Used for:**
  - Reading/writing `user_profiles` table
  - Reading/writing `user_usage` table
  - Bypassing Row Level Security (RLS)
- **Security:** ⚠️ **NEVER expose this in frontend!** Only backend!

### Anon Key (`SUPABASE_ANON_KEY`)
- **Purpose:** JWT token verification
- **Used for:**
  - Verifying JWT tokens sent from frontend
  - `supabase.auth.get_user(token)` call
- **Security:** ✅ Safe to use in frontend (that's why it's "anon")

---

## 🐛 Common Issues & Fixes

### Issue: 500 Internal Server Error when logged in

**Possible causes:**
1. ❌ `SUPABASE_ANON_KEY` not set in backend
2. ❌ `SUPABASE_SERVICE_KEY` not set in backend
3. ❌ User profile doesn't exist in database
4. ❌ Token verification failing

**Fix:**
1. Check backend logs for the exact error
2. Verify all 3 Supabase keys are set in backend:
   - `SUPABASE_URL` ✅
   - `SUPABASE_SERVICE_KEY` ✅
   - `SUPABASE_ANON_KEY` ✅ (this is the NEW one we added!)

### Issue: 401 Unauthorized

**Possible causes:**
1. ❌ `SUPABASE_ANON_KEY` missing in backend
2. ❌ Token not being sent from frontend
3. ❌ Token expired

**Fix:**
1. Make sure `SUPABASE_ANON_KEY` is set in backend
2. Check browser console for token errors
3. Try signing out and back in

### Issue: "Failed to fetch"

**Possible causes:**
1. ❌ Backend not running
2. ❌ CORS issues
3. ❌ Wrong `NEXT_PUBLIC_API_URL` in frontend

**Fix:**
1. Check backend is running on correct port
2. Verify `NEXT_PUBLIC_API_URL` matches your backend URL
3. Check CORS settings in backend

---

## ✅ Quick Checklist

### Backend Environment Variables:
- [ ] `SUPABASE_URL` = `https://vauomdmvwtjywdjvilsw.supabase.co`
- [ ] `SUPABASE_SERVICE_KEY` = (get from Supabase dashboard)
- [ ] `SUPABASE_ANON_KEY` = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (same as frontend anon key)
- [ ] `OPENAI_API_KEY` = (your OpenAI key)

### Frontend Environment Variables:
- [ ] `NEXT_PUBLIC_SUPABASE_URL` = `https://vauomdmvwtjywdjvilsw.supabase.co`
- [ ] `NEXT_PUBLIC_SUPABASE_ANON_KEY` = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (same value as backend SUPABASE_ANON_KEY)
- [ ] `NEXT_PUBLIC_API_URL` = `https://twilight-imperium.onrender.com` (or `http://localhost:8000` for local)

---

## 🔧 How to Get Service Role Key

1. Go to: https://supabase.com/dashboard/project/vauomdmvwtjywdjvilsw/settings/api
2. Scroll to "Project API keys"
3. Find **"service_role"** key (NOT the anon key)
4. Click "Reveal" to show it
5. Copy and add to backend as `SUPABASE_SERVICE_KEY`

**⚠️ WARNING:** Service role key has full database access. Keep it secret!

