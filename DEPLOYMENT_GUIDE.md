# Deployment Guide - Vercel (Frontend) & Render (Backend)

## Important: Environment Variables

**NEVER commit `.env` or `.env.local` files to Git!** They're already in `.gitignore`.

Instead, you'll set environment variables directly in Vercel and Render dashboards.

---

## 🎨 Frontend Deployment (Vercel)

### Step 1: Push to GitHub

1. Make sure all changes are committed
2. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Add authentication system"
   git push origin main
   ```

### Step 2: Deploy to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. **Important:** Set the Root Directory to `frontend`
5. Click "Deploy"

### Step 3: Add Environment Variables in Vercel

After deployment, add these environment variables:

1. Go to your project in Vercel
2. Settings → Environment Variables
3. Add these **three** variables:

```
NEXT_PUBLIC_SUPABASE_URL
Value: https://vauomdmvwtjywdjvilsw.supabase.co

NEXT_PUBLIC_SUPABASE_ANON_KEY
Value: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA

NEXT_PUBLIC_API_URL
Value: https://your-backend.onrender.com
(Replace with your actual Render backend URL)
```

4. **Important:** Set these for all environments (Production, Preview, Development)
5. Click "Save"
6. **Redeploy** the project for changes to take effect

### Vercel Configuration

Create `vercel.json` in the `frontend` directory:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "devCommand": "npm run dev",
  "installCommand": "npm install"
}
```

---

## 🔧 Backend Deployment (Render)

### Step 1: Prepare for Render

Make sure your `render.yaml` is configured (check if it exists in your root directory).

### Step 2: Deploy to Render

1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name:** `twilight-imperium-api`
   - **Root Directory:** Leave empty (uses project root)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn twilight_api:app --host 0.0.0.0 --port $PORT`

### Step 3: Add Environment Variables in Render

In the "Environment Variables" section, add:

```
SUPABASE_URL
Value: https://vauomdmvwtjywdjvilsw.supabase.co

SUPABASE_SERVICE_KEY
Value: your_service_role_key_from_supabase
(Get from Supabase Dashboard → Settings → API → service_role key)

SUPABASE_ANON_KEY
Value: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhdW9tZG12d3RqeXdkanZpbHN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1NjEwMTcsImV4cCI6MjA3NjEzNzAxN30.DWRe6PSym78zDk9EGfZgeDz4dJCOjEhjm_5aBepkFxA

OPENAI_API_KEY
Value: your_openai_api_key

ALLOWED_ORIGINS
Value: https://your-frontend.vercel.app
(Replace with your actual Vercel URL)
```

Optional (if using Redis):
```
REDIS_URL
Value: your_redis_connection_string
```

### Step 4: Deploy

Click "Create Web Service" - Render will automatically deploy!

---

## 🔄 Update Frontend with Backend URL

After your Render backend is deployed:

1. Copy the backend URL (e.g., `https://twilight-imperium-api.onrender.com`)
2. Go to Vercel → Your Project → Settings → Environment Variables
3. Update `NEXT_PUBLIC_API_URL` to your Render backend URL
4. Redeploy the frontend

---

## 🔐 CORS Configuration

Make sure your backend allows requests from your frontend domain:

In `.env` on Render, set:
```
ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://your-frontend-preview.vercel.app
```

This is already handled in `twilight_api.py`!

---

## 📋 Deployment Checklist

### Before Deploying:

- [ ] `.env` and `.env.local` are in `.gitignore`
- [ ] All code is committed to GitHub
- [ ] You have your Supabase service_role key ready
- [ ] You have your OpenAI API key ready

### Frontend (Vercel):

- [ ] Repository imported to Vercel
- [ ] Root directory set to `frontend`
- [ ] Environment variables added:
  - [ ] `NEXT_PUBLIC_SUPABASE_URL`
  - [ ] `NEXT_PUBLIC_SUPABASE_ANON_KEY`
  - [ ] `NEXT_PUBLIC_API_URL`
- [ ] Deployment successful

### Backend (Render):

- [ ] Web Service created on Render
- [ ] Environment variables added:
  - [ ] `SUPABASE_URL`
  - [ ] `SUPABASE_SERVICE_KEY`
  - [ ] `SUPABASE_ANON_KEY`
  - [ ] `OPENAI_API_KEY`
  - [ ] `ALLOWED_ORIGINS`
- [ ] Deployment successful
- [ ] Backend URL copied

### Final Steps:

- [ ] Update Vercel's `NEXT_PUBLIC_API_URL` with Render backend URL
- [ ] Redeploy Vercel frontend
- [ ] Test authentication on production
- [ ] Test chatbot on production

---

## 🔍 Troubleshooting

### Frontend Issues:

**"supabaseUrl is required" in production:**
- Check environment variables are set in Vercel
- Make sure they have `NEXT_PUBLIC_` prefix
- Redeploy after adding variables

**"Failed to fetch" when chatting:**
- Check `NEXT_PUBLIC_API_URL` points to correct Render URL
- Check backend is running on Render
- Check CORS settings in backend

### Backend Issues:

**"Auth disabled (dev mode)" in Render logs:**
- Check `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set
- Check for typos in variable names

**CORS errors:**
- Update `ALLOWED_ORIGINS` in Render to include your Vercel URL
- Include both production and preview URLs

**Database errors:**
- Verify Supabase URL is correct
- Check service_role key is correct (not anon key)

---

## 💡 Pro Tips

1. **Free Tier Limitations:**
   - Render free tier: Service sleeps after 15 min of inactivity (first request takes ~30s)
   - Vercel free tier: Generous limits, should be fine for most use

2. **Custom Domains:**
   - Add custom domain in Vercel for frontend
   - Add custom domain in Render for backend
   - Update environment variables accordingly

3. **Preview Deployments:**
   - Vercel automatically creates preview deployments for each branch
   - Add preview URLs to ALLOWED_ORIGINS: `https://*.vercel.app`

4. **Monitoring:**
   - Check Vercel Analytics for frontend performance
   - Check Render logs for backend errors
   - Monitor Supabase usage in dashboard

---

## 📱 Testing Your Deployment

Once everything is deployed:

1. Visit your Vercel URL
2. Click "Sign In / Sign Up"
3. Create a new account
4. Send a test message
5. Check Supabase dashboard to verify:
   - User was created
   - Profile was created
   - Usage was tracked

If everything works - you're live! 🎉

---

## 🆘 Need Help?

- **Vercel Docs:** https://vercel.com/docs
- **Render Docs:** https://render.com/docs
- **Supabase Docs:** https://supabase.com/docs

Check logs:
- Vercel: Deployments tab → Click deployment → View logs
- Render: Dashboard → Your service → Logs tab

