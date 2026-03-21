# Vercel Deployment Guide

## Quick Deploy Steps:

### 1. Install Vercel CLI
```bash
npm install -g vercel
```

### 2. Initialize Git Repository
```bash
cd "SOURCE CODE/Knee Osteoarthritis"
git init
git add .
git commit -m "Initial commit - Knee Osteoarthritis Prediction App"
```

### 3. Push to GitHub
- Create a new GitHub repository
- Push your code:
```bash
git remote add origin https://github.com/YOUR-USERNAME/knee-osteoarthritis.git
git branch -M main
git push -u origin main
```

### 4. Deploy to Vercel
Option A: Using Vercel Dashboard
- Go to https://vercel.com
- Sign in with GitHub
- Click "New Project"
- Select the knee-osteoarthritis repository
- Click Deploy

Option B: Using Vercel CLI
```bash
vercel --prod
```

## Important Notes:

1. **Model File**: The `knee.h5` file will be deployed with your app. Ensure it's in the project root.

2. **Static Files**: All files in `static/` and `templates/` will be deployed.

3. **Upload Limits**: Vercel has file size limits (typically 50MB max for serverless functions). If your knee.h5 is > 50MB, you may need to:
   - Upload it separately to a CDN or cloud storage
   - Reference it dynamically in your app

4. **Memory Limits**: TensorFlow models can be memory-intensive. Monitor your Vercel deployment logs.

5. **Environment Variables**: If you add API keys later, set them in Vercel dashboard:
   - Project Settings → Environment Variables

## Monitoring:

After deployment, visit:
- **App URL**: https://your-project.vercel.app
- **Logs**: https://vercel.com/dashboard → Your Project → Deployments → Logs

## Troubleshooting:

- If deployment fails, check logs for TensorFlow/import errors
- Ensure all imports in app.py are compatible with serverless environment
- Test locally: `python app.py` before deploying

## Rollback:

If needed, revert to previous deployment:
- Go to Vercel dashboard
- Select deployment
- Click "Promote to Production"

---

Need help? Visit https://vercel.com/docs
