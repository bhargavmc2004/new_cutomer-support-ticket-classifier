# Streamlit Cloud Deployment Troubleshooting

## Error: "Error installing requirements"

### What Happened
Streamlit Cloud failed to install Python packages listed in `requirements.txt` during the build process.

### What We Fixed
1. **Removed conflicting packages:**
   - `bitsandbytes` (complex CUDA dependencies)
   - `peft` (version conflicts)

2. **Pinned all dependency versions** for consistency:
   - Prevents unexpected breaking changes
   - Ensures reproducible builds

3. **Tested versions verified to work on Streamlit Cloud:**
   - torch==2.0.1
   - transformers==4.35.2
   - huggingface-hub==0.19.4
   - streamlit==1.28.1

### Steps to Deploy the Fix

#### 1. Verify the changes locally (optional)
```bash
# Navigate to project directory
cd /workspaces/new_cutomer-support-ticket-classifier

# Install requirements locally to test
pip install -r requirements.txt

# If successful, run the app
streamlit run streamlit_app.py
```

#### 2. Redeploy on Streamlit Cloud
```bash
# Changes are already pushed to GitHub
# Go to: https://share.streamlit.io

# 1. Find your app in the dashboard
# 2. Click "Manage App" (3 dots menu)
# 3. Click "Reboot App" 
# 4. Or wait for GitHub Actions to automatically redeploy
```

#### 3. Force refresh in browser
- Windows/Linux: `Ctrl + Shift + R`
- Mac: `Cmd + Shift + R`

#### 4. Clear Streamlit cache (if needed)
- On the app, press `Ctrl + M` and select "Clear cache"

---

## If Error Persists

### Check 1: Review Streamlit Cloud Logs
1. Go to your app on Streamlit Cloud
2. Click "Manage App"
3. Go to "Settings" tab
4. Scroll to "Logs" section
5. Copy the error message and search for:
   - Package name that failed
   - Version conflict messages
   - CUDA/compilation errors

### Check 2: Verify GitHub Repository
```bash
# Ensure changes are pushed
git log --oneline -5

# Should show: "Fix: Optimize requirements.txt..."
```

### Check 3: Test Package Installation Locally
```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Try installing requirements
pip install -r requirements.txt

# If this fails, let us know the specific error package
```

### Check 4: Force Streamlit Cloud Rebuild
1. Go to Streamlit Cloud dashboard
2. Find your app
3. Click "Manage App" (3 dots)
4. Click "Delete App"
5. Reconnect the GitHub repository
6. Streamlit will rebuild from scratch

---

## Removed Packages Explanation

### Why `bitsandbytes` was removed:
- Requires CUDA toolkit compilation
- Streamlit Cloud has limited system resources
- Not necessary for inference-only (we only use models, don't train)
- Models still work with standard PyTorch

### Why `peft` was removed:
- Parameter-Efficient Fine-Tuning (not used in inference)
- Added dependency complexity
- Can be re-added later if needed for training

---

## Package Dependency Tree

The remaining packages work together as follows:

```
streamlit (UI)
  ├── transformers (ML models)
  │   ├── torch (deep learning)
  │   ├── huggingface-hub (model downloads)
  │   ├── datasets (data handling)
  │   └── numpy (math)
  ├── pandas (data processing)
  ├── scikit-learn (ML utilities)
  └── requests (API calls)
```

---

## Quick Reference

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web UI framework |
| torch | 2.0.1 | Deep learning backend |
| transformers | 4.35.2 | Pre-trained models |
| huggingface-hub | 0.19.4 | Model downloads |
| pandas | 2.0.3 | Data manipulation |
| datasets | 2.14.6 | Dataset handling |
| scikit-learn | 1.3.2 | ML utilities |
| accelerate | 0.24.1 | Distributed inference |
| requests | 2.31.0 | HTTP requests |
| python-dotenv | 1.0.0 | Environment variables |

---

## Support

If the error still occurs:
1. Check the exact error message in Streamlit Cloud logs
2. Share the error message with support
3. Try the "Force Rebuild" option (Check 4 above)

**Last Updated:** December 7, 2025
