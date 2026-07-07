# HomeWorth — AI House Value Predictor

ML-powered house price estimator with auto-generated floor plans and AI architectural preview.

> **ACTION REQUIRED — SECURITY:** The Perplexity API key previously hardcoded in `app.py`
> (`pplx-2I1gnppYMuoksHNQZy06To9ZaAnmDMi7MNxqpY8DsrCAk`) was publicly exposed in the git
> history. **Go to your Perplexity dashboard and revoke that key immediately.**

---

## Local Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY (see below)

# 4. Run the app
python app.py
# Visit http://localhost:5000
```

## Getting an OpenRouter API Key

1. Sign up at https://openrouter.ai
2. Go to **Keys** and create a new key
3. Paste it into `.env` as `OPENROUTER_API_KEY=sk-or-...`

The default model is `openai/gpt-4o-mini`. Override with `OPENROUTER_MODEL` in `.env`.

---

## Render Deployment

| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT` |
| **Runtime** | Python 3.12.3 (from `runtime.txt`) |

### Environment Variables to set in Render dashboard

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter key (required for house image) |
| `OPENROUTER_MODEL` | Model override, default `openai/gpt-4o-mini` |
| `FLASK_DEBUG` | Set to `0` in production |
| `APP_URL` | Your Render URL e.g. `https://homeworth.onrender.com` |
| `APP_NAME` | Sent as `X-Title` header to OpenRouter |

### Health Check

Configure Render health check path as `/ping`. It returns:

```json
{"status": "ok"}
```

---

## Project Structure

```
app.py              Flask application (ML prediction, floor plan, OpenRouter API)
ML_Models/          Trained model + scaler (joblib)
templates/          Jinja2 HTML templates
static/             CSS, favicon (served via WhiteNoise in production)
requirements.txt    Pinned top-level dependencies
Procfile            gunicorn start command for Render
runtime.txt         Python version pin for Render
.env.example        Environment variable template (safe to commit)
.gitignore          Excludes .env, __pycache__, venv, etc.
```
