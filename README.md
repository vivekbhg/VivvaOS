# VivvaAPI - Passive Income Developer API

A complete, deployable **paid API service** that generates recurring revenue. Developers sign up, get API keys, and pay monthly for access to useful utility endpoints.

## Revenue Model

| Plan | Price | Daily Requests |
|------|-------|----------------|
| Free | $0/mo | 100 |
| Pro | $9/mo | 10,000 |
| Business | $29/mo | 100,000 |

## API Endpoints

### Text Processing
- `POST /api/v1/text/analyze` - Word count, reading time, top words
- `POST /api/v1/text/markdown-to-html` - Markdown to sanitized HTML
- `POST /api/v1/text/extract` - Extract emails, URLs, phones, hashtags
- `POST /api/v1/text/hash` - MD5, SHA1, SHA256, SHA512
- `POST /api/v1/text/slugify` - URL-friendly slugs

### Data Tools
- `POST /api/v1/data/csv-to-json` - CSV to JSON conversion
- `POST /api/v1/data/json-to-csv` - JSON to CSV conversion
- `POST /api/v1/data/base64` - Base64 encode/decode
- `POST /api/v1/data/json-diff` - Compare two JSON objects
- `POST /api/v1/data/flatten-json` - Flatten nested JSON

### Generators
- `POST /api/v1/generate/qr-code` - QR codes as base64 PNG
- `POST /api/v1/generate/password` - Secure random passwords
- `POST /api/v1/generate/uuid` - UUID v4 generation

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Stripe keys

# 3. Run
uvicorn app.main:app --reload

# 4. Register and get API key
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "secure123"}'

# 5. Use the API
curl -X POST http://localhost:8000/api/v1/text/analyze \
  -H "X-API-Key: vv_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}'
```

## Deploy

### Render (Recommended - Free tier available)
1. Push to GitHub
2. Connect repo on [render.com](https://render.com)
3. `render.yaml` auto-configures everything
4. Add Stripe env vars in Render dashboard

### Fly.io
```bash
fly launch
fly secrets set STRIPE_SECRET_KEY=sk_... STRIPE_WEBHOOK_SECRET=whsec_...
fly deploy
```

### Docker
```bash
docker compose up -d
```

### Railway / Heroku
Uses the included `Procfile` automatically.

## Stripe Setup

1. Create a [Stripe account](https://stripe.com)
2. Create two Products with monthly recurring prices:
   - **Pro** - $9/month
   - **Business** - $29/month
3. Copy the Price IDs to your `.env`
4. Set up a webhook endpoint pointing to `https://yourdomain.com/billing/webhook`
   - Events: `checkout.session.completed`, `customer.subscription.deleted`

## How It Makes Money

1. **Freemium funnel**: Free tier (100 req/day) gets developers hooked
2. **Usage-based upgrade pressure**: Hit the limit? Upgrade prompt in the 429 error
3. **Sticky by design**: Once integrated, switching costs are high
4. **Zero marginal cost**: Text/data processing costs nothing to serve
5. **Auto-billing**: Stripe handles recurring payments automatically

## Tech Stack

- **FastAPI** - Async Python web framework
- **SQLite** - Zero-config database (swap to Postgres for scale)
- **Stripe** - Payment processing
- **Jinja2** - Landing page templates
