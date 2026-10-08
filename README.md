# Wax Fashion StyleGAN — public web app (Azure + Vercel)

A version of the [Hugging Face Space](https://huggingface.co/spaces/paceunivai/WaxFashionStyleGAN)
that anyone can use: **no Hugging Face account and no 5‑minute GPU quota.**

```
Visitor's browser ──► Vercel (static page: frontend/)
        │
        └── GET /image?seed=…&psi=… ──► Azure Container Apps (backend/, CPU)
                                              └── downloads the model from HF once at startup (HF_TOKEN secret)
```

**Why CPU?** StyleGAN2 is small. With fp16 turned off, one 1024×1024 pattern
took **~3.6 s on 2 CPU threads** in a local test. Azure for Students
subscriptions usually can't get GPU quota, and CPU containers can scale to
zero, so it costs nothing while idle.

| Folder | What it is |
|---|---|
| `backend/` | FastAPI server (`app.py`) + `Dockerfile`. Endpoints: `GET /health`, `GET /image?seed=&psi=&format=jpg\|png` |
| `frontend/` | Static site for Vercel (same design as the Space). Backend URL lives in `frontend/config.js` |
| `.github/workflows/backend-image.yml` | Builds the backend image and pushes it to GitHub Container Registry (GHCR) |
| `deploy_azure.sh` | Creates or updates the Azure Container App |
| `hf_space/` | Copy of the current Space code, for reference |

---

## Deploy for a test today (about an hour)

You need: the Azure CLI (`brew install azure-cli`), a GitHub account, a Vercel
account (sign in with GitHub), and a Hugging Face **read** token that can
access the gated `paceailab/WaxFashionStyleGAN` model. The Space's `HF_TOKEN`
secret is one example.

### 1. Put this folder on GitHub and build the image

1. Create a new repo, e.g. `researchpace/waxfashion-web`, and push this folder to it.
2. The **backend-image** action runs automatically (or start it from the Actions tab).
   It takes about 5 minutes and publishes `ghcr.io/<owner>/waxfashion-api:latest`.
3. On GitHub, go to **Packages → waxfashion-api → Package settings → Change visibility → Public**.
   The image holds no secrets, and making it public lets Azure pull it without registry credentials.

> You don't have Docker installed locally, and Azure's own builder (`az acr build`)
> is often blocked on student/free-credit subscriptions. That's why GitHub
> builds the image.

### 2. Deploy the backend to Azure

```bash
az login
```
```bash
HF_TOKEN=hf_xxx IMAGE=ghcr.io/<owner>/waxfashion-api:latest ./deploy_azure.sh
```

It prints the backend URL. Open `https://<url>/health` and wait until it shows
`"ready": true`. The first start downloads the 369 MB model, which takes about 30–60 s.
Then open `https://<url>/image?seed=137&psi=0.7` to see a real pattern.

If it fails with `RequestDisallowedByAzure`, your student subscription only
allows certain regions. Rerun with e.g. `LOCATION=westeurope` or `LOCATION=swedencentral`.

### 3. Deploy the frontend to Vercel

1. Put the backend URL in `frontend/config.js` and push.
2. In Vercel: **Add New → Project →** import the repo → set **Root Directory = `frontend`**,
   Framework preset **Other**, no build command → **Deploy**.
3. Lock the backend to your site (optional but recommended):
   ```bash
   ALLOWED_ORIGINS=https://<your-site>.vercel.app HF_TOKEN=hf_xxx IMAGE=ghcr.io/<owner>/waxfashion-api:latest ./deploy_azure.sh
   ```

---

## Things to know

- **Cold starts:** with `MIN_REPLICAS=0`, the first visitor after a quiet
  period waits ~30–60 s while the container starts and loads the model. The
  page pings `/health` on load to start waking the server early, and shows a
  "waking up" message. For the announcement day, set `MIN_REPLICAS=1` to keep
  one replica warm, which costs more. Set it back to 0 afterwards.
- **Cost:** Container Apps has a monthly free allowance (180,000 vCPU‑seconds +
  360,000 GiB‑seconds). At 2 vCPU / 4 GiB that is about 25 hours of busy time,
  or very roughly 10k+ patterns. Usage beyond that comes out of the $100 Azure
  for Students credit. Check current prices in the Azure pricing calculator,
  and add a budget alert under Cost Management.
- **Limits:** each visitor IP can make 40 *new* patterns per 10 minutes
  (`RATE_LIMIT`, `RATE_WINDOW_S`). Re-downloading tiles you already see doesn't count.
  Up to 3 replicas run at once (`MAX_REPLICAS`).
- **Same seed = same image:** the backend uses `noise_mode="const"`, so results
  are fully repeatable. The Space uses random per-run noise, so its images can
  differ in fine detail for the same seed.

## Run locally

```bash
cd backend && pip install torch && pip install -r requirements.txt
```
```bash
HF_TOKEN=hf_xxx uvicorn app:app --port 8000
```
Needs Python 3.10–3.11, because the StyleGAN2 code imports `distutils`. Then set
`apiUrl: "http://localhost:8000"` in `frontend/config.js` and run
`python3 -m http.server 3000 -d frontend`.
