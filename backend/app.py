"""
WaxFashionStyleGAN API — Pace AI Lab
CPU inference server for the StyleGAN2-ADA African wax pattern model, built to
run on Azure Container Apps behind a static frontend (Vercel). No login needed
by visitors; the Hugging Face token stays on the server.

    GET /health                                  -> {"ready": bool, ...}
    GET /image?seed=137&psi=0.7&format=jpg|png   -> image bytes

Images are deterministic for a (seed, psi) pair, so responses are cacheable and
a small in-memory LRU means the preview (jpg) and download (png) of the same
tile only run the generator once.
"""

import io, os, sys, time, random, logging, threading, subprocess
from collections import OrderedDict, defaultdict, deque

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("waxfashion")

# ── Settings (environment variables) ──────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN")
MODEL_REPO      = os.environ.get("MODEL_REPO", "paceailab/WaxFashionStyleGAN")
MODEL_FILE      = os.environ.get("MODEL_FILE", "selected_models/styleGAN2ada_Africanwax.pkl")
MODEL_PATH      = os.environ.get("MODEL_PATH")  # local .pkl, skips the download
STYLEGAN_DIR    = os.environ.get("STYLEGAN_DIR", "/opt/waxfashion/stylegan2-ada-pytorch")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
# New images (cache misses) each visitor IP may request per window
RATE_LIMIT      = int(os.environ.get("RATE_LIMIT", "40"))
RATE_WINDOW_S   = int(os.environ.get("RATE_WINDOW_S", "600"))
CACHE_SIZE      = int(os.environ.get("CACHE_SIZE", "64"))
TORCH_THREADS   = int(os.environ.get("TORCH_THREADS", "0"))  # 0 = torch default

if TORCH_THREADS > 0:
    torch.set_num_threads(TORCH_THREADS)

# ── StyleGAN2-ADA code ────────────────────────────────────────────────────────
# The Docker image clones github.com/researchpace/waxfashion at build time; for
# local runs, clone it and point STYLEGAN_DIR at its stylegan2-ada-pytorch dir.
if not os.path.isdir(STYLEGAN_DIR):
    repo_dir = os.path.dirname(STYLEGAN_DIR)
    log.info("Cloning waxfashion repo into %s...", repo_dir)
    subprocess.run(["git", "clone", "--depth=1",
                    "https://github.com/researchpace/waxfashion.git", repo_dir], check=True)
sys.path.insert(0, STYLEGAN_DIR)

import dnnlib, legacy  # noqa: E402

# No CUDA here, so use the reference ops directly (same as the Space)
from torch_utils.ops import bias_act, upfirdn2d  # noqa: E402
bias_act._init  = lambda: False
upfirdn2d._init = lambda: False

# ── Model (loaded in the background so the server answers health checks) ──────
G = None
model_error = None
model_ready = threading.Event()

def load_model():
    global G, model_error
    try:
        path = MODEL_PATH
        if not path:
            from huggingface_hub import hf_hub_download
            log.info("Downloading %s from %s...", MODEL_FILE, MODEL_REPO)
            path = hf_hub_download(MODEL_REPO, MODEL_FILE, token=HF_TOKEN)
        log.info("Loading generator...")
        with dnnlib.util.open_url(path) as f:
            net = legacy.load_network_pkl(f)["G_ema"].to("cpu")
        net.eval()
        G = net
        log.info("Ready: %dx%d, z_dim=%d, torch threads=%d",
                 G.img_resolution, G.img_resolution, G.z_dim, torch.get_num_threads())
    except Exception as e:  # surfaced through /health
        model_error = f"{type(e).__name__}: {e}"
        log.exception("Model failed to load")
    finally:
        model_ready.set()

threading.Thread(target=load_model, daemon=True).start()

# ── Inference ─────────────────────────────────────────────────────────────────
gen_lock = threading.Lock()   # one forward pass at a time; scale out with replicas
cache: "OrderedDict[tuple, Image.Image]" = OrderedDict()
cache_lock = threading.Lock()

def synthesize(seed: int, psi: float) -> Image.Image:
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))
    with gen_lock, torch.inference_mode():
        # noise_mode="const": the same seed + psi always gives the same image.
        # force_fp32: the model runs its top layers in fp16, which is very slow on CPU.
        t = G(z, None, truncation_psi=psi, noise_mode="const", force_fp32=True)
    arr = (t.permute(0, 2, 3, 1).mul(127.5).add(128)
           .clamp(0, 255).to(torch.uint8)[0].numpy())
    return Image.fromarray(arr)

def get_image(seed: int, psi: float):
    """Returns (image, was_cached)."""
    key = (seed, psi)
    with cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key], True
    img = synthesize(seed, psi)
    with cache_lock:
        cache[key] = img
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
    return img, False

# ── Simple per-IP rate limit (per replica, in memory) ─────────────────────────
hits = defaultdict(deque)
hits_lock = threading.Lock()

def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")

def check_rate(ip: str):
    now = time.time()
    with hits_lock:
        q = hits[ip]
        while q and q[0] < now - RATE_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            retry = int(q[0] + RATE_WINDOW_S - now) + 1
            raise HTTPException(429, "Too many patterns for now, please try again in a few minutes.",
                                headers={"Retry-After": str(retry)})
        q.append(now)
        if len(hits) > 10_000:  # drop idle IPs
            for k in [k for k, v in hits.items() if not v]:
                del hits[k]

# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="WaxFashionStyleGAN API", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["GET"], allow_headers=["*"],
                   expose_headers=["Retry-After", "X-Seed"])

@app.get("/")
@app.get("/health")
def health():
    return {
        "ready": G is not None,
        "loading": not model_ready.is_set(),
        "error": model_error,
        "resolution": G.img_resolution if G is not None else None,
    }

@app.get("/image")
def image(request: Request,
          seed: int = Query(..., ge=0, le=2**32 - 1),
          psi: float = Query(0.7, ge=0.1, le=1.5),
          format: str = Query("jpg", pattern="^(jpg|png)$")):
    # Wait out a cold start instead of failing the first visitor
    if not model_ready.wait(timeout=180) or G is None:
        raise HTTPException(503, "The generator is still starting, please try again shortly.")

    psi = round(psi, 2)
    key = (seed, psi)
    with cache_lock:
        cached = key in cache
    if not cached:
        check_rate(client_ip(request))

    t0 = time.time()
    img, was_cached = get_image(seed, psi)
    if not was_cached:
        log.info("seed=%d psi=%.2f generated in %.2fs", seed, psi, time.time() - t0)

    buf = io.BytesIO()
    if format == "png":
        img.save(buf, format="PNG")
        media = "image/png"
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        media = "image/jpeg"
    return Response(buf.getvalue(), media_type=media, headers={
        "Cache-Control": "public, max-age=31536000, immutable",
        "X-Seed": str(seed),
    })
