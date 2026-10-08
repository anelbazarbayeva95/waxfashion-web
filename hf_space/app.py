"""
WaxFashionStyleGAN — Pace AI Lab
StyleGAN2-ADA unconditional model.
Controls: Truncation Psi, Samples, Seed.
Layout: compact editorial header, then a controls card beside a canvas that
holds the result and a strip of example patterns. Light and dark themes follow
Gradio's `.dark` class.
"""

import os, sys, io, base64, zipfile, tempfile, subprocess, random, logging
import gradio as gr
import spaces
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("waxfashion")

# ── StyleGAN2-ADA bootstrap ───────────────────────────────────────────────────
REPO_DIR     = "/tmp/waxfashion"
STYLEGAN_DIR = os.path.join(REPO_DIR, "stylegan2-ada-pytorch")

if not os.path.exists(STYLEGAN_DIR):
    log.info("Cloning waxfashion repo...")
    subprocess.run(["git", "clone", "--depth=1",
                    "https://github.com/researchpace/waxfashion.git", REPO_DIR],
                   check=True, capture_output=True)

if STYLEGAN_DIR not in sys.path:
    sys.path.insert(0, STYLEGAN_DIR)

import dnnlib, legacy

# The Space has no CUDA compiler, so StyleGAN's custom CUDA kernels can't be
# built. upfirdn2d retries the build on every call (slow, and floods the logs),
# so skip straight to the reference implementation for both ops.
from torch_utils.ops import bias_act, upfirdn2d
bias_act._init  = lambda: False
upfirdn2d._init = lambda: False

# ── Model ─────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN secret not set.")

log.info("Downloading model...")
MODEL_PKL = hf_hub_download("paceailab/Waxfashion_StyleGAN",
                             "selected_models/styleGAN2ada_Africanwax.pkl",
                             token=HF_TOKEN)
log.info("Loading generator on CPU...")
with dnnlib.util.open_url(MODEL_PKL) as f:
    G = legacy.load_network_pkl(f)["G_ema"].to("cpu")
G.eval()
log.info("Ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_seed(s):
    s = (s or "").strip().lower()
    if s in {"", "random"}: return random.randint(0, 2**32 - 1)
    try:    return int(s) % (2**32)
    except: return random.randint(0, 2**32 - 1)

def random_seed():
    return str(random.randint(100_000_000, 999_999_999))

def to_data_uri(img, quality=92):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ── Inference ─────────────────────────────────────────────────────────────────
def synthesize(seeds, truncation, device):
    images = []
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        with torch.no_grad():
            t = G(z, None, truncation_psi=float(truncation))
        arr = (t.permute(0,2,3,1).mul(127.5).add(128)
               .clamp(0,255).to(torch.uint8)[0].cpu().numpy())
        images.append(Image.fromarray(arr))
    return images

@spaces.GPU(duration=120)
def generate(truncation, seeds):
    G.to("cuda")
    try:
        return synthesize(seeds, truncation, "cuda")
    finally:
        G.to("cpu")
        torch.cuda.empty_cache()

# Example patterns are the images uploaded for this space. They are pulled at
# startup: first from an `examples/` folder next to this file, otherwise from
# the model repo (files whose path mentions "example" or "sample").
MODEL_REPO  = "paceailab/Waxfashion_StyleGAN"
IMAGE_EXTS  = (".png", ".jpg", ".jpeg", ".webp")
MAX_EXAMPLES = 4

def load_examples():
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    if os.path.isdir(local_dir):
        paths = sorted(os.path.join(local_dir, f) for f in os.listdir(local_dir)
                       if f.lower().endswith(IMAGE_EXTS))
    else:
        files = [f for f in list_repo_files(MODEL_REPO, token=HF_TOKEN)
                 if f.lower().endswith(IMAGE_EXTS)
                 and any(k in f.lower() for k in ("example", "sample"))]
        paths = [hf_hub_download(MODEL_REPO, f, token=HF_TOKEN) for f in sorted(files)]
    images = []
    for p in paths[:MAX_EXAMPLES]:
        im = Image.open(p).convert("RGB")
        im.thumbnail((640, 640))
        images.append(im)
    return images

try:
    EXAMPLES = load_examples()
    log.info(f"Loaded {len(EXAMPLES)} example images.")
except Exception as e:
    log.warning(f"Could not load example images: {e}")
    EXAMPLES = []

# ── HTML fragments ────────────────────────────────────────────────────────────
INFO_ICON = ('<svg class="ni" viewBox="0 0 24 24" aria-hidden="true">'
             '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>')

# Small woven kente-style square used as the brand mark
MARK = ('<svg class="mark" viewBox="0 0 32 32" aria-hidden="true">'
        '<rect x="0" y="0" width="16" height="16" class="k-g"/>'
        '<rect x="0" y="3" width="16" height="3" class="k-i"/><rect x="0" y="10" width="16" height="3" class="k-i"/>'
        '<rect x="16" y="0" width="16" height="16" class="k-i"/>'
        '<rect x="19" y="0" width="3" height="16" class="k-g"/><rect x="26" y="0" width="3" height="16" class="k-g"/>'
        '<rect x="0" y="16" width="16" height="16" class="k-c"/>'
        '<rect x="3" y="16" width="3" height="16" class="k-g"/><rect x="10" y="16" width="3" height="16" class="k-g"/>'
        '<rect x="16" y="16" width="16" height="16" class="k-g"/>'
        '<rect x="16" y="19" width="16" height="3" class="k-c"/><rect x="16" y="26" width="16" height="3" class="k-c"/>'
        '</svg>')

TOP = (
    '<div id="band" aria-hidden="true"></div>'
    '<div id="gpu-notice" role="note">' + INFO_ICON +
    '<span>Shared GPU time: <strong>40&nbsp;min</strong> with Hugging Face PRO, '
    '<strong>5&nbsp;min</strong> on a free account. '
    'Keep variations at 1 to make it&nbsp;last.</span></div>'
)

HEADER = (
    '<header id="hero">'
    '<div class="hero-l">'
    f'<div class="brand">{MARK}<span>A space by paceailab</span></div>'
    '<h1>Generate AI-powered <em>African wax-inspired</em> textile patterns</h1>'
    '</div>'
    '<div class="hero-r">'
    '<p>This StyleGAN2 model generates vibrant wax-inspired textile patterns '
    'trained on a synthetic dataset of African Wax Prints.</p>'
    '</div>'
    '</header>'
)

DOWNLOAD_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>')
EXPAND_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/></svg>')

CTL_HEAD = '<div class="card-head"><span class="ctitle">Pattern generator</span></div>'

TRUNC_HINT = ('<p class="hint">Controls variety. Low = safe &amp; typical '
              '&middot; High = creative &amp; unpredictable</p>')
SEED_HINT  = '<p class="hint">Same seed + same settings = same image every time.</p>'

EMPTY = ('<div class="empty">' + MARK +
         '<span>Your generated pattern will appear here</span></div>')

def result_html(files, previews):
    """Result tiles. Each tile has its own download icon, and clicking a tile
    opens a larger view (pure CSS :target lightbox, no JS) with a download
    button for that single tile. Images are linked as files rather than
    embedded, which keeps the server response small."""
    n   = len(files)
    uid = random.randint(0, 10**9)
    tiles, boxes = [], []
    for i, (path, preview) in enumerate(zip(files, previews), 1):
        src, box_id = f"/gradio_api/file={preview}", f"wlb-{uid}-{i}"
        label = f"Pattern {i} of {n}" if n > 1 else "Your pattern"
        href, fname = f"/gradio_api/file={path}", os.path.basename(path)
        tiles.append(
            f'<figure><a class="zoom" href="#{box_id}" aria-label="View {label.lower()} larger">'
            f'<img src="{src}" alt="Generated African wax-inspired pattern"/>'
            f'<span class="zoom-hint" aria-hidden="true">{EXPAND_ICON}</span></a>'
            f'<a class="tile-dl" href="{href}" download="{fname}" aria-label="Download {label.lower()}" '
            f'title="Download">{DOWNLOAD_ICON}</a></figure>')
        boxes.append(
            f'<div class="wlb" id="{box_id}" role="dialog" aria-label="{label}">'
            f'<a class="wlb-bg" href="#_" aria-label="Close"></a>'
            f'<div class="wlb-box"><img src="{src}" alt="Generated African wax-inspired pattern"/>'
            f'<div class="wlb-bar"><span>{label}</span><span class="wlb-actions">'
            f'<a class="wlb-x" href="#_">Close</a>'
            f'<a class="wlb-dl" href="{href}" download="{fname}">↓ Download this tile</a>'
            f'</span></div></div></div>')
    return (f'<div class="gen"><div class="tiles" style="--n:{n}">{"".join(tiles)}</div></div>'
            + "".join(boxes))

def examples_html():
    if not EXAMPLES:
        return ""
    figs = "".join(
        f'<figure><img src="{to_data_uri(im, 88)}" alt="Example African wax-inspired pattern"/></figure>'
        for im in EXAMPLES)
    return (f'<section id="examples"><div class="card-head"><span class="ctitle">Examples</span></div>'
            f'<div class="strip">{figs}</div></section>')

GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
    '5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94'
    '-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87'
    '.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 '
    '2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12'
    '.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 '
    '.21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
)

DATASET_ICON = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1C4.7 1 2 2.1 2 3.5v9C2 13.9 4.7 15 8 15s6'
    '-1.1 6-2.5v-9C14 2.1 11.3 1 8 1zm0 1.5c2.8 0 4.5.9 4.5 1s-1.7 1-4.5 1-4.5-.9-4.5-1 1.7-1 4.5-1zM3.5 '
    '5.6C4.8 6.2 6.3 6.5 8 6.5s3.2-.3 4.5-.9V8c0 .1-1.7 1-4.5 1S3.5 8.1 3.5 8V5.6zm0 4.4c1.3.6 2.8.9 4.5'
    '.9s3.2-.3 4.5-.9v2.5c0 .1-1.7 1-4.5 1s-4.5-.9-4.5-1V10z"/></svg>'
)

def footer_link(href, icon, label):
    return f'<a href="{href}" target="_blank" rel="noopener">{icon}{label}</a>'

# Same footer as the StableDiffusion space, without the 5K dataset link
FOOTER = (
    '<div id="app-footer">'
    '<span>Pace AI Lab · African Wax Patterns</span>'
    '<nav aria-label="Project links">'
    + footer_link("https://github.com/researchpace/waxfashion", GITHUB_ICON, "Code on GitHub")
    + footer_link("https://huggingface.co/datasets/paceailab/AfricanWaxPatterns_2KDataset", DATASET_ICON, "2K Dataset")
    + '</nav>'
    '</div>'
)

# ── Handlers ──────────────────────────────────────────────────────────────────
# Generated PNGs live here; the folder is passed to launch(allowed_paths=...)
# so single tiles can be downloaded from the larger view.
DOWNLOAD_ROOT = os.path.join(tempfile.gettempdir(), "wax_downloads")
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)

def run(truncation, n_samples, seed_str):
    n    = int(n_samples)
    base = parse_seed(seed_str)
    seeds = [(base + i) % (2**32) for i in range(n)]
    images = generate(truncation, seeds)

    # Every tile is saved as a PNG so it can be downloaded on its own; the main
    # button downloads one PNG or a ZIP of all tiles.
    out_dir = tempfile.mkdtemp(prefix="wax_", dir=DOWNLOAD_ROOT)
    # PNGs are for download; lighter JPEG previews are what the page shows.
    files, previews = [], []
    for im, s in zip(images, seeds):
        p = os.path.join(out_dir, f"wax_pattern_seed{s}.png")
        im.save(p)
        files.append(p)
        q = os.path.join(out_dir, f"preview_seed{s}.jpg")
        im.convert("RGB").save(q, quality=90)
        previews.append(q)
    if n == 1:
        path  = files[0]
        label = "↓ Download tile"
    else:
        path = os.path.join(out_dir, f"wax_patterns_seed{seeds[0]}-{seeds[-1]}.zip")
        with zipfile.ZipFile(path, "w") as zf:
            for p in files:
                zf.write(p, arcname=os.path.basename(p))
        label = f"↓ Download {n} tiles"

    return (result_html(files, previews),
            gr.DownloadButton(value=path, label=label, visible=True))

# On small screens the result sits below the controls, so bring it into view.
SCROLL_JS = """
() => {
    if (window.innerWidth > 900) return;
    setTimeout(() => {
        const el = document.querySelector('#canvas');
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}
"""

# ── Styling ───────────────────────────────────────────────────────────────────
# Fonts: Newsreader (display, Google Fonts) + Switzer (UI, Fontshare;
# free for commercial use). Palette: warm ivory with adire indigo and ochre;
# the dark theme is a soft charcoal rather than pure black. Every colour is a
# CSS variable, and `.dark` swaps the set when Gradio runs in dark mode.
HEAD = (
    '<meta name="color-scheme" content="light dark">'
    '<meta name="darkreader-lock">'
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Newsreader:ital,opsz,wght@0,6..72,300..600;1,6..72,300..600&display=swap">'
    '<link rel="preconnect" href="https://api.fontshare.com">'
    '<link rel="stylesheet" href="https://api.fontshare.com/v2/css?'
    'f[]=switzer@400,500,600&display=swap">'
)

CSS = """
:root {
    color-scheme: light;
    --bg: #F6F3ED;   --sub: #EEE9E0;  --card: #FFFDF9;
    --tx: #1B1916;   --mut: #6D665C;  --line: #E4DDD1;
    --acc: #2F3C8F;  --gold: #C8902E; --clay: #B4492B;
    --btn: #1B1916;  --btn-tx: #FFFDF9;
    --shadow: 0 1px 2px rgba(27,25,22,.04), 0 8px 24px -12px rgba(27,25,22,.12);
    --fd: 'Newsreader', Georgia, 'Times New Roman', serif;
    --fb: 'Switzer', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.dark {
    color-scheme: dark;
    --bg: #1C1B19;   --sub: #232120;  --card: #272522;
    --tx: #F2EEE7;   --mut: #ABA398;  --line: #37342F;
    --acc: #AEB6F5;  --gold: #D9A857; --clay: #E07C5A;
    --btn: #F2EEE7;  --btn-tx: #1C1B19;
    --shadow: 0 1px 2px rgba(0,0,0,.2), 0 10px 30px -14px rgba(0,0,0,.45);
}

/* Map Gradio's own theme variables onto ours so built-in components match */
.gradio-container {
    --body-background-fill: var(--bg);
    --background-fill-primary: var(--bg);
    --background-fill-secondary: var(--sub);
    --block-background-fill: transparent;
    --block-border-color: transparent;
    --block-border-width: 0px;
    --block-shadow: none;
    --block-padding: 0px;
    --block-label-background-fill: transparent;
    --block-label-text-color: var(--tx);
    --block-title-text-color: var(--tx);
    --panel-background-fill: transparent;
    --body-text-color: var(--tx);
    --body-text-color-subdued: var(--mut);
    --input-background-fill: var(--card);
    --input-background-fill-focus: var(--card);
    --input-border-color: var(--line);
    --input-border-color-focus: var(--tx);
    --input-shadow: none;
    --input-shadow-focus: none;
    --border-color-primary: var(--line);
    --border-color-accent: var(--acc);
    --color-accent: var(--acc);
    --color-accent-soft: var(--sub);
    --slider-color: var(--acc);
    --loader-color: var(--acc);
    --link-text-color: var(--acc);
    --button-secondary-background-fill: var(--card);
    --button-secondary-background-fill-hover: var(--sub);
    --button-secondary-text-color: var(--tx);
    --button-secondary-border-color: var(--line);
    --font: var(--fb);
    --font-mono: var(--fb);
    --layout-gap: 0px;
    --form-gap-width: 0px;
}

/* The page background covers the whole window, not just the content column */
html, body, gradio-app, .gradio-container {
    background: var(--bg) !important;
}
html:has(body.dark), html:has(.dark) { background: #1C1B19 !important; }
body { min-height: 100vh; margin: 0; }
body, .gradio-container {
    color: var(--tx) !important;
    font-family: var(--fb) !important;
    -webkit-font-smoothing: antialiased;
}
/* Full-width page; content sections keep a centred 1216px column */
.gradio-container { max-width: none !important; width: 100% !important; margin: 0 !important; padding: 0 !important; }
.gradio-container > .main, .gradio-container .contain { padding: 0 !important; }
.gradio-container .main.fillable { max-width: none !important; margin: 0 !important; }
footer { display: none !important; }
.gradio-container .html-container { padding: 0 !important; }
.gradio-container .form { border: 0 !important; background: transparent !important;
                          box-shadow: none !important; gap: 0 !important; }

/* Visible keyboard focus */
.gradio-container button:focus-visible,
.gradio-container a:focus-visible,
.gradio-container input:focus-visible {
    outline: 2px solid var(--acc) !important;
    outline-offset: 2px !important;
}

/* ---------- Top: woven band + GPU note ---------- */
#band {
    height: 6px;
    background:
        conic-gradient(from 135deg at 50% 0, var(--acc) 90deg, transparent 0) 0 0 / 12px 6px repeat-x,
        var(--gold);
}
#gpu-notice {
    display: flex; align-items: center; justify-content: center; gap: 9px;
    padding: 9px 24px; background: var(--sub); border-bottom: 1px solid var(--line);
    font-size: 12.5px; line-height: 1.5; color: var(--mut); text-align: center; text-wrap: balance;
}
#gpu-notice .ni { width: 14px; height: 14px; flex-shrink: 0; fill: none;
                  stroke: var(--gold); stroke-width: 2; stroke-linecap: round; }
#gpu-notice strong { color: var(--tx); font-weight: 600; }

/* ---------- Header ---------- */
#hero {
    display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(0, 1fr);
    gap: 40px; align-items: end; padding: 34px max(32px, calc((100% - 1216px) / 2)) 26px;
}
#hero .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 16px;
               font-size: 13px; font-weight: 500; color: var(--mut); letter-spacing: .01em; }
#hero h1 {
    margin: 0; font-family: var(--fd); font-weight: 400; font-optical-sizing: auto;
    font-size: clamp(38px, 4.3vw, 58px); line-height: 1.02; letter-spacing: -.025em;
    color: var(--tx); text-wrap: balance;
}
#hero h1 em { font-style: italic; font-weight: 400; color: var(--acc); }
#hero p { margin: 0; color: var(--mut); font-size: 15px; line-height: 1.6; text-wrap: pretty; }

.mark { width: 24px; height: 24px; flex-shrink: 0; border-radius: 6px; overflow: hidden; }
.mark .k-g { fill: var(--gold); }
.mark .k-i { fill: #2F3C8F; }
.mark .k-c { fill: var(--clay); }

/* ---------- Main: controls + canvas ---------- */
#main { gap: 20px !important; padding: 0 max(32px, calc((100% - 1216px) / 2)) 28px !important;
        align-items: stretch !important; flex-wrap: nowrap !important; }
#controls { align-self: flex-start !important; }
#controls, #canvas {
    background: var(--card) !important; border: 1px solid var(--line) !important;
    border-radius: 20px !important; box-shadow: var(--shadow);
    padding: 22px 22px 22px !important; gap: 0 !important;
}

.card-head { display: flex; align-items: baseline; gap: 10px; margin: 0 !important; padding-bottom: 20px; }
.ctitle { font-family: var(--fd); font-weight: 400; font-size: 25px; line-height: 1.05;
          letter-spacing: -.02em; color: var(--tx); }
.hint { margin: 6px 0 0 !important; padding-bottom: 18px; font-size: 12.5px; line-height: 1.45; color: var(--mut); }

#trunc [data-testid="block-info"], #samples [data-testid="block-info"], #seed-box [data-testid="block-info"] {
    font-family: var(--fb) !important; font-size: 13px !important; font-weight: 500 !important;
    color: var(--tx) !important; text-transform: none !important; letter-spacing: 0 !important;
}
#trunc, #samples, #seed-box { overflow: visible !important; }
#trunc input[type=number] {
    border: 0 !important; background: transparent !important; box-shadow: none !important;
    font-family: var(--fb) !important; font-weight: 600 !important; font-size: 13px !important;
    color: var(--tx) !important; text-align: right; width: 56px !important; padding: 0 !important;
    font-variant-numeric: tabular-nums;
}
#trunc .reset-button, #trunc .min_value, #trunc .max_value { display: none !important; }
#trunc .tab-like-container { border: 0 !important; background: transparent !important; box-shadow: none !important; }
#trunc .wrap { overflow: visible !important; }
#trunc .slider_input_container { margin-top: 4px; padding: 6px 0; }
#trunc input[type=range] { accent-color: var(--acc); }
#trunc input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%;
    background: var(--card); border: 2px solid var(--acc); box-shadow: 0 1px 3px rgba(0,0,0,.18); cursor: pointer;
    margin-top: -5px; }
#trunc input[type=range]::-moz-range-thumb {
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--card); border: 2px solid var(--acc); box-shadow: 0 1px 3px rgba(0,0,0,.18); cursor: pointer; }

/* Samples: segmented control */
#samples { margin-bottom: 18px !important; }
#samples .wrap { display: grid !important; grid-template-columns: repeat(4, 1fr); gap: 4px !important;
                 background: var(--sub); border-radius: 12px; padding: 4px; margin-top: 8px; }
#samples label { justify-content: center !important; margin: 0 !important; border: 0 !important;
                 border-radius: 9px !important; padding: 7px 0 !important;
                 font-weight: 600 !important; font-size: 13px !important; color: var(--mut) !important;
                 background: transparent !important; box-shadow: none !important; cursor: pointer; }
#samples label.selected, #samples label:has(input:checked) {
    background: var(--card) !important; color: var(--tx) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.08), 0 2px 8px -2px rgba(0,0,0,.12) !important; }
#samples input[type=radio] { display: none !important; }
.dark #samples label.selected, .dark #samples label:has(input:checked) { background: var(--line) !important; }

/* Seed */
#seed-row { gap: 6px !important; align-items: flex-end !important; flex-wrap: nowrap !important; }
#seed-box input {
    border-radius: 12px !important; padding: 10px 13px !important; background: var(--bg) !important;
    font-family: var(--fb) !important; font-size: 14px !important; font-weight: 500 !important;
    font-variant-numeric: tabular-nums;
}
#random-btn {
    border: 1px solid var(--line) !important; border-radius: 12px !important;
    background: transparent !important; color: var(--tx) !important; box-shadow: none !important;
    font-family: var(--fb) !important; font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 13px !important; height: 42px; white-space: nowrap;
    flex: 0 0 auto !important; width: auto !important; min-width: 92px !important;
}
#random-btn:hover { background: var(--sub) !important; }

/* Primary action */
#gen-btn {
    width: 100%; margin-top: 8px !important; border: 0 !important; border-radius: 999px !important;
    padding: 14px !important; background: var(--btn) !important; color: var(--btn-tx) !important;
    box-shadow: none !important; font-family: var(--fb) !important; font-weight: 600 !important;
    font-size: 15px !important; letter-spacing: .01em; transition: background .15s ease;
}
#gen-btn:hover { background: var(--acc) !important; color: #fff !important; }
.dark #gen-btn:hover { color: var(--btn-tx) !important; }

/* ---------- Canvas: result + examples ---------- */
#res-head { align-items: center !important; justify-content: space-between !important;
            flex-wrap: nowrap !important; min-height: 34px; margin-bottom: 14px; }
#dl-btn {
    flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
    border: 1px solid var(--line) !important; border-radius: 999px !important; padding: 7px 14px !important;
    background: var(--card) !important; color: var(--tx) !important; box-shadow: none !important;
    font-family: var(--fb) !important; font-weight: 600 !important; font-size: 13px !important;
}
#dl-btn:hover { background: var(--btn) !important; color: var(--btn-tx) !important; border-color: var(--btn) !important; }

#result-area .empty {
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px;
    height: 230px; border: 1px dashed var(--line); border-radius: 14px;
    background: var(--bg); color: var(--mut); font-size: 13.5px; text-align: center; padding: 0 20px;
}
#result-area .empty .mark { width: 32px; height: 32px; border-radius: 8px; }
#result-area .tiles { display: grid; gap: 10px;
                      grid-template-columns: repeat(var(--n), minmax(0, 1fr));
                      max-width: calc(var(--n) * 250px); }
#result-area figure, #examples figure {
    position: relative; margin: 0; border-radius: 12px; overflow: hidden; aspect-ratio: 1;
    background: var(--sub);
}
#result-area img, #examples img { display: block; width: 100%; height: 100%; object-fit: cover; }
/* Click a tile to see it larger */
#result-area .zoom { display: block; width: 100%; height: 100%; position: relative; cursor: zoom-in;
                    padding: 0 !important; margin: 0 !important; }
#result-area .zoom img { transition: transform .25s ease; }
#result-area .zoom:hover img { transform: scale(1.03); }
#result-area .zoom-hint {
    position: absolute; right: 8px; top: 8px; width: 28px; height: 28px; border-radius: 999px;
    display: grid; place-items: center; background: rgba(20,18,16,.55); backdrop-filter: blur(4px);
    opacity: 0; transition: opacity .15s ease;
}
#result-area .zoom:hover .zoom-hint, #result-area .zoom:focus-visible .zoom-hint { opacity: 1; }
#result-area .zoom-hint svg { width: 14px; height: 14px; fill: none; stroke: #fff; stroke-width: 2;
                              stroke-linecap: round; stroke-linejoin: round; }

#result-area .tile-dl {
    position: absolute; right: 8px; bottom: 8px; width: 32px; height: 32px; border-radius: 999px;
    padding: 0 !important; display: grid; place-items: center; background: var(--card); color: var(--tx) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.25); opacity: 0; transition: opacity .15s ease, background .15s ease;
}
#result-area figure:hover .tile-dl, #result-area .tile-dl:focus-visible { opacity: 1; }
#result-area .tile-dl:hover { background: var(--btn); color: var(--btn-tx) !important; }
#result-area .tile-dl svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2;
                            stroke-linecap: round; stroke-linejoin: round; }
@media (hover: none) { #result-area .tile-dl, #result-area .zoom-hint { opacity: 1; } }

/* Larger view (CSS :target lightbox) */
.wlb { position: fixed; inset: 0; z-index: 1000; display: none;
       align-items: center; justify-content: center; padding: 20px; }
.wlb:target { display: flex; }
.wlb-bg { position: absolute; inset: 0; background: rgba(15,13,11,.72); backdrop-filter: blur(6px); cursor: zoom-out; }
.wlb-box { position: relative; background: var(--card); border: 1px solid var(--line);
           border-radius: 18px; padding: 12px; box-shadow: 0 30px 80px -20px rgba(0,0,0,.5); }
#result-area .wlb-box img { display: block; width: min(520px, 84vw, calc(100vh - 150px)); height: auto; aspect-ratio: 1;
               border-radius: 10px; object-fit: cover; }
.wlb-bar { display: flex; align-items: center; justify-content: space-between; gap: 10px 12px;
           flex-wrap: wrap; padding: 12px 4px 2px; font-size: 13px; color: var(--mut); }
.wlb-bar > span:first-child { white-space: nowrap; }
.wlb-actions { display: flex; gap: 8px; }
.wlb-x, .wlb-dl { text-decoration: none !important; font-weight: 600; font-size: 13px;
                  border-radius: 999px; padding: 8px 14px; white-space: nowrap; }
.wlb-x  { color: var(--tx) !important; border: 1px solid var(--line); }
.wlb-x:hover { background: var(--sub); }
.wlb-dl { color: var(--btn-tx) !important; background: var(--btn); }
.wlb-dl:hover { background: var(--acc); color: #fff !important; }

#examples { margin-top: 20px; padding-top: 18px; border-top: 1px solid var(--line); }
#examples .card-head { padding-bottom: 12px; }
#examples .ctitle { font-size: 22px; }
#examples .strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }

/* ---------- Footer (matches the StableDiffusion space) ---------- */
#app-footer {
    padding: 16px max(32px, calc((100% - 1216px) / 2)); display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px 24px; border-top: 1px solid var(--line);
    font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--mut);
}
#app-footer nav { display: flex; flex-wrap: wrap; gap: 8px 20px; }
#app-footer a { display: inline-flex; align-items: center; gap: 6px;
                color: var(--mut) !important; text-decoration: none !important; transition: color .15s ease; }
#app-footer a:hover { color: var(--acc) !important; }
#app-footer svg { width: 12px; height: 12px; fill: currentColor; flex-shrink: 0; }

/* ---------- Small screens ---------- */
@media (max-width: 900px) {
    #hero { grid-template-columns: 1fr; gap: 14px; padding: 26px 18px 20px; }
    #main { flex-direction: column !important; flex-wrap: wrap !important; padding: 0 12px 20px !important; gap: 14px !important; }
    #controls { align-self: stretch !important; }
    #controls, #canvas { min-width: 100% !important; padding: 18px !important; }
    #gpu-notice { padding: 10px 16px; justify-content: flex-start; text-align: left; align-items: flex-start; }
    #gpu-notice .ni { margin-top: 3px; }
    #result-area .tiles { grid-template-columns: repeat(min(var(--n), 2), minmax(0, 1fr)); max-width: none; }
    #examples .strip { grid-template-columns: repeat(2, 1fr); }
    #app-footer, #app-footer nav { flex-direction: column; align-items: flex-start; }
    #app-footer { padding: 16px 18px; }
    #app-footer nav { gap: 10px; }
}

@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="WaxFashionStyleGAN · Pace AI Lab", fill_width=True) as demo:

    gr.HTML(TOP, padding=False)
    gr.HTML(HEADER, padding=False)

    with gr.Row(elem_id="main", equal_height=False):

        with gr.Column(scale=0, min_width=360, elem_id="controls"):
            gr.HTML(CTL_HEAD, padding=False)

            truncation = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.7, step=0.05,
                label="Truncation Psi", elem_id="trunc",
            )
            gr.HTML(TRUNC_HINT, padding=False)

            n_samples = gr.Radio(
                choices=["1", "2", "3", "4"], value="1",
                label="Samples", elem_id="samples",
            )

            with gr.Row(elem_id="seed-row"):
                seed_box = gr.Textbox(
                    label="Seed", value="137", placeholder="integer or random",
                    scale=1, elem_id="seed-box",
                )
                rand_btn = gr.Button("↺ random", scale=0, min_width=92, elem_id="random-btn")
            gr.HTML(SEED_HINT, padding=False)

            gen_btn = gr.Button("✦ Generate", elem_id="gen-btn")

        with gr.Column(scale=1, elem_id="canvas"):
            with gr.Row(elem_id="res-head"):
                gr.HTML('<span class="ctitle">Your pattern</span>', padding=False)
                dl_btn = gr.DownloadButton(
                    "↓ Download tile", visible=False, elem_id="dl-btn", scale=0,
                )
            result = gr.HTML(EMPTY, elem_id="result-area", padding=False)
            gr.HTML(examples_html(), padding=False)

    gr.HTML(FOOTER, padding=False)

    gen_btn.click(fn=None, js=SCROLL_JS)
    gen_btn.click(
        fn=run,
        inputs=[truncation, n_samples, seed_box],
        outputs=[result, dl_btn],
    )
    rand_btn.click(fn=random_seed, inputs=[], outputs=[seed_box])

if __name__ == "__main__":
    # SSR (Gradio 6's experimental server-side rendering) is off: on Spaces it
    # causes intermittent 502s and "Could not resolve app config" errors.
    demo.launch(css=CSS, head=HEAD, allowed_paths=[DOWNLOAD_ROOT], ssr_mode=False)
