// Wax Fashion StyleGAN frontend: talks to the Azure backend's GET /image.
// Each tile is one request; the backend caches recent tiles, so the PNG
// download of a tile you already see doesn't regenerate it.

const API = ((window.WAX_CONFIG && window.WAX_CONFIG.apiUrl) || "").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);

const psi = $("psi"), psiOut = $("psi-out"), seedBox = $("seed");
const genBtn = $("gen-btn"), dlBtn = $("dl-btn"), statusEl = $("status"), result = $("result");
const lightbox = $("lightbox"), lbImg = $("lb-img"), lbLabel = $("lb-label");

const DOWNLOAD_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>';
const randomSeed = () => String(100_000_000 + Math.floor(Math.random() * 900_000_000));

let current = null;   // { psi, seeds: [] } for the tiles on screen
let lbSeed = null;

psi.addEventListener("input", () => { psiOut.textContent = Number(psi.value).toFixed(2); });
$("rand-btn").addEventListener("click", () => { seedBox.value = randomSeed(); });
seedBox.addEventListener("keydown", (e) => { if (e.key === "Enter") generate(); });
genBtn.addEventListener("click", generate);
dlBtn.addEventListener("click", downloadAll);
$("lb-close").addEventListener("click", () => lightbox.close());
$("lb-dl").addEventListener("click", () => downloadOne(lbSeed));
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) lightbox.close(); });

// Wake the backend as soon as someone opens the page, so a scaled-to-zero
// container is usually warm by the time they press Generate.
if (API) fetch(`${API}/health`).catch(() => {});

function parseSeed(s) {
  s = (s || "").trim().toLowerCase();
  if (s === "" || s === "random") return Number(randomSeed());
  const n = Number.parseInt(s, 10);
  if (!Number.isFinite(n)) return Number(randomSeed());
  return ((n % 2 ** 32) + 2 ** 32) % 2 ** 32;
}

function imageUrl(seed, p, format) {
  return `${API}/image?seed=${seed}&psi=${p}&format=${format}`;
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("err", isError);
}

async function fetchImage(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let msg = `The generator returned an error (${res.status}).`;
    try { msg = (await res.json()).detail || msg; } catch {}
    if (typeof msg !== "string") msg = "Those settings aren't valid.";
    throw new Error(msg);
  }
  return res.blob();
}

async function generate() {
  if (!API || API.includes("REPLACE-WITH")) {
    setStatus("The backend URL isn't set yet (frontend/config.js).", true);
    return;
  }
  const n = Number(document.querySelector('input[name="n"]:checked').value);
  const base = parseSeed(seedBox.value);
  const p = Number(psi.value).toFixed(2);
  const seeds = Array.from({ length: n }, (_, i) => (base + i) % 2 ** 32);
  current = { psi: p, seeds };

  genBtn.disabled = true;
  genBtn.textContent = "Generating…";
  dlBtn.hidden = true;
  result.innerHTML = `<div class="tiles" data-n="${n}" style="--n:${n}">${
    seeds.map(() => '<figure class="tile loading" aria-busy="true"></figure>').join("")}</div>`;
  const tiles = result.querySelectorAll(".tile");
  if (window.innerWidth <= 900) $("canvas").scrollIntoView({ behavior: "smooth", block: "start" });

  // If the first tile is slow, the container is most likely starting up.
  setStatus(n > 1 ? `Generating pattern 1 of ${n}…` : "Generating…");
  const slow = setTimeout(() => setStatus("Waking up the generator, this can take up to a minute…"), 8000);

  let ok = 0;
  try {
    for (let i = 0; i < n; i++) {
      if (i > 0) setStatus(`Generating pattern ${i + 1} of ${n}…`);
      const tile = tiles[i];
      try {
        const blob = await fetchImage(imageUrl(seeds[i], p, "jpg"));
        if (i === 0) clearTimeout(slow);
        renderTile(tile, URL.createObjectURL(blob), seeds[i], i, n);
        ok++;
      } catch (err) {
        if (i === 0) clearTimeout(slow);
        tile.classList.remove("loading");
        tile.classList.add("failed");
        tile.removeAttribute("aria-busy");
        tile.textContent = "Couldn't generate this one.";
        setStatus(err.message || "Couldn't reach the generator. Please try again.", true);
        if (/too many/i.test(err.message)) break;
      }
    }
    if (ok === n) setStatus("");
  } finally {
    clearTimeout(slow);
    genBtn.disabled = false;
    genBtn.textContent = "✦ Generate";
    if (ok > 0) {
      dlBtn.hidden = false;
      dlBtn.textContent = n === 1 ? "↓ Download tile" : `↓ Download ${n} tiles`;
    }
  }
}

function renderTile(tile, src, seed, i, n) {
  const label = n > 1 ? `Pattern ${i + 1} of ${n}` : "Your pattern";
  tile.classList.remove("loading");
  tile.removeAttribute("aria-busy");
  tile.innerHTML =
    `<button class="zoom" type="button" aria-label="View ${label.toLowerCase()} larger">` +
    `<img src="${src}" alt="Generated African wax-inspired pattern, seed ${seed}"></button>` +
    `<button class="tile-dl" type="button" aria-label="Download ${label.toLowerCase()}" title="Download">${DOWNLOAD_ICON}</button>`;
  tile.querySelector(".zoom").addEventListener("click", () => {
    lbSeed = seed;
    lbImg.src = src;
    lbLabel.textContent = `${label} · seed ${seed}`;
    lightbox.showModal();
  });
  tile.querySelector(".tile-dl").addEventListener("click", () => downloadOne(seed));
}

function saveBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

const pngName = (seed) => `wax_pattern_seed${seed}.png`;

async function downloadOne(seed) {
  if (!current || seed == null) return;
  try {
    saveBlob(await fetchImage(imageUrl(seed, current.psi, "png")), pngName(seed));
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function downloadAll() {
  if (!current) return;
  const { seeds, psi: p } = current;
  if (seeds.length === 1 || !window.JSZip) {
    for (const s of seeds) await downloadOne(s);
    return;
  }
  dlBtn.disabled = true;
  const label = dlBtn.textContent;
  dlBtn.textContent = "Preparing…";
  try {
    const zip = new JSZip();
    for (const s of seeds) zip.file(pngName(s), await fetchImage(imageUrl(s, p, "png")));
    saveBlob(await zip.generateAsync({ type: "blob" }),
             `wax_patterns_seed${seeds[0]}-${seeds[seeds.length - 1]}.zip`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    dlBtn.disabled = false;
    dlBtn.textContent = label;
  }
}
