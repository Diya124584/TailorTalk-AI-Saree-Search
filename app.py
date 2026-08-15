import os
import re
from pathlib import Path

import cv2
import faiss
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel

# Gemini / LangChain agent (used only for intent understanding and tool calling)
try:
    from langchain.tools import tool
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    tool = None
    create_agent = None
    ChatGoogleGenerativeAI = None


# ============================================================
# TailorTalk — FINAL VISUAL + COLOR SEARCH ENGINE
# ============================================================
# Ranking philosophy:
#   DINOv2 visual similarity = overall look / shape / styling
#   Image palette similarity = dominant colours + shades
#   CSV colour metadata = small supporting signal
#
# Final score:
#   0.70 * DINO
# + 0.25 * image colour similarity
# + 0.05 * CSV/text colour similarity
#
# The query product itself is excluded when it can be identified.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"

SAREE_CSV = DATA_DIR / "saree.csv"
EMBEDDING_METADATA_FILE = DATA_DIR / "searchable_metadata.csv"
FAISS_INDEX_FILE = DATA_DIR / "dinov2_faiss.index"

if not EMBEDDING_METADATA_FILE.exists():
    EMBEDDING_METADATA_FILE = DATA_DIR / "embedding_metadata.csv"
if not FAISS_INDEX_FILE.exists():
    FAISS_INDEX_FILE = DATA_DIR / "saree_dinov2_faiss.index"

for required in (SAREE_CSV, EMBEDDING_METADATA_FILE, FAISS_INDEX_FILE):
    if not required.exists():
        raise FileNotFoundError(f"Required file missing: {required}")

saree_df = pd.read_csv(SAREE_CSV)
embedding_metadata = pd.read_csv(EMBEDDING_METADATA_FILE)


# ------------------------------------------------------------
# Metadata alignment
# ------------------------------------------------------------

if "image_file" not in saree_df.columns:
    saree_df["image_file"] = saree_df.index.astype(str) + ".jpg"

if "image_file" in embedding_metadata.columns and "Name" in embedding_metadata.columns:
    search_catalogue = embedding_metadata.copy()
else:
    search_catalogue = embedding_metadata.merge(
        saree_df, on="image_file", how="left", sort=False
    )

if "Name" not in search_catalogue.columns:
    name_col = next(
        (c for c in search_catalogue.columns if c.lower() in {"name", "product_name", "title"}),
        None,
    )
    if name_col:
        search_catalogue["Name"] = search_catalogue[name_col]
    else:
        search_catalogue["Name"] = "Saree"

if "SKU" not in search_catalogue.columns:
    raise KeyError("The catalogue must contain a SKU column.")

if "image_file" not in search_catalogue.columns:
    raise KeyError("The searchable metadata must contain image_file.")

# Make sure common numeric columns exist.
for col in ["Retail Price", "Discounted Price", "Stock"]:
    if col not in search_catalogue.columns:
        search_catalogue[col] = 0

if "Website Link" not in search_catalogue.columns:
    search_catalogue["Website Link"] = "#"


# ------------------------------------------------------------
# Colour / fabric vocabulary
# ------------------------------------------------------------

FABRIC_FAMILIES = {
    "tussar": {"tussar", "semi tussar"},
    "semi tussar": {"tussar", "semi tussar"},
    "banarasi": {"banarasi", "pashmina banarasi"},
    "organza": {"organza", "tissue organza"},
    "tissue": {"tissue", "tissue organza"},
    "silk": {"silk", "semi silk", "kanchipuram", "mysore silk", "chanderi silk"},
    "cotton": {"cotton", "mul cotton", "chikankari", "silk cotton"},
    "georgette": {"georgette", "poly georgette"},
    "chiffon": {"chiffon"},
}

FABRICS = [
    "semi tussar", "tussar", "banarasi", "organza", "tissue", "georgette",
    "pashmina", "kanchipuram", "mysore silk", "linen", "mul cotton",
    "silk cotton", "cotton", "chikankari", "semi silk", "munga", "crape",
    "chiffon", "satin", "silk",
]

# Longest first prevents "blue" from interfering with "navy blue".
COLORS = [
    "navy blue", "sky blue", "royal blue", "aqua blue", "powder blue",
    "baby blue", "dark green", "lime green", "mint green", "forest green",
    "olive green", "sea green", "light green", "bottle green",
    "dark pink", "hot pink", "rani pink", "baby pink", "dusty pink",
    "rose pink", "pastel pink", "peach pink", "blush pink",
    "dark purple", "light purple", "royal purple",
    "mustard yellow", "lemon yellow", "pale yellow",
    "off white", "off-white",
    "magenta", "fuchsia", "maroon", "burgundy", "wine", "coral", "peach",
    "orange", "rust", "brown", "beige", "ivory", "cream", "white", "black",
    "grey", "gray", "silver", "gold", "lavender", "purple", "pink", "red",
    "blue", "green", "yellow",
]

COLOR_FAMILY = {
    "red": "red", "maroon": "red", "burgundy": "red", "wine": "red", "rust": "red",
    "pink": "pink", "dark pink": "pink", "hot pink": "pink", "rani pink": "pink",
    "baby pink": "pink", "dusty pink": "pink", "rose pink": "pink",
    "pastel pink": "pink", "peach pink": "pink", "blush pink": "pink",
    "magenta": "pink", "fuchsia": "pink",
    "purple": "purple", "dark purple": "purple", "light purple": "purple",
    "royal purple": "purple", "lavender": "purple",
    "blue": "blue", "navy blue": "blue", "sky blue": "blue", "royal blue": "blue",
    "aqua blue": "blue", "powder blue": "blue", "baby blue": "blue",
    "green": "green", "dark green": "green", "lime green": "green",
    "mint green": "green", "forest green": "green", "olive green": "green",
    "sea green": "green", "light green": "green", "bottle green": "green",
    "yellow": "yellow", "mustard yellow": "yellow", "lemon yellow": "yellow",
    "pale yellow": "yellow", "gold": "yellow",
    "orange": "orange", "peach": "orange", "coral": "orange",
    "brown": "brown", "beige": "brown", "cream": "neutral",
    "ivory": "neutral", "white": "neutral", "off white": "neutral",
    "off-white": "neutral", "black": "black", "grey": "grey",
    "gray": "grey", "silver": "grey",
}

# Palette cache is deliberately kept in memory because catalogue images
# are reused for every query.
PALETTE_CACHE = {}


def _contains_phrase(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def extract_attributes(text):
    """Extract colour/fabric words without substring false positives."""
    text = str(text or "").lower()
    found_colors = {
        c for c in COLORS
        if _contains_phrase(text, c)
    }
    found_fabrics = {
        f for f in FABRICS
        if _contains_phrase(text, f)
    }
    return {
        "colors": found_colors,
        "fabrics": found_fabrics,
        "raw_tokens": set(re.findall(r"\b\w+\b", text)),
    }


def row_text_for_attributes(row):
    """Use all useful CSV text fields, especially Colour if present."""
    fields = []
    for col in row.index:
        if col.lower() in {
            "name", "product_name", "title", "colour", "color",
            "description", "material", "fabric", "category", "type"
        }:
            value = row.get(col)
            if pd.notna(value):
                fields.append(str(value))
    return " ".join(fields)


# Use actual CSV metadata when available, not only the product title.
search_catalogue["attr"] = search_catalogue.apply(
    lambda r: extract_attributes(row_text_for_attributes(r)),
    axis=1,
)


# ============================================================
# 1. ROBUST SAREE FOREGROUND + PALETTE EXTRACTION
# ============================================================

def extract_saree_foreground_pixels(pil_img):
    """
    Build a cloth-focused pixel set.

    The old implementation simply removed bright/low-saturation pixels.
    That fails badly for pastel sarees because pastel fabric itself can be
    bright and low-saturation.

    This version estimates the studio background from the image border,
    measures LAB distance from that background, and combines it with
    central/lower spatial weighting. This keeps pale saree colours while
    suppressing most white/cream studio background.
    """
    rgb = np.array(pil_img.convert("RGB"))
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)

    lab = color.rgb2lab(small.astype(np.float32) / 255.0)

    # Border pixels are a good estimate of the studio backdrop.
    border = np.concatenate([
        lab[:18].reshape(-1, 3),
        lab[-18:].reshape(-1, 3),
        lab[:, :18].reshape(-1, 3),
        lab[:, -18:].reshape(-1, 3),
    ], axis=0)

    bg_lab = np.median(border, axis=0)
    bg_dist = np.linalg.norm(lab - bg_lab, axis=2)

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    # Saree is usually concentrated around the middle/lower image.
    yy, xx = np.mgrid[0:256, 0:256]
    center_x = 1.0 - np.abs(xx - 128.0) / 128.0
    vertical = np.clip((yy - 12.0) / 210.0, 0.0, 1.0)
    spatial = 0.45 * center_x + 0.55 * vertical

    # Colour distance is the strongest clue; saturation helps when
    # background and fabric are similarly bright.
    score = (
        0.78 * bg_dist
        + 20.0 * saturation
        + 7.0 * spatial
    )

    # Ignore extreme outer margins.
    valid = np.zeros((256, 256), dtype=bool)
    valid[8:250, 18:238] = True
    score[~valid] = -np.inf

    flat = score[valid]
    finite = flat[np.isfinite(flat)]

    # Keep the strongest cloth-like 38% of the central/lower pixels.
    # This is intentionally generous for pale sarees.
    if finite.size:
        threshold = np.percentile(finite, 62)
        mask = valid & (score >= threshold)
    else:
        mask = valid

    # Clean tiny isolated regions.
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    pixels = small[mask_u8 > 0]

    # Safe fallback for very uniform images.
    if len(pixels) < 800:
        crop = small[25:235, 30:226]
        pixels = crop.reshape(-1, 3)

    # Deterministic downsample keeps KMeans fast.
    if len(pixels) > 12000:
        rng = np.random.default_rng(42)
        take = rng.choice(len(pixels), 12000, replace=False)
        pixels = pixels[take]

    return pixels


def extract_color_palette(fg_pixels, k=5):
    """Return dominant LAB colours sorted by actual proportion."""
    lab_pixels = color.rgb2lab(
        fg_pixels.astype(np.float32) / 255.0
    )

    n_clusters = min(k, len(lab_pixels))
    if n_clusters < 1:
        return np.zeros((1, 3), dtype=np.float32), np.ones(1, dtype=np.float32)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=5,
        random_state=42,
    )
    labels = kmeans.fit_predict(lab_pixels)

    centers = kmeans.cluster_centers_.astype(np.float32)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
    weights = counts / max(counts.sum(), 1.0)

    # CRITICAL FIX: KMeans cluster 0 is NOT necessarily the dominant colour.
    order = np.argsort(weights)[::-1]
    centers = centers[order]
    weights = weights[order]

    return centers, weights


def get_palette(image):
    pixels = extract_saree_foreground_pixels(image)
    return extract_color_palette(pixels, k=5)


def palette_similarity(query_palette, query_weights, cand_palette, cand_weights):
    """
    Perceptual palette similarity.

    Each colour gets compared to its best matching colour in the other
    palette. Dominant colours naturally matter more because of their weights.
    The score is symmetric and uses LAB distances, so close shades remain
    similar instead of collapsing to zero.
    """
    q = np.asarray(query_palette, dtype=np.float32)
    c = np.asarray(cand_palette, dtype=np.float32)
    qw = np.asarray(query_weights, dtype=np.float32)
    cw = np.asarray(cand_weights, dtype=np.float32)

    d = np.linalg.norm(q[:, None, :] - c[None, :, :], axis=2)

    # LAB distance -> smooth similarity.
    pair_sim = np.exp(-d / 38.0)

    q_to_c = float(np.sum(qw * np.max(pair_sim, axis=1)))
    c_to_q = float(np.sum(cw * np.max(pair_sim, axis=0)))
    symmetric = 0.5 * (q_to_c + c_to_q)

    # Dominant shade gets extra attention.
    dominant = float(np.max(pair_sim[0]))

    # Overall colour score: dominant shade + full palette relationship.
    score = 0.65 * dominant + 0.35 * symmetric
    return float(np.clip(score, 0.0, 1.0))


def get_catalogue_palette(idx, image_file):
    if idx in PALETTE_CACHE:
        return PALETTE_CACHE[idx]

    image_path = IMAGE_DIR / str(image_file)

    if image_path.exists():
        try:
            with Image.open(image_path) as img:
                result = get_palette(img)
            PALETTE_CACHE[idx] = result
            return result
        except Exception:
            pass

    # Neutral fallback. This is only used if an image cannot be read.
    fallback = (
        np.array([[50.0, 0.0, 0.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )
    PALETTE_CACHE[idx] = fallback
    return fallback


# ============================================================
# 2. CSV COLOUR / TEXT SUPPORT
# ============================================================

def colour_metadata_similarity(query_colors, candidate_colors):
    """
    Small supporting signal from CSV/title text.

    Exact colour family = strong.
    Related family = moderate.
    No usable metadata = neutral.
    """
    if not query_colors or not candidate_colors:
        return 0.50

    best = 0.0
    for q in query_colors:
        q_family = COLOR_FAMILY.get(q, q)
        for c in candidate_colors:
            c_family = COLOR_FAMILY.get(c, c)

            if q == c:
                best = max(best, 1.0)
            elif q_family == c_family:
                best = max(best, 0.82)
            elif {q_family, c_family} <= {"pink", "red", "purple", "orange"}:
                best = max(best, 0.55)
            else:
                best = max(best, 0.12)

    return float(best)


def get_query_catalogue_row(query_image_path, faiss_indices, faiss_scores):
    """
    Identify the uploaded image if it is one of our catalogue images.
    Prefer exact filename; otherwise use an extremely high DINO identity
    score. Returns (row_index, SKU).
    """
    query_name = Path(query_image_path).name

    for score, idx in zip(faiss_scores, faiss_indices):
        if idx < 0 or idx >= len(search_catalogue):
            continue
        row = search_catalogue.iloc[int(idx)]
        if str(row["image_file"]) == query_name:
            return int(idx), str(row["SKU"])

    for score, idx in zip(faiss_scores, faiss_indices):
        if idx < 0 or idx >= len(search_catalogue):
            continue
        if float(score) >= 0.999:
            row = search_catalogue.iloc[int(idx)]
            return int(idx), str(row["SKU"])

    return -1, None


# ============================================================
# 3. DINOv2 + FAISS
# ============================================================

MODEL_NAME = "facebook/dinov2-small"
device = torch.device("cpu")

image_processor = None
visual_model = None
faiss_index = None


def load_resources():
    global image_processor, visual_model, faiss_index

    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        visual_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        visual_model.eval()

    if faiss_index is None:
        faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))


def get_dino_embedding(image):
    load_resources()

    inputs = image_processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = visual_model(**inputs)

    embedding = (
        outputs.last_hidden_state[:, 0]
        .cpu()
        .numpy()
        .astype("float32")[0]
    )

    norm = np.linalg.norm(embedding)
    if norm > 1e-12:
        embedding /= norm

    return embedding


# ============================================================
# 4. FINAL HYBRID SEARCH
# ============================================================

def execute_hybrid_search(query_image_path, user_text="", top_k=5):
    load_resources()

    if not query_image_path or not Path(query_image_path).exists():
        raise FileNotFoundError(f"Image not found: {query_image_path}")

    query_img = Image.open(query_image_path).convert("RGB")

    # ---------- Query features ----------
    query_emb = get_dino_embedding(query_img).reshape(1, -1)
    faiss.normalize_L2(query_emb)

    query_palette, query_weights = get_palette(query_img)
    query_text_attr = extract_attributes(user_text)

    # Search a large enough visual pool so a good colour match isn't
    # lost just because it was rank 6-20 in raw DINO retrieval.
    search_k = min(200, faiss_index.ntotal)
    raw_scores, indices = faiss_index.search(query_emb, search_k)

    excluded_idx, excluded_sku = get_query_catalogue_row(
        query_image_path,
        indices[0],
        raw_scores[0],
    )

    # If the image itself is a catalogue product, use its CSV colour
    # information as a query-side signal.
    query_catalogue_attr = None
    if excluded_idx >= 0:
        query_catalogue_attr = search_catalogue.iloc[excluded_idx]["attr"]

    query_colors = set(query_text_attr["colors"])
    if query_catalogue_attr is not None:
        query_colors.update(query_catalogue_attr["colors"])

    # ---------- Candidate reranking ----------
    candidates = []

    for dino_raw, idx in zip(raw_scores[0], indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(search_catalogue):
            continue

        row = search_catalogue.iloc[idx]
        sku = str(row["SKU"])

        # Exact product exclusion.
        if excluded_sku and sku == excluded_sku:
            continue
        if excluded_idx >= 0 and idx == excluded_idx:
            continue

        try:
            stock = int(float(row.get("Stock", 0)))
        except Exception:
            stock = 0

        # Keep catalogue recommendations purchasable.
        if stock <= 0:
            continue

        dino_sim = float(np.clip(dino_raw, 0.0, 1.0))

        cand_palette, cand_weights = get_catalogue_palette(
            idx,
            row["image_file"],
        )

        image_color_sim = palette_similarity(
            query_palette,
            query_weights,
            cand_palette,
            cand_weights,
        )

        metadata_color_sim = colour_metadata_similarity(
            query_colors,
            row["attr"]["colors"],
        )

        # IMPORTANT: visual appearance remains dominant.
        final_score = (
            0.70 * dino_sim
            + 0.25 * image_color_sim
            + 0.05 * metadata_color_sim
        )

        candidates.append({
            "image_file": str(row["image_file"]),
            "name": str(row["Name"]),
            "sku": sku,
            "dino_score": dino_sim,
            "color_score": float(image_color_sim),
            "metadata_color_score": float(metadata_color_sim),
            "final_score": float(np.clip(final_score, 0.0, 1.0)),
            "retail_price": float(row.get("Retail Price", 0.0)),
            "discounted_price": float(row.get("Discounted Price", 0.0)),
            "stock": stock,
            "website_link": str(row.get("Website Link", "#")),
        })

    # Final ranking: highest combined visual + colour similarity first.
    candidates.sort(
        key=lambda item: (
            item["final_score"],
            item["dino_score"],
            item["color_score"],
        ),
        reverse=True,
    )

    # Deduplicate SKU.
    results = []
    seen_skus = set()

    for item in candidates:
        if item["sku"] in seen_skus:
            continue
        seen_skus.add(item["sku"])
        results.append(item)

        if len(results) >= top_k:
            break

    return results, excluded_sku


# ============================================================
# 5. OUTPUT
# ============================================================

def format_results(results, excluded_sku):
    if not results:
        return (
            "### No matching sarees found.\n\n"
            "Try another saree image or check that the catalogue images exist."
        )

    output = []

    if excluded_sku:
        output.append(
            f"> 🛡️ **Excluded Query Product SKU:** `{excluded_sku}`\n"
        )

    output.append("## 👗 Recommended Sarees\n")
    output.append(
        "_Ranked using overall visual appearance, dominant colours/shades, "
        "and a small CSV colour-metadata signal._\n"
    )

    for i, res in enumerate(results, start=1):
        output.append(
            f"### {i}. {res['name']}\n\n"
            f"- **Match Score:** `{res['final_score'] * 100:.1f}%`\n"
            f"  - Overall Visual (DINOv2): `{res['dino_score'] * 100:.1f}%`\n"
            f"  - Colour & Shade Similarity: `{res['color_score'] * 100:.1f}%`\n"
            f"  - CSV Colour Match: `{res['metadata_color_score'] * 100:.1f}%`\n"
            f"- **SKU:** `{res['sku']}`\n"
            f"- **Retail Price:** ₹{res['retail_price']:,.2f}\n"
            f"- **Discounted Price:** ₹{res['discounted_price']:,.2f}\n"
            f"- **Stock:** {res['stock']} units\n"
            f"- **Product Link:** [View Product]({res['website_link']})\n\n"
            "---"
        )

    return "\n".join(output)


# ============================================================
# 5. GEMINI / LANGCHAIN AGENT LAYER
# ============================================================
#
# IMPORTANT:
# Gemini does NOT calculate, modify, or override similarity scores.
# The verified hybrid engine above remains the only ranking engine.
#
# Gemini's role is:
#   user intent -> callable search tool -> natural response
#
# If Gemini is unavailable, the exact same hybrid search works directly.
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

gemini_agent = None
gemini_enabled = False


def _format_agent_results(results, excluded_sku):
    """Format the verified hybrid-search results exactly as the app expects."""
    return format_results(results, excluded_sku)


if tool is not None and create_agent is not None and ChatGoogleGenerativeAI is not None and GEMINI_API_KEY:

    @tool
    def visual_saree_search_tool(image_path: str, top_k: int = 5, user_request: str = "") -> str:
        """
        Search TailorTalk's verified saree catalogue for visually similar
        products using DINOv2 + FAISS + colour/shade reranking.

        image_path must be a local path to the user's uploaded/downloaded image.
        The returned JSON contains the authoritative product information and
        similarity scores. Do not invent or modify those values.
        user_request may contain the user's colour/style preference; pass it
        through when relevant.
        """
        try:
            top_k = max(1, min(int(top_k), 5))
        except Exception:
            top_k = 5

        results, excluded_sku = execute_hybrid_search(
            query_image_path=image_path,
            user_text=user_request or "",
            top_k=top_k,
        )

        import json
        return json.dumps(
            {
                "excluded_query_sku": excluded_sku,
                "results": results,
            },
            ensure_ascii=False,
            default=str,
        )

    try:
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

        gemini_agent = create_agent(
            model=gemini_llm,
            tools=[visual_saree_search_tool],
            system_prompt="""
You are TailorTalk, an AI saree shopping assistant.

Your job is to understand the user's request and use the
visual_saree_search_tool when the user wants visually similar
sarees or recommendations based on the uploaded image.

IMPORTANT:
- The search tool is the ONLY source of catalogue products.
- When visual search is requested, ALWAYS call the tool.
- Never invent product names, SKUs, prices, stock, links, or scores.
- Never calculate, change, round, reinterpret, or override the tool's
  similarity scores.
- Do not claim that a product is similar unless it appears in the tool result.
- Keep the response concise and useful.
- The tool's returned scores are authoritative.

The underlying ranking is deliberately deterministic:
DINOv2 visual similarity + image colour/shade similarity + a small
CSV colour-metadata signal. Do not replace that ranking with your own
judgement.

If the user asks a general fashion question that does not require
catalogue search, answer normally.
""",
        )

        gemini_enabled = True
        print("Gemini + LangChain agent initialized successfully.")

    except Exception as exc:
        gemini_agent = None
        gemini_enabled = False
        print(f"Gemini initialization failed: {exc}")
        print("Using direct hybrid-search fallback.")

else:
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not configured.")
    else:
        print("LangChain/Gemini packages are unavailable.")
    print("Using direct hybrid-search fallback.")


def _extract_last_agent_text(response):
    """Safely extract the agent's final textual response."""
    try:
        content = response["messages"][-1].content
    except Exception:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()

    return str(content).strip()


def _extract_tool_payload(response):
    """
    Extract the authoritative JSON returned by visual_saree_search_tool.
    This prevents Gemini's wording from becoming the source of truth for
    scores/product fields.
    """
    import json

    try:
        messages = response["messages"]
    except Exception:
        return None

    # Search from the end because the tool result normally appears after
    # the assistant's tool-call message.
    for message in reversed(messages):
        content = getattr(message, "content", None)

        if content is None and isinstance(message, dict):
            content = message.get("content")

        if not isinstance(content, str):
            continue

        try:
            payload = json.loads(content)
        except Exception:
            continue

        if isinstance(payload, dict) and "results" in payload:
            return payload

    return None


def _run_direct_search(image_path, user_query):
    results, excluded_sku = execute_hybrid_search(
        query_image_path=image_path,
        user_text=user_query or "",
        top_k=5,
    )
    return format_results(results, excluded_sku)


def run_tailortalk_agent(image_path, user_query=""):
    """
    Main application pipeline.

    Gemini/LangChain is used when configured to understand the request
    and call the verified visual-search tool.

    The final product cards/scores are ALWAYS formatted from the tool's
    authoritative results, not generated by Gemini.

    If Gemini is unavailable/fails, direct hybrid search is used.
    """
    if not image_path:
        return "## ⚠️ Please upload a saree image first."

    user_query = (user_query or "").strip()

    if not user_query:
        user_query = "Find 5 sarees visually similar to this saree."

    if not gemini_enabled or gemini_agent is None:
        return _run_direct_search(image_path, user_query)

    prompt = f"""
The user has provided a saree image at this local path:

{image_path}

User request:
{user_query}

If the request asks for visually similar sarees, catalogue recommendations,
or recommendations based on this image, call visual_saree_search_tool with
the exact image path above, top_k=5, and the user's request as user_request.

Use the returned catalogue data only. Do not invent or modify any values.
"""

    try:
        response = gemini_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )

        payload = _extract_tool_payload(response)

        if payload is not None:
            # The ranking/product information comes directly from our engine.
            results = payload.get("results", [])
            excluded_sku = payload.get("excluded_query_sku")
            return _format_agent_results(results, excluded_sku)

        # If the model somehow did not call the tool, guarantee the app still
        # returns the verified search results.
        print("Gemini did not call the visual-search tool; using direct search.")
        return _run_direct_search(image_path, user_query)

    except Exception as exc:
        print(f"Gemini agent failed: {exc}")
        print("Using direct hybrid-search fallback.")
        return _run_direct_search(image_path, user_query)


# ============================================================
# 6. IMAGE INPUT HELPERS
# ============================================================

def download_image_from_url(image_url):
    """
    Download an image URL to a temporary local file.

    This supports the assignment requirement of accepting an image link
    in addition to a normal upload.
    """
    import tempfile
    import requests

    image_url = (image_url or "").strip()

    if not image_url:
        return None

    if not re.match(r"^https?://", image_url, flags=re.IGNORECASE):
        raise ValueError("Please enter a valid http:// or https:// image URL.")

    response = requests.get(
        image_url,
        timeout=20,
        headers={"User-Agent": "TailorTalk/1.0"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if content_type and not content_type.startswith("image/"):
        # Some image hosts omit content-type. PIL is the final validation.
        pass

    suffix = ".jpg"
    if "png" in content_type:
        suffix = ".png"
    elif "webp" in content_type:
        suffix = ".webp"
    elif "jpeg" in content_type:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(response.content)
        return temp.name


def gradio_search(image, image_url, user_query):
    """
    Gradio entry point.

    Accept either:
      1. uploaded image, OR
      2. public image URL.

    Upload takes priority when both are supplied.
    """
    temp_download = None

    try:
        if image is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg",
            ) as temp:
                temp_path = temp.name

            image.convert("RGB").save(temp_path, "JPEG")

        elif (image_url or "").strip():
            temp_download = download_image_from_url(image_url)
            temp_path = temp_download

        else:
            return "## ⚠️ Please upload a saree image or provide an image URL."

        return run_tailortalk_agent(
            temp_path,
            user_query or "",
        )

    except Exception as exc:
        return f"## ❌ Search Error\n\n`{str(exc)}`"

    finally:
        # Never delete catalogue images; only delete temporary query files.
        for path in [locals().get("temp_path"), temp_download]:
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


# ============================================================
# 7. GRADIO UI
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.tt-title {
    text-align: center;
}

.tt-results {
    max-height: 700px;
    overflow-y: auto;
}
"""


with gr.Blocks(title="TailorTalk - AI Saree Search") as demo:

    gr.Markdown(
        """
        <div class="tt-title">

        # 👗 TailorTalk

        ### AI-Powered Visual Saree Search

        Upload a saree image **or paste a public image URL** to discover
        visually similar products from the catalogue.

        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="pil",
                label="Upload Saree Image",
                height=520,
            )

            image_url_input = gr.Textbox(
                label="Or paste an image URL",
                placeholder="https://example.com/saree.jpg",
                lines=1,
            )

            message_input = gr.Textbox(
                label="What are you looking for?",
                placeholder=(
                    "e.g. Find 5 sarees similar to this one, "
                    "prefer similar colours"
                ),
                lines=2,
            )

            search_button = gr.Button(
                "🔍 Find Similar Sarees",
                variant="primary",
            )

        with gr.Column(scale=1):

            output = gr.Markdown(
                value=(
                    "### 👗 Similar Sarees\n\n"
                    "Upload an image or paste an image URL and click "
                    "**Find Similar Sarees**."
                ),
                elem_classes="tt-results",
            )

    search_button.click(
        fn=gradio_search,
        inputs=[
            image_input,
            image_url_input,
            message_input,
        ],
        outputs=output,
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    print(f"Starting TailorTalk on port {port}...")
    print(
        "Gemini agent:",
        "ENABLED" if gemini_enabled else "DISABLED (direct fallback)"
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CUSTOM_CSS,
    )
