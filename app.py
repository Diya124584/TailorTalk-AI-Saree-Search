import os
import re
from pathlib import Path

print(">>> TAILORTALK: app.py starting", flush=True)

import cv2
import faiss
import gradio as gr
import numpy as np
import pandas as pd
import torch

print(">>> TAILORTALK: basic imports loaded", flush=True)

from PIL import Image
from skimage import color
from sklearn.cluster import KMeans

print(">>> TAILORTALK: sklearn/skimage loaded", flush=True)

from transformers import AutoImageProcessor, AutoModel

print(">>> TAILORTALK: transformers loaded", flush=True)


# ============================================================
# GEMINI / LANGCHAIN IMPORTS
# ============================================================

try:
    from langchain.tools import tool
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI

    print(">>> TAILORTALK: Gemini/LangChain imports loaded", flush=True)

except ImportError as exc:
    print(f">>> TAILORTALK: Gemini imports unavailable: {exc}", flush=True)

    tool = None
    create_agent = None
    ChatGoogleGenerativeAI = None


# ============================================================
# PATHS
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


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

for required in (
    SAREE_CSV,
    EMBEDDING_METADATA_FILE,
    FAISS_INDEX_FILE,
):
    if not required.exists():
        raise FileNotFoundError(
            f"Required file missing: {required}"
        )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image directory missing: {IMAGE_DIR}"
    )

print(">>> TAILORTALK: required data files found", flush=True)


# ============================================================
# LOAD CSV + METADATA
# ============================================================

saree_df = pd.read_csv(SAREE_CSV)
embedding_metadata = pd.read_csv(
    EMBEDDING_METADATA_FILE
)

print(
    f">>> TAILORTALK: catalogue rows = {len(saree_df)}",
    flush=True
)

print(
    f">>> TAILORTALK: metadata rows = {len(embedding_metadata)}",
    flush=True
)


# ============================================================
# IMAGE FILE MAPPING
# ============================================================

if "image_file" not in saree_df.columns:
    saree_df["image_file"] = (
        saree_df.index.astype(str) + ".jpg"
    )


# ============================================================
# BUILD SEARCH CATALOGUE
# ============================================================

if (
    "image_file" in embedding_metadata.columns
    and "Name" in embedding_metadata.columns
):
    search_catalogue = embedding_metadata.copy()

else:
    search_catalogue = embedding_metadata.merge(
        saree_df,
        on="image_file",
        how="left",
        sort=False,
    )


if "Name" not in search_catalogue.columns:

    possible_name_columns = [
        c
        for c in search_catalogue.columns
        if c.lower()
        in {
            "name",
            "product_name",
            "title",
        }
    ]

    if possible_name_columns:
        search_catalogue["Name"] = (
            search_catalogue[
                possible_name_columns[0]
            ]
        )

    else:
        search_catalogue["Name"] = "Saree"


if "SKU" not in search_catalogue.columns:
    raise KeyError(
        "The catalogue/searchable metadata must contain a SKU column."
    )


if "image_file" not in search_catalogue.columns:
    raise KeyError(
        "The searchable metadata must contain image_file."
    )


for column in (
    "Retail Price",
    "Discounted Price",
    "Stock",
):
    if column not in search_catalogue.columns:
        search_catalogue[column] = 0


if "Website Link" not in search_catalogue.columns:
    search_catalogue["Website Link"] = "#"


print(
    f">>> TAILORTALK: searchable catalogue rows = "
    f"{len(search_catalogue)}",
    flush=True
)


# ============================================================
# COLOUR / FABRIC VOCABULARY
# ============================================================

FABRICS = [
    "semi tussar",
    "tussar",
    "banarasi",
    "organza",
    "tissue",
    "georgette",
    "pashmina",
    "kanchipuram",
    "mysore silk",
    "linen",
    "mul cotton",
    "silk cotton",
    "cotton",
    "chikankari",
    "semi silk",
    "munga",
    "crape",
    "chiffon",
    "satin",
    "silk",
]


COLORS = [
    "navy blue",
    "sky blue",
    "royal blue",
    "aqua blue",
    "powder blue",
    "baby blue",
    "dark green",
    "lime green",
    "mint green",
    "forest green",
    "olive green",
    "sea green",
    "light green",
    "bottle green",
    "dark pink",
    "hot pink",
    "rani pink",
    "baby pink",
    "dusty pink",
    "rose pink",
    "pastel pink",
    "peach pink",
    "blush pink",
    "dark purple",
    "light purple",
    "royal purple",
    "mustard yellow",
    "lemon yellow",
    "pale yellow",
    "off white",
    "off-white",
    "magenta",
    "fuchsia",
    "maroon",
    "burgundy",
    "wine",
    "coral",
    "peach",
    "orange",
    "rust",
    "brown",
    "beige",
    "ivory",
    "cream",
    "white",
    "black",
    "grey",
    "gray",
    "silver",
    "gold",
    "lavender",
    "purple",
    "pink",
    "red",
    "blue",
    "green",
    "yellow",
]


COLOR_FAMILY = {
    "red": "red",
    "maroon": "red",
    "burgundy": "red",
    "wine": "red",
    "rust": "red",

    "pink": "pink",
    "dark pink": "pink",
    "hot pink": "pink",
    "rani pink": "pink",
    "baby pink": "pink",
    "dusty pink": "pink",
    "rose pink": "pink",
    "pastel pink": "pink",
    "peach pink": "pink",
    "blush pink": "pink",
    "magenta": "pink",
    "fuchsia": "pink",

    "purple": "purple",
    "dark purple": "purple",
    "light purple": "purple",
    "royal purple": "purple",
    "lavender": "purple",

    "blue": "blue",
    "navy blue": "blue",
    "sky blue": "blue",
    "royal blue": "blue",
    "aqua blue": "blue",
    "powder blue": "blue",
    "baby blue": "blue",

    "green": "green",
    "dark green": "green",
    "lime green": "green",
    "mint green": "green",
    "forest green": "green",
    "olive green": "green",
    "sea green": "green",
    "light green": "green",
    "bottle green": "green",

    "yellow": "yellow",
    "mustard yellow": "yellow",
    "lemon yellow": "yellow",
    "pale yellow": "yellow",
    "gold": "yellow",

    "orange": "orange",
    "peach": "orange",
    "coral": "orange",

    "brown": "brown",
    "beige": "brown",

    "cream": "neutral",
    "ivory": "neutral",
    "white": "neutral",
    "off white": "neutral",
    "off-white": "neutral",

    "black": "black",
    "grey": "grey",
    "gray": "grey",
    "silver": "grey",
}


# ============================================================
# ATTRIBUTE EXTRACTION
# ============================================================

def _contains_phrase(text, phrase):
    return (
        re.search(
            r"(?<!\w)"
            + re.escape(phrase)
            + r"(?!\w)",
            text,
        )
        is not None
    )


def extract_attributes(text):

    text = str(text or "").lower()

    found_colors = {
        c
        for c in COLORS
        if _contains_phrase(text, c)
    }

    found_fabrics = {
        f
        for f in FABRICS
        if _contains_phrase(text, f)
    }

    return {
        "colors": found_colors,
        "fabrics": found_fabrics,
        "raw_tokens": set(
            re.findall(r"\b\w+\b", text)
        ),
    }


def row_text_for_attributes(row):

    fields = []

    useful_columns = {
        "name",
        "product_name",
        "title",
        "colour",
        "color",
        "description",
        "material",
        "fabric",
        "category",
        "type",
    }

    for column in row.index:

        if column.lower() in useful_columns:

            value = row.get(column)

            if pd.notna(value):
                fields.append(str(value))

    return " ".join(fields)


search_catalogue["attr"] = search_catalogue.apply(
    lambda row: extract_attributes(
        row_text_for_attributes(row)
    ),
    axis=1,
)


# ============================================================
# PALETTE EXTRACTION
# ============================================================

PALETTE_CACHE = {}


def extract_saree_foreground_pixels(pil_img):

    rgb = np.array(
        pil_img.convert("RGB")
    )

    small = cv2.resize(
        rgb,
        (256, 256),
        interpolation=cv2.INTER_AREA,
    )

    lab = color.rgb2lab(
        small.astype(np.float32) / 255.0
    )

    border = np.concatenate(
        [
            lab[:18].reshape(-1, 3),
            lab[-18:].reshape(-1, 3),
            lab[:, :18].reshape(-1, 3),
            lab[:, -18:].reshape(-1, 3),
        ],
        axis=0,
    )

    bg_lab = np.median(
        border,
        axis=0,
    )

    bg_dist = np.linalg.norm(
        lab - bg_lab,
        axis=2,
    )

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2HSV,
    )

    saturation = (
        hsv[..., 1].astype(np.float32)
        / 255.0
    )

    yy, xx = np.mgrid[
        0:256,
        0:256,
    ]

    center_x = (
        1.0
        - np.abs(xx - 128.0)
        / 128.0
    )

    vertical = np.clip(
        (yy - 12.0) / 210.0,
        0.0,
        1.0,
    )

    spatial = (
        0.45 * center_x
        + 0.55 * vertical
    )

    score = (
        0.78 * bg_dist
        + 20.0 * saturation
        + 7.0 * spatial
    )

    valid = np.zeros(
        (256, 256),
        dtype=bool,
    )

    valid[
        8:250,
        18:238
    ] = True

    score[~valid] = -np.inf

    flat = score[valid]
    finite = flat[
        np.isfinite(flat)
    ]

    if finite.size:

        threshold = np.percentile(
            finite,
            62,
        )

        mask = (
            valid
            & (score >= threshold)
        )

    else:
        mask = valid

    mask_u8 = (
        mask.astype(np.uint8)
        * 255
    )

    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel,
    )

    pixels = small[
        mask_u8 > 0
    ]

    if len(pixels) < 800:

        crop = small[
            25:235,
            30:226
        ]

        pixels = crop.reshape(
            -1,
            3,
        )

    if len(pixels) > 4000:
        rng = np.random.default_rng(42)
        take = rng.choice(len(pixels), 4000, replace=False)
        pixels = pixels[take]

    return pixels


def extract_color_palette(
    fg_pixels,
    k=5,
):

    lab_pixels = color.rgb2lab(
        fg_pixels.astype(
            np.float32
        ) / 255.0
    )

    n_clusters = min(
        k,
        len(lab_pixels),
    )

    if n_clusters < 1:

        return (
            np.zeros(
                (1, 3),
                dtype=np.float32,
            ),
            np.ones(
                1,
                dtype=np.float32,
            ),
        )

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=1,
        random_state=42,
    )

    labels = kmeans.fit_predict(
        lab_pixels
    )

    centers = (
        kmeans.cluster_centers_
        .astype(np.float32)
    )

    counts = np.bincount(
        labels,
        minlength=n_clusters,
    ).astype(np.float32)

    weights = counts / max(
        counts.sum(),
        1.0,
    )

    order = np.argsort(
        weights
    )[::-1]

    centers = centers[order]
    weights = weights[order]

    return centers, weights


def get_palette(image):

    pixels = (
        extract_saree_foreground_pixels(
            image
        )
    )

    return extract_color_palette(
        pixels,
        k=5,
    )


def palette_similarity(
    query_palette,
    query_weights,
    cand_palette,
    cand_weights,
):

    q = np.asarray(
        query_palette,
        dtype=np.float32,
    )

    c = np.asarray(
        cand_palette,
        dtype=np.float32,
    )

    qw = np.asarray(
        query_weights,
        dtype=np.float32,
    )

    cw = np.asarray(
        cand_weights,
        dtype=np.float32,
    )

    distances = np.linalg.norm(
        q[:, None, :]
        - c[None, :, :],
        axis=2,
    )

    pair_sim = np.exp(
        -distances / 38.0
    )

    q_to_c = float(
        np.sum(
            qw
            * np.max(
                pair_sim,
                axis=1,
            )
        )
    )

    c_to_q = float(
        np.sum(
            cw
            * np.max(
                pair_sim,
                axis=0,
            )
        )
    )

    symmetric = (
        0.5
        * (q_to_c + c_to_q)
    )

    dominant = float(
        np.max(
            pair_sim[0]
        )
    )

    score = (
        0.65 * dominant
        + 0.35 * symmetric
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def get_catalogue_palette(
    idx,
    image_file,
):

    if idx in PALETTE_CACHE:
        return PALETTE_CACHE[idx]

    image_path = (
        IMAGE_DIR
        / str(image_file)
    )

    if image_path.exists():

        try:

            with Image.open(
                image_path
            ) as img:

                result = get_palette(
                    img
                )

            PALETTE_CACHE[idx] = result

            return result

        except Exception:
            pass

    fallback = (
        np.array(
            [[50.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        np.array(
            [1.0],
            dtype=np.float32,
        ),
    )

    PALETTE_CACHE[idx] = fallback

    return fallback


# ============================================================
# CSV COLOUR SUPPORT
# ============================================================

def colour_metadata_similarity(
    query_colors,
    candidate_colors,
):

    if (
        not query_colors
        or not candidate_colors
    ):
        return 0.50

    best = 0.0

    for q in query_colors:

        q_family = COLOR_FAMILY.get(
            q,
            q,
        )

        for c in candidate_colors:

            c_family = COLOR_FAMILY.get(
                c,
                c,
            )

            if q == c:

                best = max(
                    best,
                    1.0,
                )

            elif q_family == c_family:

                best = max(
                    best,
                    0.82,
                )

            elif {
                q_family,
                c_family,
            } <= {
                "pink",
                "red",
                "purple",
                "orange",
            }:

                best = max(
                    best,
                    0.55,
                )

            else:

                best = max(
                    best,
                    0.12,
                )

    return float(
        np.clip(
            best,
            0.0,
            1.0,
        )
    )


# ============================================================
# DINOv2 + FAISS
# ============================================================

MODEL_NAME = (
    "facebook/dinov2-small"
)

device = torch.device("cpu")

image_processor = None
visual_model = None
faiss_index = None


def load_resources():

    global image_processor
    global visual_model
    global faiss_index

    if image_processor is None:

        print(
            ">>> TAILORTALK: loading DINOv2 processor...",
            flush=True,
        )

        image_processor = (
            AutoImageProcessor
            .from_pretrained(
                MODEL_NAME
            )
        )

        print(
            ">>> TAILORTALK: loading DINOv2 model...",
            flush=True,
        )

        visual_model = (
            AutoModel
            .from_pretrained(
                MODEL_NAME
            )
            .to(device)
        )

        visual_model.eval()

        print(
            ">>> TAILORTALK: DINOv2 loaded",
            flush=True,
        )

    if faiss_index is None:

        print(
            ">>> TAILORTALK: loading FAISS index...",
            flush=True,
        )

        faiss_index = faiss.read_index(
            str(FAISS_INDEX_FILE)
        )

        print(
            f">>> TAILORTALK: FAISS loaded "
            f"({faiss_index.ntotal} vectors)",
            flush=True,
        )


def get_dino_embedding(image):

    load_resources()

    inputs = image_processor(
        images=image.convert("RGB"),
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = visual_model(
            **inputs
        )

    embedding = (
        outputs.last_hidden_state[:, 0]
        .cpu()
        .numpy()
        .astype("float32")[0]
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm > 1e-12:
        embedding /= norm

    return embedding


# ============================================================
# IDENTIFY QUERY CATALOGUE PRODUCT
# ============================================================

def get_query_catalogue_row(
    query_image_path,
    faiss_indices,
    faiss_scores,
):

    query_name = Path(
        query_image_path
    ).name

    for score, idx in zip(
        faiss_scores,
        faiss_indices,
    ):

        if (
            idx < 0
            or idx >= len(
                search_catalogue
            )
        ):
            continue

        row = search_catalogue.iloc[
            int(idx)
        ]

        if (
            str(row["image_file"])
            == query_name
        ):
            return (
                int(idx),
                str(row["SKU"]),
            )

    for score, idx in zip(
        faiss_scores,
        faiss_indices,
    ):

        if (
            idx < 0
            or idx >= len(
                search_catalogue
            )
        ):
            continue

        if float(score) >= 0.999:

            row = search_catalogue.iloc[
                int(idx)
            ]

            return (
                int(idx),
                str(row["SKU"]),
            )

    return -1, None


# ============================================================
# HYBRID SEARCH
# ============================================================

def execute_hybrid_search(
    query_image_path,
    user_text="",
    top_k=5,
):

    load_resources()

    if (
        not query_image_path
        or not Path(
            query_image_path
        ).exists()
    ):

        raise FileNotFoundError(
            f"Image not found: "
            f"{query_image_path}"
        )

    query_img = Image.open(
        query_image_path
    ).convert("RGB")

    query_emb = get_dino_embedding(
        query_img
    ).reshape(1, -1)

    faiss.normalize_L2(
        query_emb
    )

    query_palette, query_weights = (
        get_palette(query_img)
    )

    query_text_attr = (
        extract_attributes(
            user_text
        )
    )

    search_k = min(
        8,
        faiss_index.ntotal,
    )

    raw_scores, indices = (
        faiss_index.search(
            query_emb,
            search_k,
        )
    )

    excluded_idx, excluded_sku = (
        get_query_catalogue_row(
            query_image_path,
            indices[0],
            raw_scores[0],
        )
    )

    query_catalogue_attr = None

    if excluded_idx >= 0:

        query_catalogue_attr = (
            search_catalogue
            .iloc[
                excluded_idx
            ]["attr"]
        )

    query_colors = set(
        query_text_attr["colors"]
    )

    if query_catalogue_attr is not None:

        query_colors.update(
            query_catalogue_attr[
                "colors"
            ]
        )

    candidates = []

    for dino_raw, idx in zip(
        raw_scores[0],
        indices[0],
    ):

        idx = int(idx)

        if (
            idx < 0
            or idx >= len(
                search_catalogue
            )
        ):
            continue

        row = search_catalogue.iloc[
            idx
        ]

        sku = str(
            row["SKU"]
        )

        if (
            excluded_sku
            and sku == excluded_sku
        ):
            continue

        if (
            excluded_idx >= 0
            and idx == excluded_idx
        ):
            continue

        try:

            stock = int(
                float(
                    row.get(
                        "Stock",
                        0,
                    )
                )
            )

        except Exception:

            stock = 0

        if stock <= 0:
            continue

        dino_sim = float(
            np.clip(
                dino_raw,
                0.0,
                1.0,
            )
        )

        cand_palette, cand_weights = (
            get_catalogue_palette(
                idx,
                row["image_file"],
            )
        )

        image_color_sim = (
            palette_similarity(
                query_palette,
                query_weights,
                cand_palette,
                cand_weights,
            )
        )

        metadata_color_sim = (
            colour_metadata_similarity(
                query_colors,
                row["attr"]["colors"],
            )
        )

        # Visual appearance remains the main signal.
        # Colour/shade is the second strongest signal.
        final_score = (
            0.70 * dino_sim
            + 0.25 * image_color_sim
            + 0.05 * metadata_color_sim
        )

        candidates.append(
            {
                "image_file": str(
                    row["image_file"]
                ),
                "name": str(
                    row["Name"]
                ),
                "sku": sku,
                "dino_score": dino_sim,
                "color_score": float(
                    image_color_sim
                ),
                "metadata_color_score": float(
                    metadata_color_sim
                ),
                "final_score": float(
                    np.clip(
                        final_score,
                        0.0,
                        1.0,
                    )
                ),
                "retail_price": float(
                    row.get(
                        "Retail Price",
                        0.0,
                    )
                ),
                "discounted_price": float(
                    row.get(
                        "Discounted Price",
                        0.0,
                    )
                ),
                "stock": stock,
                "website_link": str(
                    row.get(
                        "Website Link",
                        "#",
                    )
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["final_score"],
            item["dino_score"],
            item["color_score"],
        ),
        reverse=True,
    )

    results = []
    seen_skus = set()

    for item in candidates:

        if item["sku"] in seen_skus:
            continue

        seen_skus.add(
            item["sku"]
        )

        results.append(item)

        if len(results) >= top_k:
            break

    return (
        results,
        excluded_sku,
    )


# ============================================================
# FORMAT RESULTS
# ============================================================

def format_results(
    results,
    excluded_sku,
):

    if not results:

        return (
            "### No matching sarees found.\n\n"
            "Try another saree image or check "
            "that the catalogue images exist."
        )

    output = []

    if excluded_sku:

        output.append(
            f"> 🛡️ **Excluded Query Product SKU:** "
            f"`{excluded_sku}`\n"
        )

    output.append(
        "## 👗 Recommended Sarees\n"
    )

    output.append(
        "_Ranked using overall visual appearance, "
        "dominant colours/shades, and CSV colour metadata._\n"
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        output.append(
            f"### {i}. {result['name']}\n\n"
            f"- **Match Score:** "
            f"`{result['final_score'] * 100:.1f}%`\n"
            f"  - Overall Visual (DINOv2): "
            f"`{result['dino_score'] * 100:.1f}%`\n"
            f"  - Colour & Shade Similarity: "
            f"`{result['color_score'] * 100:.1f}%`\n"
            f"  - CSV Colour Match: "
            f"`{result['metadata_color_score'] * 100:.1f}%`\n"
            f"- **SKU:** `{result['sku']}`\n"
            f"- **Retail Price:** "
            f"₹{result['retail_price']:,.2f}\n"
            f"- **Discounted Price:** "
            f"₹{result['discounted_price']:,.2f}\n"
            f"- **Stock:** "
            f"{result['stock']} units\n"
            f"- **Product Link:** "
            f"[View Product]({result['website_link']})\n\n"
            "---"
        )

    return "\n".join(output)


# ============================================================
# GEMINI AGENT
# ============================================================

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "",
).strip()

gemini_agent = None
gemini_enabled = False


if (
    tool is not None
    and create_agent is not None
    and ChatGoogleGenerativeAI is not None
    and GEMINI_API_KEY
):

    @tool
    def visual_saree_search_tool(
        image_path: str,
        top_k: int = 5,
        user_request: str = "",
    ) -> str:

        """
        Search TailorTalk's saree catalogue using
        DINOv2 + FAISS + colour/shade reranking.
        """

        try:

            top_k = max(
                1,
                min(
                    int(top_k),
                    5,
                ),
            )

        except Exception:

            top_k = 5

        results, excluded_sku = (
            execute_hybrid_search(
                query_image_path=image_path,
                user_text=user_request or "",
                top_k=top_k,
            )
        )

        import json

        return json.dumps(
            {
                "excluded_query_sku": (
                    excluded_sku
                ),
                "results": results,
            },
            ensure_ascii=False,
            default=str,
        )


    try:

        print(
            ">>> TAILORTALK: initializing Gemini...",
            flush=True,
        )

        gemini_llm = (
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0,
            )
        )

        gemini_agent = create_agent(
            model=gemini_llm,
            tools=[
                visual_saree_search_tool
            ],
            system_prompt="""
You are TailorTalk, an AI saree shopping assistant.

When the user requests visual saree recommendations,
ALWAYS call visual_saree_search_tool.

The tool is the authoritative source for:
- product names
- SKUs
- prices
- stock
- links
- similarity scores

Never invent, change, or recalculate those values.

The ranking engine combines:
DINOv2 visual similarity,
image colour/shade similarity,
and CSV colour metadata.

Return the tool results clearly and concisely.
""",
        )

        gemini_enabled = True

        print(
            ">>> TAILORTALK: Gemini initialized successfully",
            flush=True,
        )

    except Exception as exc:

        gemini_agent = None
        gemini_enabled = False

        print(
            f">>> TAILORTALK: Gemini initialization failed: {exc}",
            flush=True,
        )

        print(
            ">>> TAILORTALK: direct hybrid-search fallback enabled",
            flush=True,
        )

else:

    if not GEMINI_API_KEY:

        print(
            ">>> TAILORTALK: GEMINI_API_KEY not configured",
            flush=True,
        )

    print(
        ">>> TAILORTALK: direct hybrid-search fallback available",
        flush=True,
    )


# ============================================================
# GEMINI RESPONSE HELPERS
# ============================================================

def _extract_tool_payload(response):

    import json

    try:
        messages = response["messages"]

    except Exception:
        return None

    for message in reversed(messages):

        content = getattr(
            message,
            "content",
            None,
        )

        if (
            content is None
            and isinstance(
                message,
                dict,
            )
        ):
            content = message.get(
                "content"
            )

        if not isinstance(
            content,
            str,
        ):
            continue

        try:

            payload = json.loads(
                content
            )

        except Exception:
            continue

        if (
            isinstance(
                payload,
                dict,
            )
            and "results" in payload
        ):
            return payload

    return None


def _run_direct_search(
    image_path,
    user_query,
):

    results, excluded_sku = (
        execute_hybrid_search(
            query_image_path=image_path,
            user_text=user_query or "",
            top_k=5,
        )
    )

    return format_results(
        results,
        excluded_sku,
    )


def run_tailortalk_agent(
    image_path,
    user_query="",
):

    if not image_path:

        return (
            "## ⚠️ Please upload a saree image first."
        )

    user_query = (
        user_query or ""
    ).strip()

    if not user_query:

        user_query = (
            "Find 5 sarees visually "
            "similar to this saree."
        )

    if (
        not gemini_enabled
        or gemini_agent is None
    ):

        return _run_direct_search(
            image_path,
            user_query,
        )

    prompt = f"""
The user provided a saree image at:

{image_path}

User request:

{user_query}

Use visual_saree_search_tool with:
- image_path = exactly the path above
- top_k = 5
- user_request = the user's request

Use only the returned catalogue data.
Do not invent or modify any product information or scores.
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

        payload = _extract_tool_payload(
            response
        )

        if payload is not None:

            return format_results(
                payload.get(
                    "results",
                    [],
                ),
                payload.get(
                    "excluded_query_sku"
                ),
            )

        print(
            ">>> TAILORTALK: Gemini did not call tool; using direct search",
            flush=True,
        )

        return _run_direct_search(
            image_path,
            user_query,
        )

    except Exception as exc:

        print(
            f">>> TAILORTALK: Gemini request failed: {exc}",
            flush=True,
        )

        return _run_direct_search(
            image_path,
            user_query,
        )


# ============================================================
# IMAGE URL SUPPORT
# ============================================================

def download_image_from_url(
    image_url,
):

    import tempfile
    import requests

    image_url = (
        image_url or ""
    ).strip()

    if not image_url:
        return None

    if not re.match(
        r"^https?://",
        image_url,
        flags=re.IGNORECASE,
    ):

        raise ValueError(
            "Please enter a valid "
            "http:// or https:// image URL."
        )

    response = requests.get(
        image_url,
        timeout=20,
        headers={
            "User-Agent":
            "TailorTalk/1.0"
        },
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get(
            "content-type",
            "",
        )
        .lower()
    )

    suffix = ".jpg"

    if "png" in content_type:
        suffix = ".png"

    elif "webp" in content_type:
        suffix = ".webp"

    elif "jpeg" in content_type:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:

        temp.write(
            response.content
        )

        return temp.name


# ============================================================
# GRADIO SEARCH
# ============================================================

def gradio_search(
    image,
    image_url,
    user_query,
):

    temp_download = None
    temp_path = None

    try:

        if image is not None:

            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg",
            ) as temp:

                temp_path = temp.name

            image.convert(
                "RGB"
            ).save(
                temp_path,
                "JPEG",
            )

        elif (
            image_url or ""
        ).strip():

            temp_download = (
                download_image_from_url(
                    image_url
                )
            )

            temp_path = (
                temp_download
            )

        else:

            return (
                "## ⚠️ Please upload a saree image "
                "or provide an image URL."
            )

        return run_tailortalk_agent(
            temp_path,
            user_query or "",
        )

    except Exception as exc:

        print(
            f">>> TAILORTALK: search error: {exc}",
            flush=True,
        )

        return (
            "## ❌ Search Error\n\n"
            f"`{str(exc)}`"
        )

    finally:

        for path in [
            temp_path,
            temp_download,
        ]:

            if (
                path
                and os.path.isfile(path)
            ):

                try:
                    os.remove(path)

                except Exception:
                    pass


# ============================================================
# GRADIO UI
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


print(
    ">>> TAILORTALK: building Gradio UI...",
    flush=True,
)


with gr.Blocks(
    title="TailorTalk - AI Saree Search",
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(
        """
<div class="tt-title">

# 👗 TailorTalk

### AI-Powered Visual Saree Search

Upload a saree image **or paste a public image URL**
to discover visually similar products from the catalogue.

</div>
"""
    )

    with gr.Row():

        with gr.Column(
            scale=1
        ):

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
                    "e.g. Find 5 sarees similar "
                    "to this one, prefer similar colours"
                ),
                lines=2,
            )

            search_button = gr.Button(
                "🔍 Find Similar Sarees",
                variant="primary",
            )

        with gr.Column(
            scale=1
        ):

            output = gr.Markdown(
                value=(
                    "### 👗 Similar Sarees\n\n"
                    "Upload an image or paste an image URL "
                    "and click **Find Similar Sarees**."
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


print(
    ">>> TAILORTALK: Gradio UI created",
    flush=True,
)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        ">>> TAILORTALK: reached main()",
        flush=True,
    )

    port = int(
        os.environ.get(
            "PORT",
            7860,
        )
    )

    print(
        f">>> TAILORTALK: starting Gradio "
        f"on port {port}",
        flush=True,
    )

    print(
        ">>> TAILORTALK: Gemini agent:",
        "ENABLED"
        if gemini_enabled
        else "DISABLED",
        flush=True,
    )

    # --------------------------------------------------------
    # PRELOAD DINOv2 + FAISS
    # This prevents the first user's search from waiting
    # for the Hugging Face model to download/load.
    # --------------------------------------------------------

    print(
        ">>> TAILORTALK: preloading DINOv2 + FAISS...",
        flush=True,
    )

    try:
        load_resources()

        print(
            ">>> TAILORTALK: DINOv2 + FAISS preloaded successfully",
            flush=True,
        )

    except Exception as exc:

        print(
            f">>> TAILORTALK: preload failed: {exc}",
            flush=True,
        )

        print(
            ">>> TAILORTALK: resources will be loaded on first search",
            flush=True,
        )

    print(
        ">>> TAILORTALK: launching Gradio...",
        flush=True,
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )