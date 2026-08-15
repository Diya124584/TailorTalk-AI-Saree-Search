# ============================================================
# TAILORTALK - DEPLOYMENT APPLICATION
# AI-POWERED VISUAL SAREE SEARCH
# ============================================================

import os
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"

SAREE_CSV = DATA_DIR / "saree.csv"
EMBEDDING_METADATA_FILE = DATA_DIR / "embedding_metadata.csv"
FAISS_INDEX_FILE = DATA_DIR / "saree_dinov2_faiss.index"


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

required_files = [
    SAREE_CSV,
    EMBEDDING_METADATA_FILE,
    FAISS_INDEX_FILE,
]

missing_files = [
    str(file)
    for file in required_files
    if not file.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Required TailorTalk files are missing:\n"
        + "\n".join(missing_files)
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found: {IMAGE_DIR}"
    )


# ============================================================
# 3. LOAD CATALOGUE
# ============================================================

print("Loading saree catalogue...")

saree_df = pd.read_csv(
    SAREE_CSV
)

print(
    "Catalogue rows:",
    len(saree_df)
)


# ============================================================
# 4. CREATE IMAGE -> CATALOGUE MAPPING
# ============================================================

saree_df["image_file"] = (
    saree_df.index.astype(str) + ".jpg"
)


# ============================================================
# 5. LOAD EMBEDDING METADATA
# ============================================================

print(
    "Loading embedding metadata..."
)

embedding_metadata = pd.read_csv(
    EMBEDDING_METADATA_FILE
)


# ============================================================
# 6. BUILD SEARCHABLE CATALOGUE
# ============================================================

search_catalogue = embedding_metadata.merge(
    saree_df,
    on="image_file",
    how="left",
    sort=False
)


# ============================================================
# 7. VALIDATE MAPPING
# ============================================================

print(
    "Embedding metadata rows:",
    len(embedding_metadata)
)

print(
    "Searchable catalogue rows:",
    len(search_catalogue)
)


assert (
    len(search_catalogue)
    == len(embedding_metadata)
), (
    "Catalogue and embedding metadata are misaligned!"
)


assert (
    search_catalogue["image_file"].tolist()
    == embedding_metadata["image_file"].tolist()
), (
    "Image order and embedding order are misaligned!"
)


print(
    "Catalogue metadata/image mapping is aligned!"
)


# ============================================================
# 8. DINOv2 CONFIGURATION
#
# IMPORTANT:
# DINOv2 is loaded lazily.
# This prevents Render from waiting for the model
# before the web server opens its port.
# ============================================================

MODEL_NAME = "facebook/dinov2-small"

# Render Free does not provide a GPU.
# Force CPU for deployment.
device = torch.device("cpu")

image_processor = None
visual_model = None


def load_dinov2():

    global image_processor
    global visual_model

    # Already loaded
    if (
        image_processor is not None
        and visual_model is not None
    ):
        return

    print(
        "Loading DINOv2..."
    )

    image_processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    visual_model = (
        AutoModel.from_pretrained(
            MODEL_NAME
        )
    )

    visual_model = visual_model.to(
        device
    )

    visual_model.eval()

    print(
        "DINOv2 loaded successfully!"
    )


# ============================================================
# 9. DINOv2 EMBEDDING FUNCTIONS
# ============================================================

def normalize_vector(vector):

    vector = np.asarray(
        vector,
        dtype="float32"
    )

    norm = np.linalg.norm(
        vector
    )

    if norm < 1e-12:
        return vector

    return vector / norm


def get_dino_embedding(image):

    # Load DINOv2 only when actually needed
    load_dinov2()

    inputs = image_processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = visual_model(
            **inputs
        )

    # CLS token
    embedding = (
        outputs.last_hidden_state[:, 0]
    )

    embedding = (
        embedding
        .cpu()
        .numpy()
        .astype("float32")
    )

    embedding = embedding[0]

    return normalize_vector(
        embedding
    )


# ============================================================
# 10. FAISS CONFIGURATION
#
# FAISS is also loaded lazily.
# ============================================================

index = None


def load_faiss():

    global index

    if index is not None:
        return

    print(
        "Loading DINOv2 FAISS index..."
    )

    index = faiss.read_index(
        str(FAISS_INDEX_FILE)
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    print(
        "FAISS dimension:",
        index.d
    )

    # --------------------------------------------------------
    # Validate FAISS + catalogue
    # --------------------------------------------------------

    assert (
        index.ntotal
        == len(search_catalogue)
    ), (
        "FAISS index and searchable catalogue "
        "are misaligned!\n"
        f"FAISS vectors: {index.ntotal}\n"
        f"Catalogue rows: {len(search_catalogue)}"
    )

    assert index.d == 384, (
        f"Expected DINOv2 dimension 384, "
        f"got {index.d}"
    )

    print(
        "FAISS index and catalogue are aligned!"
    )


# ============================================================
# 11. LANGCHAIN VISUAL SEARCH TOOL
# ============================================================

@tool
def search_similar_sarees(
    image_path: str,
    top_k: int = 5
) -> list:
    """
    Find visually similar sarees from the catalogue.

    The FAISS index uses normalized RGB DINOv2 embeddings.

    Returns unique in-stock products.
    """

    # Load resources only when searching
    load_faiss()
    load_dinov2()

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    try:

        top_k = int(
            top_k
        )

    except Exception:

        top_k = 5

    top_k = max(
        1,
        min(
            top_k,
            10
        )
    )

    # --------------------------------------------------------
    # Validate image path
    # --------------------------------------------------------

    if not image_path:

        raise ValueError(
            "No image path was provided."
        )

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Load query image
    # --------------------------------------------------------

    query_image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The FAISS index was created using RGB DINOv2 embeddings.
    # Therefore the query MUST remain RGB as well.
    # DO NOT convert the query to grayscale.
    # --------------------------------------------------------

    query_embedding = get_dino_embedding(
        query_image
    )

    # --------------------------------------------------------
    # FAISS shape
    # --------------------------------------------------------

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    ).reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Normalize
    #
    # IndexFlatIP + normalized vectors
    # = cosine similarity
    # --------------------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )

    # --------------------------------------------------------
    # Search extra candidates
    #
    # Extra candidates are required because:
    #
    # - some products may be out of stock
    # - duplicate SKUs may exist
    # --------------------------------------------------------

    search_k = min(
        max(
            top_k * 10,
            50
        ),
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    # ========================================================
    # BUILD RESULTS
    # ========================================================

    results = []

    seen_skus = set()

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(search_catalogue):
            continue

        # ----------------------------------------------------
        # Get catalogue row
        # ----------------------------------------------------

        row = search_catalogue.iloc[
            int(idx)
        ]

        # ----------------------------------------------------
        # SKU
        # ----------------------------------------------------

        sku = str(
            row["SKU"]
        )

        # ----------------------------------------------------
        # Stock
        # ----------------------------------------------------

        try:

            stock = int(
                float(
                    row["Stock"]
                )
            )

        except Exception:

            stock = 0

        # ----------------------------------------------------
        # Only return in-stock products
        # ----------------------------------------------------

        if stock <= 0:
            continue

        # ----------------------------------------------------
        # Remove duplicate SKUs
        # ----------------------------------------------------

        if sku in seen_skus:
            continue

        seen_skus.add(
            sku
        )

        # ----------------------------------------------------
        # Retail price
        # ----------------------------------------------------

        try:

            retail_price = float(
                row["Retail Price"]
            )

        except Exception:

            retail_price = 0.0

        # ----------------------------------------------------
        # Discounted price
        # ----------------------------------------------------

        try:

            discounted_price = float(
                row["Discounted Price"]
            )

        except Exception:

            discounted_price = 0.0

        # ----------------------------------------------------
        # Website link
        # ----------------------------------------------------

        website_link = str(
            row["Website Link"]
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "image_file": str(
                row["image_file"]
            ),

            "name": str(
                row["Name"]
            ),

            "sku": sku,

            "similarity_score": float(
                score
            ),

            "retail_price": retail_price,

            "discounted_price": discounted_price,

            "stock": stock,

            "website_link": website_link
        })

        if len(results) >= top_k:
            break

    return results


print(
    "Visual search tool created successfully!"
)


# ============================================================
# 12. FORMAT SEARCH RESULTS
# ============================================================

def format_results(results):

    if not results:

        return (
            "### No similar sarees found.\n\n"
            "Try another image."
        )

    output = [
        "## 👗 TailorTalk Search Results\n"
    ]

    for i, result in enumerate(
        results,
        start=1
    ):

        output.append(
            f"""
### {i}. {result["name"]}

- **Similarity:** `{result["similarity_score"]:.4f}`
- **SKU:** `{result["sku"]}`
- **Retail Price:** ₹{result["retail_price"]:,.2f}
- **Discounted Price:** ₹{result["discounted_price"]:,.2f}
- **Stock:** {result["stock"]} units
- **Product Link:** [View Product]({result["website_link"]})

---
"""
        )

    return "\n".join(
        output
    )


# ============================================================
# 13. LAZY GEMINI AGENT
#
# Gemini is initialized only when a user performs a search.
# ============================================================

agent = None
gemini_initialized = False


def get_gemini_agent():

    global agent
    global gemini_initialized

    # Don't initialize repeatedly
    if gemini_initialized:

        return agent

    gemini_initialized = True

    # --------------------------------------------------------
    # Read API key from environment
    # --------------------------------------------------------

    gemini_api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not gemini_api_key:

        print(
            "GEMINI_API_KEY not configured."
        )

        print(
            "Using direct visual search fallback."
        )

        return None

    try:

        print(
            "Initializing Gemini agent..."
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=gemini_api_key
        )

        agent = create_agent(
            model=llm,
            tools=[
                search_similar_sarees
            ],
            system_prompt="""
You are TailorTalk, an AI fashion assistant that helps users
find visually similar sarees from a saree catalogue.

When the user asks to:

- find similar sarees
- search for sarees similar to an image
- recommend sarees based on an uploaded image
- find visually similar clothing

use the search_similar_sarees tool.

The tool requires:

- image_path
- top_k

When an uploaded image path is provided, use that exact path.

When displaying search results, clearly show:

1. Saree name
2. Similarity score
3. SKU
4. Retail price
5. Discounted price
6. Stock
7. Product link

Rules:

- Preserve the similarity scores returned by the tool.
- Never invent or modify similarity scores.
- Never invent product information.
- Do not repeat the exact same SKU.
- Only describe products returned by the search tool.
- Be concise, friendly and helpful.

If the user asks a general fashion question and does
not request visual similarity search, answer normally
without calling the visual search tool.
"""
        )

        print(
            "TailorTalk Gemini agent initialized!"
        )

        return agent

    except Exception as e:

        print(
            "Gemini initialization failed:"
        )

        print(
            str(e)
        )

        agent = None

        return None


# ============================================================
# 14. EXTRACT AGENT RESPONSE
# ============================================================

def extract_agent_text(response):

    try:

        content = (
            response["messages"][-1].content
        )

        if isinstance(
            content,
            list
        ):

            text_parts = []

            for block in content:

                if isinstance(
                    block,
                    dict
                ):

                    if block.get("type") == "text":

                        text_parts.append(
                            block.get(
                                "text",
                                ""
                            )
                        )

                elif isinstance(
                    block,
                    str
                ):

                    text_parts.append(
                        block
                    )

            return "\n".join(
                text_parts
            )

        return str(
            content
        )

    except Exception as e:

        return (
            f"Could not read agent response: {e}"
        )


# ============================================================
# 15. MAIN TAILORTALK SEARCH
# ============================================================

def tailor_talk_search(
    image_path,
    user_message
):

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not image_path:

        return (
            "## ⚠️ Please upload a saree image first."
        )

    # --------------------------------------------------------
    # Clean user message
    # --------------------------------------------------------

    if not user_message:

        user_message = ""

    user_message = user_message.strip()

    if not user_message:

        user_message = (
            "Find sarees visually similar "
            "to this image."
        )

    # ========================================================
    # GEMINI AGENT
    # ========================================================

    current_agent = (
        get_gemini_agent()
    )

    if current_agent is not None:

        prompt = f"""
The user uploaded a saree image.

Image path:
{image_path}

User request:
{user_message}

If the user is asking for visually similar sarees,
use the search_similar_sarees tool.

Use the uploaded image path exactly as the image_path
argument.

Return the catalogue results clearly.
"""

        try:

            response = current_agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

            text = extract_agent_text(
                response
            )

            if (
                text
                and text.strip()
            ):

                return text

        except Exception as e:

            print(
                "Gemini agent failed:"
            )

            print(
                str(e)
            )

            print(
                "Falling back to direct FAISS search."
            )

    # ========================================================
    # DIRECT FAISS FALLBACK
    # ========================================================

    try:

        results = search_similar_sarees.invoke({
            "image_path": str(image_path),
            "top_k": 5
        })

        return format_results(
            results
        )

    except Exception as e:

        print(
            "Search failed:"
        )

        print(
            str(e)
        )

        return (
            "## ❌ Search Error\n\n"
            f"`{str(e)}`"
        )


# ============================================================
# 16. GRADIO FRONTEND
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
    max-height: 650px;
    overflow-y: auto;
}
"""


with gr.Blocks(
    title="TailorTalk - AI Saree Search",
    css=CUSTOM_CSS
) as demo:

    gr.Markdown(
        """
        <div class="tt-title">

        # 👗 TailorTalk

        ### AI-Powered Visual Saree Search

        Upload a saree image and discover visually
        similar products from the catalogue.

        </div>
        """
    )

    with gr.Row():

        # ----------------------------------------------------
        # LEFT SIDE
        # ----------------------------------------------------

        with gr.Column(
            scale=1
        ):

            image_input = gr.Image(
                type="filepath",
                label="Upload Saree Image",
                height=560
            )

            message_input = gr.Textbox(
                label="What are you looking for?",
                placeholder=(
                    "e.g. Find 5 sarees similar to this one"
                ),
                lines=2
            )

            search_button = gr.Button(
                "🔍 Find Similar Sarees",
                variant="primary"
            )

        # ----------------------------------------------------
        # RIGHT SIDE
        # ----------------------------------------------------

        with gr.Column(
            scale=1
        ):

            output = gr.Markdown(
                value=(
                    "### 👗 Similar Sarees\n\n"
                    "Upload an image and click "
                    "**Find Similar Sarees**."
                ),
                elem_classes="tt-results"
            )

    # --------------------------------------------------------
    # BUTTON EVENT
    # --------------------------------------------------------

    search_button.click(
        fn=tailor_talk_search,
        inputs=[
            image_input,
            message_input
        ],
        outputs=output
    )


# ============================================================
# 17. LAUNCH
# ============================================================

if __name__ == "__main__":

    print(
        "Starting TailorTalk..."
    )

    PORT = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    print(
        f"Starting Gradio on port {PORT}..."
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT
    )