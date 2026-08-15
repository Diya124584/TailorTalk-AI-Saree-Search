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

# Images were downloaded using the original catalogue
# row index:
#
# catalogue row 0 -> 0.jpg
# catalogue row 1 -> 1.jpg
# catalogue row 10 -> 10.jpg

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
# 8. LOAD DINOv2
# ============================================================

MODEL_NAME = "facebook/dinov2-small"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Loading DINOv2 on:",
    device
)

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

visual_model = AutoModel.from_pretrained(
    MODEL_NAME
)

visual_model = visual_model.to(
    device
)

visual_model.eval()

print(
    "DINOv2 loaded successfully!"
)


# ============================================================
# 9. DINOv2 EMBEDDING FUNCTION
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
# 10. LOAD FAISS INDEX
# ============================================================

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


# ============================================================
# 11. VALIDATE FAISS + METADATA
# ============================================================

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
# 12. LANGCHAIN VISUAL SEARCH TOOL
# ============================================================

@tool
def search_similar_sarees(
    image_path: str,
    top_k: int = 5
) -> list:
    """
    Find visually similar sarees from the catalogue.

    The FAISS index uses normalized DINOv2 structural/
    grayscale embeddings.

    Returns unique in-stock products.
    """

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    try:
        top_k = int(top_k)

    except Exception:
        top_k = 5

    top_k = max(
        1,
        min(top_k, 10)
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
    # Load image
    # --------------------------------------------------------

    query_image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The current FAISS index was created using
    # structure_embeddings.
    #
    # Therefore the query MUST also use the
    # grayscale DINOv2 representation.
    # --------------------------------------------------------

    grayscale_image = (
        query_image
        .convert("L")
        .convert("RGB")
    )


    query_embedding = get_dino_embedding(
        grayscale_image
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
    # We need extra results because:
    #
    # - some products are out of stock
    # - duplicate SKUs can exist
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
        # Prices
        # ----------------------------------------------------

        try:

            retail_price = float(
                row["Retail Price"]
            )

        except Exception:

            retail_price = 0.0


        try:

            discounted_price = float(
                row["Discounted Price"]
            )

        except Exception:

            discounted_price = 0.0


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

            "website_link": str(
                row["Website Link"]
            )
        })


        if len(results) >= top_k:
            break


    return results


print(
    "Visual search tool created successfully!"
)


# ============================================================
# 13. FORMAT SEARCH RESULTS
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
# 14. GEMINI AGENT
# ============================================================

agent = None

gemini_api_key = os.getenv(
    "GEMINI_API_KEY"
)


if gemini_api_key:

    try:

        print(
            "Initializing Gemini agent..."
        )


        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
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


    except Exception as e:

        print(
            "Gemini initialization failed:"
        )

        print(
            str(e)
        )

        agent = None

else:

    print(
        "GEMINI_API_KEY not configured."
    )

    print(
        "Using direct visual search fallback."
    )


# ============================================================
# 15. EXTRACT AGENT RESPONSE
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
# 16. MAIN TAILORTALK SEARCH
# ============================================================

def tailor_talk_search(
    image_path,
    user_message
):

    if not image_path:

        return (
            "## ⚠️ Please upload a saree image first."
        )


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

    if agent is not None:

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

            response = agent.invoke({
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


            if text and text.strip():

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

        results = search_similar_sarees(
            image_path=image_path,
            top_k=5
        )

        return format_results(
            results
        )


    except Exception as e:

        return (
            "## ❌ Search Error\n\n"
            f"`{str(e)}`"
        )


# ============================================================
# 17. GRADIO FRONTEND
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
        # LEFT
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
        # RIGHT
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
# 18. LAUNCH
# ============================================================

if __name__ == "__main__":

    print(
        "Starting TailorTalk..."
    )

    demo.launch()