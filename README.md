# TailorTalk — AI-Powered Visual Saree Search

TailorTalk is an AI-powered visual search assistant for a saree catalogue.

Users can upload a saree image and retrieve visually similar sarees from the catalogue using image embeddings and FAISS. A Gemini-powered agent acts as the conversational layer and calls the LangChain visual-search tool when visual similarity search is required. The user-facing application is built with Gradio.

## Features

- Upload a saree image for visual similarity search
- Extract visual features using a pretrained ResNet-50
- Compare both colour and structural visual information
- Search the catalogue using FAISS cosine similarity
- Return product name, SKU, similarity score, prices, stock and product link
- Remove duplicate SKUs so the same product is not repeatedly recommended
- Use LangChain to expose visual search as a callable tool
- Use Gemini as the conversational agent
- Provide a simple Gradio web interface

## System Pipeline

```text
User Image
    ↓
Image Preprocessing
    ↓
ResNet-50 Feature Extraction
    ├── Original Colour Representation
    └── Grayscale / Structural Representation
    ↓
Weighted Feature Combination
    ↓
L2 Normalization
    ↓
FAISS Similarity Search
    ↓
Unique Product / SKU Filtering
    ↓
Product Metadata
    ↓
Gemini + LangChain Agent
    ↓
Gradio Interface
```

## Visual Search Approach

TailorTalk uses a pretrained ResNet-50 model as a feature extractor rather than as a classifier.

Two visual representations are generated for each catalogue image:

1. **Colour representation** — preserves colour and overall visual appearance.
2. **Grayscale representation** — reduces dependence on colour and emphasizes structural information such as patterns, borders and design.

The representations are normalized and combined using:

- **Colour:** 35%
- **Structural / grayscale:** 65%

The final embedding is normalized and stored in a FAISS inner-product index. Because the vectors are L2-normalized, inner product corresponds to cosine similarity.

The weighting is intended to reduce purely colour-driven matches while still retaining useful colour similarity.

## Catalogue and Metadata

Catalogue images are downloaded locally and stored using their catalogue row index as the filename. An embedding metadata file maintains the mapping between image files and their generated embeddings.

The searchable catalogue combines the embedding metadata with the original catalogue information so that search results can be translated back into product information.

Returned information includes:

- Saree name
- SKU
- Similarity score
- Retail price
- Discounted price
- Stock
- Product website link

Multiple catalogue images may belong to the same SKU. The search tool therefore searches additional candidates and filters duplicate SKUs before returning the requested number of unique products.

## LangChain Visual Search Tool

The visual search functionality is exposed through the LangChain tool:

```text
search_similar_sarees(image_path, top_k)
```

The tool:

1. Loads the uploaded image.
2. Generates colour and grayscale embeddings.
3. Applies the same preprocessing and weighting used for the catalogue.
4. Searches the FAISS index.
5. Removes duplicate SKUs.
6. Returns product metadata and similarity scores.

The tool preserves the similarity scores returned by FAISS and does not invent or modify catalogue information.

## Gemini-Powered TailorTalk Agent

Gemini acts as the conversational layer of TailorTalk.

The agent determines whether the user's request requires visual similarity search. When appropriate, it calls the LangChain `search_similar_sarees` tool using the uploaded image.

The agent is instructed to:

- show only unique products
- preserve similarity scores
- use catalogue information returned by the tool
- avoid inventing product details
- answer general fashion questions without unnecessarily calling the search tool

## Gradio Interface

The Gradio interface provides the user-facing application.

Users can:

1. Upload a saree image.
2. Optionally describe what they are looking for.
3. Click **Find Similar Sarees**.
4. Receive the matching catalogue results from TailorTalk.

## Project Structure

```text
TailorTalk/
├── data/
│   ├── images/
│   ├── saree.csv
│   ├── embedding_metadata.csv
│   ├── colour_embeddings.npy
│   ├── grayscale_embeddings.npy
│   ├── image_embeddings.npy
│   └── saree_faiss.index
│
├── notebook.ipynb
└── README.md
```

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install numpy pandas pillow torch torchvision faiss-cpu
pip install langchain langchain-google-genai gradio
```

## Gemini API Key

TailorTalk requires a Gemini API key for the conversational agent.

Set the key as an environment variable.

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

### Windows Command Prompt

```cmd
set GEMINI_API_KEY=YOUR_API_KEY
```

**Do not commit the API key to GitHub.**

## Running the Project

Open `notebook.ipynb` in Jupyter Notebook or VS Code and run the cells in order.

The notebook:

1. Loads and validates the catalogue.
2. Downloads and verifies catalogue images.
3. Generates colour and structural embeddings.
4. Builds the FAISS index and metadata mapping.
5. Creates the LangChain visual-search tool.
6. Validates the search using catalogue images.
7. Creates the Gemini-powered TailorTalk agent.
8. Launches the Gradio interface.

The final Gradio application is launched with:

```python
demo.launch()
```

## Visual Search Validation

The visual search pipeline includes a sanity check using an image that already exists in the catalogue.

An exact catalogue image should retrieve itself as the highest-scoring result. In testing, the same catalogue image produced a similarity score of **1.0000**, confirming that the embedding generation, normalization and FAISS search pipeline are aligned for an exact-match query.

Additional tests were performed using sarees with different colours and patterns to evaluate whether the retrieval system could identify visually related products rather than relying only on colour.

## API Rate Limits

The Gemini conversational layer depends on the active Gemini API quota and rate limits.

If the Gemini API temporarily returns a `429 RESOURCE_EXHAUSTED` response, this indicates that the API quota or rate limit has been reached rather than a failure of the FAISS visual-search implementation. The application can be tested again after the applicable quota becomes available.

## Technology Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **ResNet-50**
- **FAISS**
- **NumPy**
- **Pandas**
- **Pillow**
- **LangChain**
- **Google Gemini**
- **Gradio**

## Project Goal

TailorTalk demonstrates an end-to-end visual product-search workflow that combines computer vision embeddings, vector similarity search, product metadata retrieval and a conversational AI interface for a saree catalogue.