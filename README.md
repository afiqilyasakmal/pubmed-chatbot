# PubMed Chatbot

A Python-based RAG (Retrieval Augmented Generation) chatbot that answers questions using PubMed article abstracts. It features data fetching, preprocessing, embedding, retrieval, re-ranking with a Cross-Encoder, and answer generation using the Llama 3.

Slides: [coming soon]

## Core Technologies

* Python 3.x
* Hugging Face:
    * `transformers` (for Llama 3 LLM, tokenizer)
    * `sentence-transformers` (for `all-MiniLM-L6-v2` embeddings & `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranking)
    * `accelerate`, `bitsandbytes` (for LLM optimization)
* PyTorch
* Biopython (for PubMed API)
* NLTK (for text preprocessing)
* Pandas, NumPy, Scikit-learn
* Google Drive (for data and LLM storage)

## Setup

1.  **Environment**:
    * This script is designed to be run in an environment like Google Colab, but can be adapted.
    * Ensure Python 3.8+ is available.

2.  **Dependencies**:
    * Run the `!pip install ...` commands provided at the beginning of the script to install all necessary Python packages. Alternatively, create a `requirements.txt` from these commands for easier installation:
        ```bash
        pip install -r requirements.txt
        ```

3.  **NLTK Resources**:
    * The script automatically downloads NLTK resources (`stopwords`, `punkt`, `wordnet`). Ensure this step completes.

4.  **Hugging Face Login**:
    * To download models like Llama 3, log in to Hugging Face Hub by running the `notebook_login()` cell. Enter your token there.

5.  **Configuration**:
    * **Entrez Email**: Update `Entrez.email` in the script with your valid email address for PubMed API access.
        ```python
        Entrez.email = "your_email@example.com"
        ```
    * **Google Drive**:
        * The script will attempt to mount Google Drive at `/content/drive`.
        * Ensure the `gdrive_base_dir` variable (default: `"/content/drive/MyDrive/pubmed-chatbot-using-rag-llm"`) points to your desired storage location in Google Drive. This path will be used for datasets and the LLM.

## Data Preparation

The chatbot requires data fetched from PubMed, preprocessed, and converted into embeddings. The script handles this pipeline:

1.  **Fetching & Preprocessing**: `Workspace_pubmed_abstracts_multi_keyword()` and `batch_preprocess_dataframe_texts()` are called to get and clean data, saving `pubmed_dataset.csv` and `pubmed_dataset_cleaned.csv` to your Google Drive.
2.  **Embedding Generation**: `run_embedding_pipeline()` (called within the `if __name__ == "__main__":` block) processes the cleaned data to create and save:
    * `pubmed_article_embeddings.npy` (article embeddings)
    * `pubmed_articles_for_retrieval.csv` (metadata for retrieval)
    These files are stored in the configured Google Drive path and are essential for the chatbot. This data preparation stage can take a considerable amount of time, especially on the first run.

## Running the Chatbot

1.  Ensure all setup steps and configurations are complete.
2.  Execute the Python script (or run all cells in your Colab notebook).
3.  The script will:
    * Perform data preparation steps (fetching, preprocessing, embedding) if the necessary files are not found or if the logic is run sequentially.
    * Load the SBERT model, Cross-Encoder model.
    * Load the LLM (`meta-llama/Meta-Llama-3-8B-Instruct`). If not found in your Google Drive (`model` subdirectory), it will be downloaded from Hugging Face Hub and saved there.
    * Start the interactive chat session.
4. Run the cell under the `See it live!` section to try the chatbot.