# Analysis of Age-Friendly Community Reports using NLP and Generative AI

## 🎯 Project Objective

Its purpose is to demonstrate proficiency in the specific data compilation and analysis skills outlined in the job description, including web scraping, OCR, NLP, and the implementation of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG).

The project analyzes official "Age-Friendly Community" action plans to extract key themes, identify stakeholders, and create an interactive Q&A system to make the information within these reports easily accessible.

## ✨ Key Feature: Interactive Q&A System

The final output of this project is a command-line tool that allows a user to "chat" with the downloaded reports. You can ask questions in plain English and receive answers generated by Google's Gemini model, based on the information contained within the documents.

**Example Query:**
> "What are the primary recommendations for improving social participation for older adults?"

## 🛠️ Technologies & Libraries Used

* **Programming Language:** Python
* **Data Ingestion:** `requests`, `BeautifulSoup4`
* **Text Extraction (OCR):** `PyMuPDF`, `pytesseract`
* **NLP Analysis:**
    * **Topic Modeling:** `scikit-learn` (Latent Dirichlet Allocation)
    * **Named Entity Recognition (NER):** `spaCy`
* **Generative AI (RAG):**
    * **Framework:** `LangChain`
    * **LLM:** `Google Gemini Pro`
    * **Embeddings:** `GoogleGenerativeAIEmbeddings`
    * **Vector Database:** `ChromaDB`

## 🌊 Project Workflow

The project is broken down into a sequential pipeline, with each script performing a specific task.

1.  **`1_data_ingestion.py`**: Scrapes the AARP website for links to PDF reports from age-friendly communities and downloads them into the `/reports` directory.
2.  **`2_text_extraction.py`**: Processes each PDF in the `/reports` folder. It uses OCR (Tesseract) to extract the raw text and saves it as a `.txt` file in the `/text_output` directory.
3.  **`4_analysis_with_saving.py`**: Reads the extracted text files and performs two NLP tasks:
    * **Topic Modeling:** Identifies the top 10 recurring themes across the documents.
    * **Named Entity Recognition:** Extracts the most common organizations and locations mentioned.
    * The results are saved to `analysis_results/analysis_output.json`.
4.  **`5_rag_qa_system_gemini.py`**:
    * Loads the text files, splits them into chunks, and converts them into vector embeddings using Google's model.
    * Stores these embeddings in a local vector database (`/db_gemini`).
    * Launches an interactive command-line interface that uses a RAG pipeline to answer user questions based on the document contents.

## 📊 Key Findings

The automated analysis (results stored in `analysis_results/analysis_output.json`) provides a high-level overview of the strategic priorities in the reports. This includes:

* **Dominant Themes:** Identification of key topics such as `transportation`, `housing`, `healthcare access`, `social participation`, and `civic engagement`.
* **Key Stakeholders:** Recognition of frequently mentioned entities like the `Department for the Aging (DFTA)`, `AARP`, and various city councils and community boards.

## 🚀 Setup and Usage

To replicate this project, please follow these steps:

### 1. Prerequisites

* Python 3.8+
* Tesseract OCR Engine installed and accessible in your system's PATH.
* A Google AI API Key for using the Gemini model.

### 2. Installation

Clone the repository and install the required Python packages:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt
```

### 3. Set API Key

Set your Google API key as an environment variable.

On macOS/Linux:

```
export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
```

On Windows:
```
set GOOGLE_API_KEY=YOUR_API_KEY_HERE
```
### 4. Running the Pipeline
Execute the scripts in the following order:
```
# Step 1: Download PDF reports
python 1_data_ingestion.py

# Step 2: Extract text from PDFs
python 2_text_extraction.py

# Step 3: Perform NLP analysis and save results
python 4_analysis_with_saving.py

# Step 4: Launch the interactive Q&A system
python 5_rag_qa_system_gemini.py
```
After running Step 4, the system will be ready. Follow the on-screen prompts to ask questions about the documents. Type exit to quit.
