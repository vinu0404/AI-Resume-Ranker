#  AI-Powered Resume Screening System
[DEMO VIDEO LINK](https://drive.google.com/file/d/1SH-a1j0CML2JReehHwkkx9UrxYSHb98K/view?usp=sharing)

Automate and streamline resume screening using NLP, vector similarity, and agentic querying with **AWS Bedrock** and a  **Streamlit dashboard**.


## Project Overview

Companies receive hundreds or thousands of resumes per job opening. This project automates the screening process to help recruiters:

- Upload or connect resumes from local CSV.
- Enter a job requirement/query through  UI.
- Rank resumes based on similarity to the input using LLMs and embeddings.
- Display top matches with relevance scores.


## Tech Stack

| Component         | Tool/Library                         |
|------------------|--------------------------------------|
| Frontend UI      | Streamlit                            |
| Resume Parsing   | PyPDF2, Pandas                       |
| LLM              | `amazon.nova-pro-v1:0` (via AWS Bedrock) |
| Embeddings       | `amazon.titan-embed-text-v2:0`       |
