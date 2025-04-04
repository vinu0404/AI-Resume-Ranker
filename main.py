import streamlit as st
import pandas as pd
import numpy as np
import json
import boto3
import ast

# AWS Bedrock setup
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")  

# Load the preprocessed resume dataset
df_sampled = pd.read_csv("25rows_with_embeddings.csv")

# Convert embedding strings back to lists
df_sampled["embedding"] = df_sampled["embedding"].apply(ast.literal_eval)

# Function to generate embeddings using Amazon Titan
def get_embedding(text):
    """Generate an embedding for the given text using Amazon Titan."""
    payload = {"inputText": text}
    
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps(payload)
    )

    result = json.loads(response["body"].read())
    return result["embedding"]

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Streamlit UI
st.set_page_config(page_title="Resume Screening App", layout="wide")

st.title("🔍 AI-Powered Resume Screening")
st.write("Enter a job description and find the most relevant resumes!")

# Input box for job query
job_query = st.text_area("Enter Job Description", placeholder="e.g.,Machine Learning,Data Science & Have Intership In Domain Of Machine Learning")

# Submit button
if st.button("Find Matching Resumes"):
    if job_query:
        # Generate embedding for job query
        with st.spinner("Processing... ⏳"):
            job_query_embedding = get_embedding(job_query)

        # Compute similarity scores
        df_sampled["similarity_score"] = df_sampled["embedding"].apply(lambda emb: cosine_similarity(emb, job_query_embedding))

        # Rank resumes

        df_ranked = df_sampled.sort_values(by="similarity_score", ascending=False).head(10)

        # Display top resumes
        st.subheader("🏆 Top Matching Resumes")
        st.dataframe(df_ranked[["ID", "Category", "similarity_score"]])
    else:
        st.warning("⚠️ Please enter a job description.")
