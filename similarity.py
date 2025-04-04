import pandas as pd
import numpy as np

import json
import boto3

# Load the dataset with embeddings
df_sampled = pd.read_csv("25rows_with_embeddings.csv")

# Convert embedding strings back to lists
import ast
df_sampled["embedding"] = df_sampled["embedding"].apply(ast.literal_eval)


# User provides a job description
job_query = input('Enter the job description: Digital Marketing Manager with 5+ years experience')


# Initialize AWS Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")  

def get_embedding(text):
    """Generate an embedding for a given text using Amazon Titan Embed Text v2."""
    payload = {"inputText": text}
    
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps(payload)
    )

    # Extract embedding from response
    result = json.loads(response["body"].read())
    return result["embedding"]

# Generate embedding for the job query
job_query_embedding = get_embedding(job_query)
print("✅ Job query embedding generated successfully!")


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compute similarity scores for all resumes
df_sampled["similarity_score"] = df_sampled["embedding"].apply(lambda emb: cosine_similarity(emb, job_query_embedding))

# Sort resumes by similarity score (highest first)
df_ranked = df_sampled.sort_values(by="similarity_score", ascending=False)

# Display top-ranked resumes
print("✅ Resumes ranked successfully! Here are the top matches:")
print(df_ranked[["ID", "Category", "similarity_score"]].head(10))
