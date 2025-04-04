import boto3
import json
import pandas as pd

# Load the sampled resumes dataset
df_sampled = pd.read_csv("25_rows.csv")
# Initialize the Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")  # Change region if needed

def get_embedding(text):
    """Generate an embedding for a given text using Amazon Titan Embed Text v2."""
    payload = {
        "inputText": text
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps(payload)
    )

    # Extract embedding from response
    result = json.loads(response["body"].read())
    return result["embedding"]

# Generate embeddings for each resume
df_sampled["embedding"] = df_sampled["Resume_str"].apply(get_embedding)

# Save the dataset with embeddings to avoid recomputation
df_sampled.to_csv("25rows_with_embeddings.csv", index=False)

print("✅ Embeddings generated and saved successfully!")
