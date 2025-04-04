import pandas as pd
import re
# Load the dataset
df = pd.read_csv("Resume.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# OR fill missing text with an empty string (if necessary)
df.fillna("", inplace=True)

# Select 10 resumes per category
df_sampled = df.groupby("Category").apply(lambda x: x.sample(n=20, random_state=42)).reset_index(drop=True)

# Check if sampling worked
print(df_sampled["Category"].value_counts())



def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)   # Remove extra spaces
    text = re.sub(r'[^A-Za-z0-9.,!? ]+', '', text)  # Keep only relevant characters
    return text.strip()

# Apply cleaning function to Resume_str column
df_sampled["Resume_str"] = df_sampled["Resume_str"].apply(clean_text)
output_file = "25_rows.csv"
df_sampled.to_csv(output_file, index=False)