import pandas as pd
import re
df = pd.read_csv("Resume.csv")
df.dropna(inplace=True)
df.fillna("", inplace=True)

df_sampled = df.groupby("Category").apply(lambda x: x.sample(n=20, random_state=42)).reset_index(drop=True)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'\s+', ' ', text)   
    text = re.sub(r'[^A-Za-z0-9.,!? ]+', '', text)  
    return text.strip()

# Apply cleaning function to Resume_str column
df_sampled["Resume_str"] = df_sampled["Resume_str"].apply(clean_text)
output_file = "25_rows.csv"
df_sampled.to_csv(output_file, index=False)