# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "numpy",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import httpx
import chardet
from dotenv import load_dotenv

# Force matplotlib to use non-interactive backend
matplotlib.use('Agg')

# Load environment variables from a .env file
load_dotenv()

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # API token from environment

if not AIPROXY_TOKEN:
    raise ValueError("API token not set. Please set AIPROXY_TOKEN in the environment.")

def load_data(file_path):
    """Load CSV data with automatic encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # Detect file encoding
        encoding = result['encoding']
        return pd.read_csv(file_path, encoding=encoding)  # Load the CSV with detected encoding
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform basic analysis: summary stats, missing values, and correlation."""
    numeric_df = df.select_dtypes(include=['number'])  # Select numeric columns only
    analysis = {
        'summary': df.describe(include='all').to_dict(),  # Summary stats
        'missing_values': df.isnull().sum().to_dict(),   # Count missing values per column
        'correlation': numeric_df.corr().to_dict()       # Correlation matrix for numeric columns
    }
    return analysis

def visualize_data(df, output_dir):
    """Create and save distribution plots for numeric columns."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns  # Numeric columns
    for column in numeric_columns:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)  # Plot histogram with KDE
        plt.title(f'Distribution of {column}')
        output_path = os.path.join(output_dir, f'{column}_distribution.png')
        plt.savefig(output_path)  # Save plot to file
        plt.close()

def generate_narrative(analysis):
    """Generate a narrative from the analysis using an API call."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',  # Bearer token for authorization
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
    data = {
        "model": "gpt-4o-mini",  # Specify the model
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=90.0)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def main():
    """Main function: Load data, analyze, visualize, and generate insights."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze datasets and generate insights.")
    parser.add_argument("file_path", help="Path to the dataset CSV file.")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Load dataset
    df = load_data(args.file_path)

    # Analyze dataset
    analysis = analyze_data(df)

    # Visualize data
    visualize_data(df, args.output_dir)

    # Generate narrative
    narrative = generate_narrative(analysis)

    # Save narrative to README.md in the output directory
    with open(os.path.join(args.output_dir, 'README.md'), 'w') as f:
        f.write(narrative)

if __name__ == "__main__":
    main()
