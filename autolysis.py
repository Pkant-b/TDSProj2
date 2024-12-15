# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "seaborn",
#     "pandas",
#     "matplotlib",
#     "httpx",
#     "chardet",
#     "numpy",
#     "python-dotenv",
#     "tenacity"
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
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from io import StringIO

# Force matplotlib to use non-interactive backend
matplotlib.use('Agg')

# Load environment variables from a .env file
load_dotenv()

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # API token from environment
MAX_VISUALIZATIONS = 3  # Set a limit on the number of visualizations

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
        'correlation': numeric_df.corr().to_dict(),       # Correlation matrix for numeric columns
        'outliers': {}                                    # Outlier detection
    }

    # Outlier detection using IQR
    for column in numeric_df.columns:
        Q1 = numeric_df[column].quantile(0.25)
        Q3 = numeric_df[column].quantile(0.75)
        IQR = Q3 - Q1
        analysis['outliers'][column] = numeric_df[(numeric_df[column] < Q1 - 1.5 * IQR) | (numeric_df[column] > Q3 + 1.5 * IQR)].shape[0]

    return analysis

def visualize_data(df, output_dir):
    """Create and save main plots for the dataset."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns[:MAX_VISUALIZATIONS]  # Limit the number of visualizations

    # Heatmap of the correlation matrix
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        plt.savefig(os.path.join(output_dir, 'correlation_matrix_heatmap.png'), dpi=100)
        plt.close()

    # Histogram for the first numeric column
    if len(numeric_columns) > 0:
        plt.figure()
        sns.histplot(df[numeric_columns[0]].dropna(), kde=True)
        plt.title(f'Distribution of {numeric_columns[0]}')
        plt.savefig(os.path.join(output_dir, f'{numeric_columns[0]}_distribution.png'), dpi=100)
        plt.close()

    # Scatter plot for the first two numeric columns (if available)
    if len(numeric_columns) > 1:
        plt.figure()
        sns.scatterplot(x=df[numeric_columns[0]], y=df[numeric_columns[1]])
        plt.title(f'Scatter Plot of {numeric_columns[0]} vs {numeric_columns[1]}')
        plt.savefig(os.path.join(output_dir, f'{numeric_columns[0]}_vs_{numeric_columns[1]}_scatter.png'), dpi=100)
        plt.close()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_narrative(file_path, column_info, df_info, df_describe, missing_values, outliers):
    """Generate a narrative from the analysis using an API call."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',  # Bearer token for authorization
        'Content-Type': 'application/json'
    }
    prompt = f"""
    You are analyzing a dataset. Below are details about it:

    File: {file_path}
    Columns and Data Types:
    {column_info}

    Dataset Info:
    {df_info}

    Summary Statistics:
    {df_describe}

    Missing Values:
    {missing_values}

    Outliers Detected:
    {outliers}

    Please write a summary of this analysis, including insights and recommendations.
    """
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
        raise
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def generate_readme(df, file_path, analysis, output_dir):
    """Generate README.md with analysis summary and insights."""
    column_info = {
        col: {"dtype": str(df[col].dtype), "examples": df[col].dropna().head().tolist() if not df[col].dropna().empty else []}
        for col in df.columns
    }

    # Get dataset info without printing it to the terminal
    buffer = StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()
    df_describe = df.describe(include='all').to_string()
    missing_values = pd.Series(analysis['missing_values']).to_string()
    outliers = pd.Series(analysis['outliers']).to_string()

    # Generate narrative using analysis
    narrative = generate_narrative(file_path, column_info, df_info, df_describe, missing_values, outliers)

    # Write README content
    readme_content = f"""# Analysis of {file_path}

## Detailed Explanation of the Result of Analysis

{narrative}

## Visualizations

"""
    # Add individual column distribution plots
    numeric_columns = df.select_dtypes(include=['number']).columns[:MAX_VISUALIZATIONS]  # Limit the number of visualizations

    if len(numeric_columns) > 0:
        chart_filename = f"{numeric_columns[0]}_distribution.png"
        readme_content += f"### Distribution of {numeric_columns[0]}\n\n![{chart_filename}]({chart_filename})\n\n"

    if len(numeric_columns) > 1:
        chart_filename = f"{numeric_columns[0]}_vs_{numeric_columns[1]}_scatter.png"
        readme_content += f"### Scatter Plot of {numeric_columns[0]} vs {numeric_columns[1]}\n\n![{chart_filename}]({chart_filename})\n\n"

    readme_content += "### Correlation Matrix Heatmap\n\n![correlation_matrix_heatmap.png](correlation_matrix_heatmap.png)\n\n"

    # Save README.md
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)

def main():
    """Main function: Load data, analyze, visualize, and generate insights."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze datasets and generate insights.")
    parser.add_argument("file_path", help="Path to the dataset CSV file.")
    parser.add_argument("-o", "--output_dir", default=".", help="Directory to save outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Load dataset
    df = load_data(args.file_path)

    # Analyze dataset
    analysis = analyze_data(df)

    # Visualize data
    visualize_data(df, args.output_dir)

    # Create README.md
    generate_readme(df, args.file_path, analysis, args.output_dir)

    print("Done!")

if __name__ == "__main__":
    main()
