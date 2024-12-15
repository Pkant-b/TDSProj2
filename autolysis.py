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
#     "tabulate"
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
from tabulate import tabulate
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

    # Outlier detection using IQR
    outliers = {}
    for column in numeric_df.columns:
        Q1 = numeric_df[column].quantile(0.25)
        Q3 = numeric_df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers[column] = numeric_df[(numeric_df[column] < Q1 - 1.5 * IQR) | (numeric_df[column] > Q3 + 1.5 * IQR)].shape[0]
    analysis['outliers'] = outliers

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

    # Heatmap of the correlation matrix
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        plt.savefig(os.path.join(output_dir, 'correlation_matrix_heatmap.png'), dpi=100)
        plt.close()

def generate_narrative(file_path, column_info, summary, missing_values, correlation, outliers):
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

    Summary Statistics:
    {summary}

    Missing Values:
    {missing_values}

    Correlation Matrix (if any):
    {correlation}

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
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def generate_readme(df, file_path, analysis, output_dir):
    """Generate README.md with analysis summary and insights."""
    column_info = {
        col: {"dtype": str(df[col].dtype), "examples": df[col].dropna().sample(5).tolist() if not df[col].dropna().empty else []}
        for col in df.columns
    }

    summary = tabulate(pd.DataFrame(analysis['summary']), headers='keys', tablefmt='pipe')
    missing_values = tabulate(pd.Series(analysis['missing_values']).to_frame(), headers='keys', tablefmt='pipe')
    correlation = tabulate(pd.DataFrame(analysis['correlation']), headers='keys', tablefmt='pipe') if analysis['correlation'] else "No correlation matrix available."
    outliers = tabulate(pd.Series(analysis['outliers']).to_frame(), headers='keys', tablefmt='pipe')

    # Generate narrative using analysis
    narrative = generate_narrative(file_path, column_info, summary, missing_values, correlation, outliers)

    # Write README content
    readme_content = f"""# Analysis of {file_path}

## Analysis Summary

### Summary Statistics

{summary}

### Missing Values

{missing_values}

### Correlation Matrix

{correlation}

### Outliers Detected

{outliers}

## Insights

{narrative}

## Visualizations

"""
    visualization_descriptions = {
        "correlation_matrix_heatmap.png": "### Correlation Matrix Heatmap\nHighlights the strength of relationships between numerical features.",
    }

    # Add individual column distribution plots
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        chart_filename = f"{column}_distribution.png"
        chart_path = os.path.join(output_dir, chart_filename)
        if os.path.exists(chart_path):
            readme_content += f"### Distribution of {column}\n\n![{chart_filename}]({chart_filename})\n\n"

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
