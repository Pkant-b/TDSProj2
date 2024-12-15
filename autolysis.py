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
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
MAX_VISUALIZATIONS = 3

if not AIPROXY_TOKEN:
    raise ValueError("API token not set. Please set AIPROXY_TOKEN in the environment.")

def load_data(file_path):
    """Load CSV data with automatic encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform basic analysis: summary stats, missing values, and correlation."""
    numeric_df = df.select_dtypes(include=['number'])
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict(),
        'outliers': {}
    }

    for column in numeric_df.columns:
        Q1 = numeric_df[column].quantile(0.25)
        Q3 = numeric_df[column].quantile(0.75)
        IQR = Q3 - Q1
        analysis['outliers'][column] = numeric_df[(numeric_df[column] < Q1 - 1.5 * IQR) | (numeric_df[column] > Q3 + 1.5 * IQR)].shape[0]

    return analysis

def visualize_data(df, output_dir):
    """Create and save main plots for the dataset."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns[:MAX_VISUALIZATIONS]

    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        plt.savefig(os.path.join(output_dir, 'correlation_matrix_heatmap.png'), dpi=100)
        plt.close()

    if len(numeric_columns) > 0:
        plt.figure()
        sns.histplot(df[numeric_columns[0]].dropna(), kde=True)
        plt.title(f'Distribution of {numeric_columns[0]}')
        plt.savefig(os.path.join(output_dir, f'{numeric_columns[0]}_distribution.png'), dpi=100)
        plt.close()

    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        column_to_plot = min(categorical_columns, key=lambda col: df[col].nunique())
        plt.figure()
        sns.countplot(y=column_to_plot, data=df, order=df[column_to_plot].value_counts().index)
        plt.title(f'Distribution of {column_to_plot}')
        plt.xlabel('Count')
        plt.savefig(os.path.join(output_dir, f'{column_to_plot}_bar_chart.png'), dpi=100)
        plt.close()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_story(file_path, column_info, df_info, summary, missing_values, correlation, outliers):
    """Generate a narrative from the analysis using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"""
    You are an AI assisting with data analysis. Here's the analysis:

    **Dataset Information:**
    File Path: {file_path}
    Columns and Data Types: {column_info}
    Dataset Info: {df_info}
    Summary Statistics: {summary}
    Missing Values: {missing_values}
    Correlation: {correlation}
    Outliers: {outliers}

    Write a detailed story explaining:
    1. The dataset and its key features.
    2. The analysis performed and its significance.
    3. Key insights uncovered, including insights from outliers.
    4. Implications and recommended actions based on findings.
    """
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=90.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def generate_readme(df, file_path, analysis, output_dir):
    """Generate a professional README.md with analysis summary and insights."""
    column_info = {
        col: {"dtype": str(df[col].dtype), "examples": df[col].dropna().head().tolist() if not df[col].dropna().empty else []}
        for col in df.columns
    }

    buffer = StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()

    summary = pd.DataFrame(analysis['summary']).to_string()
    missing_values = pd.Series(analysis['missing_values']).to_string()
    correlation = pd.DataFrame(analysis['correlation']).to_string()
    outliers = pd.Series(analysis['outliers']).to_string()

    visualizations = """Generated visualizations include correlation heatmaps, histograms, and bar charts."""
    narrative = generate_story(file_path, column_info, df_info, summary, missing_values, correlation, outliers)

    readme_content = f"""# Analysis of {os.path.basename(file_path)}

## Dataset Overview

This dataset was analyzed to uncover insights regarding its structure, trends, and patterns. The analysis focused on identifying missing values, statistical summaries, correlations, and potential outliers.

## Key Insights

{narrative}

## Visualizations

Key visualizations include:
- Correlation heatmap to identify relationships between numeric features.
- Histograms for feature distribution.
- Bar charts for categorical feature distribution.

The visualizations have been saved in the output directory.

### Correlation Matrix Heatmap
![correlation_matrix_heatmap.png](correlation_matrix_heatmap.png)

### Distribution of {df.select_dtypes(include=['number']).columns[0]}
![{df.select_dtypes(include=['number']).columns[0]}_distribution.png]({df.select_dtypes(include=['number']).columns[0]}_distribution.png)

### Distribution of {min(df.select_dtypes(include=['object']).columns, key=lambda col: df[col].nunique())}
![{min(df.select_dtypes(include=['object']).columns, key=lambda col: df[col].nunique())}_bar_chart.png]({min(df.select_dtypes(include=['object']).columns, key=lambda col: df[col].nunique())}_bar_chart.png)

## Conclusion

This analysis provides a comprehensive understanding of the dataset, highlighting areas of interest and opportunities for further exploration.
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)

def main():
    """Main function: Load, analyze, visualize, and generate insights."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze datasets and generate insights.")
    parser.add_argument("file_path", help="Path to the dataset CSV file.")
    parser.add_argument("-o", "--output_dir", default=".", help="Directory to save outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.file_path)
    analysis = analyze_data(df)
    visualize_data(df, args.output_dir)
    generate_readme(df, args.file_path, analysis, args.output_dir)

    print("Analysis completed and README generated.")

if __name__ == "__main__":
    main()
