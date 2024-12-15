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
from typing import Dict

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')

# Load environment variables from a .env file
load_dotenv()

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
MAX_VISUALIZATIONS = 3

if not AIPROXY_TOKEN:
    raise ValueError("API token not set. Please set AIPROXY_TOKEN in the environment.")

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data with automatic encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")
        df = pd.read_csv(file_path, encoding=encoding)
        if df.empty:
            raise ValueError("Dataset is empty.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df: pd.DataFrame) -> Dict:
    """Perform basic analysis: summary stats, missing values, and correlation."""
    if df.empty:
        raise ValueError("DataFrame is empty during analysis.")
    
    numeric_df = df.select_dtypes(include=['number'])
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict() if not numeric_df.empty else {},
        'outliers': {}
    }

    # Detect outliers using IQR
    for column in numeric_df.columns:
        Q1 = numeric_df[column].quantile(0.25)
        Q3 = numeric_df[column].quantile(0.75)
        IQR = Q3 - Q1
        analysis['outliers'][column] = numeric_df[(numeric_df[column] < Q1 - 1.5 * IQR) | (numeric_df[column] > Q3 + 1.5 * IQR)].shape[0]

    return analysis

def visualize_data(df: pd.DataFrame, output_dir: str) -> None:
    """Create and save visualizations for the dataset."""
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
def generate_story_with_function_call(df: pd.DataFrame, file_path: str, analysis: Dict, output_dir: str) -> str:
    """Generate a narrative using LLM with function calling enabled."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }

    column_info = {
        col: {"dtype": str(df[col].dtype), "examples": df[col].dropna().head().tolist() if not df[col].dropna().empty else []}
        for col in df.columns
    }

    # Pass only the number of null values for each column
    missing_values = pd.Series(analysis['missing_values']).to_string()

    prompt = f"""
You are an AI data analyst. Here's a detailed analysis of a dataset provided by the user.

**Dataset Overview:**
File Name: {os.path.basename(file_path)}
Number of Rows: {df.shape[0]}
Number of Columns: {df.shape[1]}
Column Information:
{column_info}

**Analysis Insights:**
- Summary Statistics:
{pd.DataFrame(analysis['summary']).to_string()}
- Missing Values:
{missing_values}
- Correlation Insights:
{pd.DataFrame(analysis['correlation']).to_string()}
- Outliers Detected:
{pd.Series(analysis['outliers']).to_string()}

**Your Task:**
1. Provide a detailed narrative explaining:
   - The structure and key characteristics of the dataset.
   - The analysis performed and its significance.
   - Insights uncovered from the data, focusing on trends, anomalies, and relationships.
2. Recommend actionable next steps for further exploration or decision-making.
3. A brief conclusion of the whole analysis.
4. The readme file must look professional and explain every findings and draw proper insights from them.
"""

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
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

def generate_readme(df: pd.DataFrame, file_path: str, analysis: Dict, output_dir: str) -> None:
    """Generate a professional README.md with analysis summary and insights."""
    narrative = generate_story_with_function_call(df, file_path, analysis, output_dir)

    # Ensure proper filenames for each plot
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Add visualizations dynamically
    visualizations = "## Visualizations\n"

    # Heatmap visualization
    if len(numeric_columns) > 1:
        visualizations += """
## Correlation Matrix Heatmap
![Correlation Matrix Heatmap](correlation_matrix_heatmap.png)

This heatmap visualizes the relationships between numeric features in the dataset.
"""

    # Histogram for the first numeric column
    if len(numeric_columns) > 0:
        visualizations += f"""
## Distribution of {numeric_columns[0]}
![{numeric_columns[0]} Distribution]({numeric_columns[0]}_distribution.png)

This histogram shows the distribution of the numeric column '{numeric_columns[0]}', providing insights into its spread and frequency.
"""

    # Bar chart for the most frequent categorical column
    if len(categorical_columns) > 0: 
        column_to_plot = min(categorical_columns, key=lambda col: df[col].nunique()) 
        column_filename = f'{column_to_plot}_bar_chart.png' 
        if df[column_to_plot].nunique() > 1:  # Ensure the column has more than 1 unique value
            visualizations += f"""
## Distribution of {column_to_plot}
![{column_to_plot} Bar Chart]({column_to_plot}_bar_chart.png)

This bar chart shows the distribution of the categorical column '{column_to_plot}', highlighting the frequency of each category.
"""
        else:
            print(f"Skipping bar chart: {column_to_plot} has only 1 unique value.")
    else:
        print("No categorical columns available for bar chart.")

    # Final README content
    readme_content = f"""
{narrative}

{visualizations}
"""

    # Write the README file
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
