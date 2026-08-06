import json
import re
import pandas as pd
import numpy as np

def run_preprocessing():
    # 1. Load raw data safely (handles standard JSON array, JSON Lines, and CSV fallback)
    try:
        df = pd.read_json('scraped_books.json')
    except ValueError:
        try:
            df = pd.read_json('scraped_books.json', lines=True)
        except Exception:
            df = pd.read_csv('scraped_books.csv')

    initial_count = len(df)

    # 2. Clean text & handle duplicates / missing values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.drop_duplicates(subset=['upc'], keep='first').reset_index(drop=True)
    dedup_count = len(df)

    df['product_description'] = df['product_description'].replace(['', 'None', 'nan', 'null'], np.nan)
    df['product_description'] = df['product_description'].fillna("No description available.")

    # 3. Numeric conversions
    df['price_gbp'] = df['price'].astype(str).str.extract(r'(\d+\.\d+)')[0].astype(float)

    rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    df['rating_num'] = df['rating'].map(rating_map).fillna(0).astype(int)

    def parse_stock(text):
        match = re.search(r'\((\d+)\s+available\)', str(text))
        return int(match.group(1)) if match else (1 if 'In stock' in str(text) else 0)

    df['stock_count'] = df['availability'].apply(parse_stock)
    df['number_of_reviews'] = pd.to_numeric(df['number_of_reviews'], errors='coerce').fillna(0).astype(int)

    # 4. Create engineered features
    df['description_word_count'] = df['product_description'].apply(
        lambda x: len(re.findall(r'\w+', str(x))) if x != "No description available." else 0
    )

    bins = [0, 20, 40, float('inf')]
    labels = ['Budget (<£20)', 'Mid-Range (£20-£40)', 'Premium (>£40)']
    df['price_band'] = pd.cut(df['price_gbp'], bins=bins, labels=labels, right=False)

    df['value_score'] = np.round((df['rating_num'] / df['price_gbp']) * 10, 2)
    df['recommended'] = (df['rating_num'] >= 4) & (df['price_gbp'] <= 35.0)

    # 5. Export cleaned dataset
    df.to_csv('cleaned_books.csv', index=False)
    df.to_json('cleaned_books.json', orient='records', indent=2)

    print("=" * 50)
    print(" TASK 2: PREPROCESSING SUMMARY ")
    print("=" * 50)
    print(f"Raw Records Loaded    : {initial_count}")
    print(f"Duplicates Removed    : {initial_count - dedup_count}")
    print(f"Cleaned Records Saved : {len(df)}")
    print("-" * 50)
    print("Sample Output:")
    print(df[['title', 'price_gbp', 'rating_num', 'stock_count', 'price_band', 'value_score']].head())
    print("=" * 50)

if __name__ == "__main__":
    run_preprocessing()