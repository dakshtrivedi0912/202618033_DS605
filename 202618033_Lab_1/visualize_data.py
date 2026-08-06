import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Set visual style
sns.set_theme(style="whitegrid")

def generate_visualizations_and_insights(csv_file='cleaned_books.csv'):
    df = pd.read_csv(csv_file)
    
    # -------------------------------------------------------------------------
    # 1. Terminal Data-Driven Insights Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" TASK 3: DATA-DRIVEN INSIGHTS & STATISTICAL REPORT ")
    print("=" * 65)
    
    # General & Missing Value Check
    total_books = len(df)
    missing_desc = (df['product_description'] == "No description available.").sum()
    
    print(f"• Dataset Integrity & Overview:")
    print(f"  - Total Processed Books : {total_books}")
    print(f"  - Unique Categories     : {df['category'].nunique()}")
    print(f"  - Missing Descriptions  : {missing_desc} ({missing_desc/total_books*100:.1f}%)")
    
    # Financial & Stock Summary
    avg_price = df['price_gbp'].mean()
    median_price = df['price_gbp'].median()
    total_stock = df['stock_count'].sum()
    min_stock, max_stock = df['stock_count'].min(), df['stock_count'].max()
    
    print(f"\n• Pricing & Stock Patterns:")
    print(f"  - Average Price         : £{avg_price:.2f} (Median: £{median_price:.2f})")
    print(f"  - Total Inventory Stock : {total_stock} units across all titles")
    print(f"  - Stock Range per Book  : {min_stock} to {max_stock} units")
    
    # Highly Rated Titles
    high_rated = df[df['rating_num'] >= 4]
    print(f"\n• Satisfaction & Highly Rated Books:")
    print(f"  - Average Star Rating   : {df['rating_num'].mean():.2f} / 5.0")
    print(f"  - 4+ Star Books Count   : {len(high_rated)} ({len(high_rated)/total_books*100:.1f}% of catalog)")
    
    # Category Patterns
    cat_summary = df.groupby('category').agg(
        book_count=('title', 'count'),
        avg_price=('price_gbp', 'mean'),
        avg_rating=('rating_num', 'mean'),
        total_stock=('stock_count', 'sum')
    ).sort_values(by='book_count', ascending=False)
    
    print(f"\n• Top Categories Overview:")
    print(cat_summary.head(5).to_string())
    print("=" * 65 + "\n")

    # -------------------------------------------------------------------------
    # 2. Four Required Plots (2x2 Grid)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Task 3 — Book Dataset Analysis & Pipeline Dashboard', fontsize=16, fontweight='bold')

    # Plot 1: Price Distribution
    sns.histplot(data=df, x='price_gbp', kde=True, ax=axes[0, 0], color='#2b5c8f', bins=20)
    axes[0, 0].set_title('1. Price Distribution (£ GBP)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Price (£)')
    axes[0, 0].set_ylabel('Book Count')

    # Plot 2: Rating Distribution
    sns.countplot(data=df, x='rating_num', ax=axes[0, 1], palette='Blues_r', hue='rating_num', legend=False)
    axes[0, 1].set_title('2. Rating Distribution (1–5 Stars)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Star Rating')
    axes[0, 1].set_ylabel('Number of Books')

    # Plot 3: Average Price by Category (Top 10 Categories)
    top_categories = df['category'].value_counts().head(10).index
    df_top_cat = df[df['category'].isin(top_categories)]
    avg_price_cat = df_top_cat.groupby('category')['price_gbp'].mean().sort_values(ascending=False).reset_index()
    
    sns.barplot(data=avg_price_cat, y='category', x='price_gbp', ax=axes[1, 0], palette='crest', hue='category', legend=False)
    axes[1, 0].set_title('3. Average Price by Category (Top 10 Categories)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Mean Price (£)')
    axes[1, 0].set_ylabel('Category')

    # Plot 4: Relationship Plot — Price vs Rating
    sns.regplot(data=df, x='price_gbp', y='rating_num', ax=axes[1, 1],
                scatter_kws={'alpha':0.5, 'color':'#d95f02'}, line_kws={'color':'black', 'linewidth':1.5})
    axes[1, 1].set_title('4. Relationship Plot: Price vs. Rating', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Price (£)')
    axes[1, 1].set_ylabel('Rating (1–5 Stars)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dashboard_path = 'book_insights_dashboard.png'
    plt.savefig(dashboard_path, dpi=300)
    print(f"Chart dashboard saved to '{dashboard_path}'.")
    plt.show()

    # -------------------------------------------------------------------------
    # 3. Word Cloud from Combined Product Descriptions
    # -------------------------------------------------------------------------
    # Combine descriptions excluding fallback default text
    valid_descriptions = df[df['product_description'] != "No description available."]['product_description']
    combined_text = " ".join(valid_descriptions.dropna())

    custom_stopwords = set(STOPWORDS).union({
        'book', 'story', 'one', 'will', 'life', 'new', 'find', 'make', 'first', 'time', 'world', 's'
    })

    wc = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        stopwords=custom_stopwords,
        colormap='viridis',
        max_words=150
    ).generate(combined_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Product Descriptions Word Cloud', fontsize=16, fontweight='bold', pad=15)
    
    wordcloud_path = 'book_descriptions_wordcloud.png'
    plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
    print(f"Word Cloud saved to '{wordcloud_path}'.")
    plt.show()

if __name__ == "__main__":
    generate_visualizations_and_insights()