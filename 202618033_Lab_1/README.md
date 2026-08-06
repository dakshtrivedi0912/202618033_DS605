# DS605 - Fundamentals of Machine Learning

## Lab Assignment 1: Data Scraping and Preprocessing using Python and Scrapy

**Name:** Daksh Trivedi  
**Student ID:** 202618033

---

## Objective

The objective of this assignment is to build an end-to-end data pipeline that harvests book information from Books to Scrape using Scrapy, cleanses and transforms the extracted raw data, performs exploratory data analysis, generates analytical visualizations, and interprets data-driven business insights.

---

## Tools and Libraries Used

- Python
- Scrapy
- Pandas & NumPy
- Matplotlib & Seaborn
- WordCloud

---

## Project Structure

```
202618033_Lab_1/
├── bookscraper/
│   ├── __pycache__/
│   ├── spiders/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   └── books_spider.py
│   ├── __init__.py
│   ├── items.py
│   ├── middlewares.py
│   ├── pipelines.py
│   └── settings.py
├── book_descriptions_wordcloud.png
├── book_insights_dashboard.png
├── cleaned_books.csv
├── cleaned_books.json
├── preprocess_data.py
├── scraped_books.csv
├── scraped_books.json
├── scrapy.cfg
├── visualize_data.py
└── README.md

```

---

## How to Run

1. Run the Scrapy spider:

```bash
python booksscraper.py
```

2. Preprocess the dataset:

```bash
python preprocess_data.py
```

3. Generate visualizations:

```bash
python visualize_data.py
```

---

## Results

- Scraped 100 book items spanning the first five catalog pagination pages.
- Extracted 9 mandatory attributes: title, category, price, star rating, stock availability, product description, UPC, review count, and product URL.
- Preprocessed the dataset by standardizing strings, casting numeric values, removing duplicates by UPC, handling missing descriptions, and -engineering novel features (price_band, value_score, recommended).
- Generated a 4-panel visual dashboard and constructed a term-frequency word cloud from product descriptions.

---

## Observations & Insights

1. A dataset of 100 records was successfully ingested across 5 catalog pages. Data verification confirmed zero duplicate UPC entries and full field completion following null-value handling.

2. The Price vs. Rating regression analysis demonstrates no statistical correlation between cost and user rating. Higher-priced books do not consistently yield higher star ratings.

3. The Average Price by Category metric highlights strong variations across genres, with technical and specialized categories holding higher mean costs than broader categories like Poetry or Children's books.

4. Evaluating the engineered Value Score ($\text{Rating} \div \text{Price}$) proves that low-cost books with 4–5 star ratings offer far superior value compared to high-cost alternatives, showing that quality recommendations exist in budget price tiers.

5. Insight: Inventory stock is consistently distributed across all price points. Because customer review text is not provided by the target site, text mining was conducted on publisher descriptions, reflecting marketing terms rather than consumer feedback.

---

## Conclusion

This project illustrates a complete workflow for web scraping, dataset cleansing, feature engineering, and exploratory data analysis using Python and Scrapy. The output provides quantifiable evidence regarding platform pricing models, rating behavior, and catalog distribution, establishing a reproducible ETL pipeline structure.
