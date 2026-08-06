import scrapy
from bookscraper.items import BookItem

class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["http://books.toscrape.com/catalogue/page-1.html"]

    page_count = 1  # Track page limit

    custom_settings = {
        'CONCURRENT_REQUESTS': 8,
        'DOWNLOAD_DELAY': 0.25,
        'CLOSESPIDER_ITEMCOUNT': 100,  # Stops spider automatically after 100 items
        'FEEDS': {
            'scraped_books.json': {
                'format': 'json',
                'encoding': 'utf8',
                'indent': 2,
            },
            'scraped_books.csv': {
                'format': 'csv',
                'encoding': 'utf8',
            },
        }
    }

    def parse(self, response):
        # Extract book detail links (20 books per catalog page)
        book_links = response.css('article.product_pod h3 a::attr(href)').getall()
        for link in book_links:
            yield response.follow(link, callback=self.parse_book)

        # Pagination: Stop strictly after 5 catalog pages (5 pages x 20 items = 100 books)
        if self.page_count < 5:
            self.page_count += 1
            next_page = response.css('li.next a::attr(href)').get()
            if next_page:
                yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):
        item = BookItem()
        
        item['title'] = response.css('div.product_main h1::text').get()
        item['category'] = response.css('ul.breadcrumb li:nth-child(3) a::text').get()
        item['price'] = response.css('p.price_color::text').get()
        
        rating_classes = response.css('p.star-rating::attr(class)').get()
        item['rating'] = rating_classes.replace('star-rating', '').strip() if rating_classes else None
        
        # Fixed: Removed the trailing '3' after .getall()
        avail_text = response.css('p.instock.availability::text').getall()
        item['availability'] = "".join(avail_text).strip() if avail_text else None
        
        item['product_description'] = response.css('#product_description + p::text').get()
        
        table_rows = response.css('table.table-striped tr')
        table_data = {}
        for row in table_rows:
            header = row.css('th::text').get()
            value = row.css('td::text').get()
            if header and value:
                table_data[header.strip()] = value.strip()
                
        item['upc'] = table_data.get('UPC')
        item['number_of_reviews'] = table_data.get('Number of reviews')
        item['product_url'] = response.url

        yield item