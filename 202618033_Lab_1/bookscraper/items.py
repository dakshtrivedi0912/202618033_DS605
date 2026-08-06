import scrapy

class BookItem(scrapy.Item):
    title = scrapy.Field()
    category = scrapy.Field()
    price = scrapy.Field()
    rating = scrapy.Field()
    availability = scrapy.Field()
    product_description = scrapy.Field()
    upc = scrapy.Field()
    number_of_reviews = scrapy.Field()
    product_url = scrapy.Field()