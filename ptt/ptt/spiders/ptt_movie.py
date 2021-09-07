# coding=utf-8
import csv
import re
import scrapy
from scrapy.crawler import CrawlerProcess
import logging
from ptt.ptt.items import PttItem

logging.getLogger('scrapy').propagate = False


class Movie(scrapy.Spider):
    name = "movie"

    def __init__(self):
        self.items = PttItem()
        self.outfile = open(f"../../../crawler_data/{self.name}.csv", "w", newline="")
        self.writer = csv.writer(self.outfile)
        self.writer.writerow(['main_content', 'comment', 'sentiment'])

        self.start_page = 2900
        self.pages = 100
        self.comment_num_each_post = 5
        self.min_comment_word = 5

    def start_requests(self):
        urls = []
        for i in range(self.pages):
            urls.append(f'https://www.ptt.cc/bbs/movie/index{self.start_page - i}.html')
        for url in urls:
            yield scrapy.Request(url=url, cookies={"over18": "yes"}, callback=self.parse)

    def parse(self, response):
        root_url = 'https://www.ptt.cc/'
        titles = response.css('div.r-ent div.title a::text').getall()
        dates = response.css('div.r-ent div.meta div.date::text').getall()
        author = response.css('div.r-ent div.meta div.author::text').getall()
        page_urls = response.css('div.r-ent div.title a::attr(href)').getall()
        page_urls = [root_url + url for url in page_urls]

        for title, url in zip(titles, page_urls):
            if '雷' in title:
                request = scrapy.Request(str(url), callback=self.parse_content)
                yield request

    def parse_content(self, response):
        content = response.css('div#main-content::text').get()
        push = response.css('div.push span.f3.push-content::text').getall()
        if len(push) > 0 and self.clear_text(content):
            content = self.clear_text(content)
            counter = 0
            for comment in push:
                comment = self.clear_text(comment).replace(':', '')
                if len(comment) > self.min_comment_word:
                    self.writer.writerow([content, comment, ''])
                    counter += 1
                    if counter == self.comment_num_each_post:
                        break

    def clear_text(self, text):
        text = re.sub(r'[\s-]', '', text)
        return text


if __name__ == '__main__':
    process = CrawlerProcess()
    process.crawl(Movie)
    process.start()

# 負面：對發文者批評或對相關事務做出負面評價
# 正面：對發文者表揚或者對相關事務表現出正面得情緒
# 中立：與發文內容無關的評論或者只是客觀描述，未做個人評價
