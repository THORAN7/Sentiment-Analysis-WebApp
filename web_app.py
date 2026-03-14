import undetected_chromedriver.v2 as uc
from bs4 import BeautifulSoup
import time

# Amazon Scraper class
class AmazonScraper:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode

    def scrape(self, url):
        if self.demo_mode:
            print("Demo mode enabled. Simulating scraping...")
            return self.simulated_scrape(url)

        options = uc.ChromeOptions()
        # Add your Chrome options here
        driver = uc.Chrome(options=options)
        driver.get(url)

        time.sleep(5)  # Adjust the sleep time as necessary
        page_source = driver.page_source
        driver.quit()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')
        product_data = self.parse_product_data(soup)
        return product_data

    def simulated_scrape(self, url):
        # Sample data to return in demo mode
        return {"title": "Sample Product", "price": "$99.99", "url": url}

    def parse_product_data(self, soup):
        # Your parsing logic here (modify as per the webpage structure)
        title = soup.find('span', {'id': 'productTitle'}).get_text(strip=True)
        price = soup.find('span', {'id': 'priceblock_ourprice'}).get_text(strip=True)
        return {"title": title, "price": price}

# Example usage
if __name__ == '__main__':
    url = 'https://www.amazon.com/dp/B09XYZ'
    scraper = AmazonScraper(demo_mode=True)
    product_info = scraper.scrape(url)
    print(product_info)