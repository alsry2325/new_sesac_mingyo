import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By  # By.Id 쓰려면 내장라이브러리 설치해야함
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


#1. 브라우저 실행
options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

url = f"https://books.toscrape.com/index.html"
driver.get(url)

WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME,"page_inner"))
        )

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

# ---1번째 페이지---
books = soup.select("section .row li")

for book in books:
    img = book.select_one("a")["href"]
    stars_tag  =  book.select_one("p.star-rating")
    stars_class = stars_tag.get("class", [])[1]
    if "Five" in stars_class:
            stars = 5
    elif "Four" in stars_class:
            stars = 4
    elif "Three" in stars_class:
            stars = 3
    elif "Two" in stars_class:
            stars = 2
    elif "One" in stars_class:
            stars = 1               
    else:
        stars = None
    name = book.select_one("h3 a")["title"]
    urldetail = book.select_one("h3 a")["href"]
    price = book.select_one("div .price_color").get_text(strip=True)
    
    print(img)
    print(stars)
    print(name)
    print(urldetail)
    print(price)

pages = soup.select(".pager .next a")["href"]


#---2번째 페이지 ---
url = f"https://books.toscrape.com/{pages}.html"
page2 = BeautifulSoup

books = soup.select("section .row li")

for book in books:
    img = book.select_one("a")["href"]
    stars_tag  =  book.select_one("p.star-rating")
    stars_class = stars_tag.get("class", [])[1]
    if "Five" in stars_class:
            stars = 5
    elif "Four" in stars_class:
            stars = 4
    elif "Three" in stars_class:
            stars = 3
    elif "Two" in stars_class:
            stars = 2
    elif "One" in stars_class:
            stars = 1               
    else:
        stars = None
    name = book.select_one("h3 a")["title"]
    urldetail = book.select_one("h3 a")["href"]
    price = book.select_one("div .price_color").get_text(strip=True)
    
    print(img)
    print(stars)
    print(name)
    print(urldetail)
    print(price)