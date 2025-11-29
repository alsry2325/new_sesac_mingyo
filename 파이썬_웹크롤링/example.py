import pprint
import requests
from bs4 import BeautifulSoup


url = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops"        #크롤링하고자하는 사이트
res = requests.get(url) #url에 대해 get 요청
bs = BeautifulSoup(res.text)  

products = bs.select("div.row div.col-md-4.col-xl-4.col-lg-4 div.card.thumbnail div.product-wrapper.card-body")


for product in products:
    
    images = product.find("img",class_="img-fluid card-img-top image img-responsive")["src"]    
    # tets1  = product.select_one("div.caption h4 a.title")['title']
    names = product.select_one("div.caption h4 a").get_text(strip=True)
    prices = product.select_one("div.caption h4").get_text(strip=True)
    details = product.select_one("div.caption p").get_text(strip=True)
    reviews =  product.select_one("div.ratings p.review-count.float-end span").get_text(strip=True)
    stars =  product.select_one("div.ratings :nth-child(2)")["data-rating"]
    productdetail = product.select_one("div.caption h4 a")["href"]
    
    alls = {
        "image":images,
        "name":names,
        "price":prices,
        "detail":details,
        "review":reviews,
        "star":stars,
        "productdetail":productdetail
    } 

    print(f"이미지: {images}\n 이름: {names}\n 가격: {prices}\n 상세 스펙 텍스트: {details}\n 리뷰: {reviews}\n 별: {stars}\n 상품 상세링크: {productdetail}\n")
    print(alls)