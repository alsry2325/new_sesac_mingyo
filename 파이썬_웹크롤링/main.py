import requests
from bs4 import BeautifulSoup


url = "https://www.scrapethissite.com/pages/simple/"        #크롤링하고자하는 사이트
res = requests.get(url) #url에 대해 get 요청
bs = BeautifulSoup(res.text)  
#받은 응답 텍스트(HTML)를  BeautifulSoup클래스로 인자를 전달하고 bs라는 객체 생성

#h3태그에 클래스이름이 컨트리 네임인 것 하나  가져와달라
# andorra   = bs.select_one("h3.country-name")
# print(andorra.get_text(strip=True))

#하나만 가져오는 함수
andorra = bs.select_one("div.row div.col-md-4.country")
print(andorra.get_text())

#여러개 가져오는 함수
countrys = bs.select("div.row div.col-md-4.country")
print(countrys)
print(len(countrys))


for country in countrys:
    print(country)

countries = bs.select("div.row div.col-md-4")
##나라이름만 가져와라
for country in countries:
    name = country.select_one("h3.country-name").get_text(strip=True)
    country_info = country.select_one("div.country-info")
    capital = country_info.select_one("span.country-capital").get_text(strip=True)
    population = country_info.select_one("span.country-population").get_text(strip=True)
    area = country_info.select_one("span.country-area").get_text(strip=True)

    population = country_info.select_one("span.country-population").get_text(strip=True)
    print(f"{name}의 정보는 capital: {capital}\n population: {population}\n area: {area}")
