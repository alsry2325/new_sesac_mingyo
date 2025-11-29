import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By  # By.Id 쓰려면 내장라이브러리 설치해야함
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

"""
현재 온도
현재 날씨 상태(맑음, 흐림, 비 등등의 키워드)
미세먼지 정도
초미세먼지 정도
자외선 정도
현재 날씨 상태에 대한 이미지 URL
이미지 URL의 경우 코드 크롤링하는 것이 아닌 개발자 도구에서 Styles 항목에 있는 URL을 코드로 하드코딩하면 됩니다.
단, 날씨 상태마다 이미지가 달라지게끔 패턴 파악으로 URL을 수정해야합니다.
"""


#1. 브라우저 실행
options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

url = f"https://www.naver.com"
driver.get(url)

WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME,"body"))
    )

query = driver.find_element(By.ID, "query")
query.send_keys("서울시 용산구 날씨")
time.sleep(1)
search = driver.find_element(By.ID, "search-btn")
search.click()


#3. 로딩 완료 대기
WebDriverWait(driver, 5).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR,"div.status_wrap"))
)

#4. 현재 페이지 HTML 가져오기
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

#BeautifulSoup으로 데이터 추출
today_weathers = soup.select_one("div.status_wrap")
today= today_weathers.select_one("div._today div.temperature_text strong").get_text(strip=True)
ct = today.replace("현재 온도", "").replace("°", "").strip()

weather = today_weathers.select_one("div.temperature_info span.weather.before_slash").get_text(strip=True)

fine_dust = today_weathers.select_one("div.report_card_wrap a span.txt").get_text(strip=True)

ultrafine_dust = today_weathers.select_one("div.report_card_wrap li:nth-child(2) span.txt").get_text(strip=True)

ultraviolet_rays = today_weathers.select_one("div.report_card_wrap li:nth-child(3) span.txt").get_text(strip=True)

# 현재 날씨 상태에 대한 이미지 url
img_name = today_weathers.select_one("div.weather_main i").get("class")[1]
img_name =  img_name.split("_")[1]
img_url = f"../img/weather_svg_v2/icon_flat_{img_name}.svg"

# 현재 날씨 상태에 대한 이미지 url
# icon = soup.find("div", class_="weather_main").select_one("i").get("class")
# for c in icon:
#     if c.startswith("ico_wt"):
#         code = c
# img_url = f"../img/weather_svg_v2/icon_flat_{code[4:]}.svg"

print("현재 온도:", ct)
print("현재 날씨 상태 :",weather)
print("미세먼지 정도 :",fine_dust)
print("초미세먼지 정도 :",ultrafine_dust)
print("자외선 정도 :",ultraviolet_rays)
print("현재 날씨 상태에 대한 이미지 URL :",img_url)

driver.quit() #quit()  실행완료되면 꺼지는 메소드