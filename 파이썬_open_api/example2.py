from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By  # By.Id 쓰려면 내장라이브러리 설치해야함
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

#1. 브라우저 실행
options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

url = "https://quotes.toscrape.com/scroll"
driver.get(url)

#2.무한 스크롤
last_height   = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

#3. 로딩 완료 대기
WebDriverWait(driver, 5).until(
    EC.presence_of_all_elements_located((By.CLASS_NAME,"quote"))
)

#4. 현재 페이지 HTML 가져오기
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

#5. BeautifulSoup으로 데이터 추출
quotes = soup.select(".quote")

for q in quotes[:5]:
    text = q.select_one(".text").get_text(strip=True)
    author = q.select_one(".author").get_text(strip=True)
    print(f"{text} - {author}")

driver.quit() #quit()  실행완료되면 꺼지는 메소드