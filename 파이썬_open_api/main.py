from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By  # By.Id 쓰려면 내장라이브러리 설치해야함
from selenium.webdriver.common.keys import Keys
import time


options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)


url = "https://www.naver.com"
driver.get(url)
time.sleep(2)  # URL을 매개변수로 호출하고서  이초후 열리게끔

#검색어 입력
query = driver.find_element(By.ID, "query")
query.send_keys("파이썬")
time.sleep(1)
#엔터 입력
query.send_keys(Keys.ENTER)


#검색어 버튼 클릭
search_btn = driver.find_element(By.CSS_SELECTOR, "#search-btn") 
# find_element(By.ID, "#search-btn") id속성이 “search-btn”인 요소 객체 반환 후,
time.sleep(1)
# search_btn.click() #click() 메서드로 클릭 수행