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

login_btn = driver.find_element(By.CLASS_NAME, "MyView-module__link_login___HpHMW")
login_btn.click()

id = driver.find_element(By.ID, "id")
id.send_keys("a")
time.sleep(1)

pw = driver.find_element(By.ID, "pw")
pw.send_keys("16!")
time.sleep(1)

login_btn2 = driver.find_element(By.ID, "log.login")
login_btn2.click()

# pw.send_keys(Keys.ENTER)