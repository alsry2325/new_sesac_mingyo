
import time
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
all = []
url = f"https://www.scrapethissite.com/pages/forms/"
driver.get(url)

# 현재 페이지 HTML 가져오기
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
pages = soup.select(".pagination li")
maxpage = len(pages)
startpage = 1
    # BeautifulSoup으로 데이터 추출
for page in range(startpage, maxpage):

    url = f"https://www.scrapethissite.com/pages/forms/?page_num={page}"
    driver.get(url)
    #3. 로딩 완료 대기
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID,"hockey"))
    )
    # #4. 현재 페이지 HTML 가져오기
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

   

    # 5. BeautifulSoup으로 데이터 추출

    tables = soup.select(".table  .team")

    for table in tables:
        name = table.select_one(".name").get_text(strip=True)
        year = table.select_one(".year").get_text(strip=True)
        wins = table.select_one(".wins").get_text(strip=True)
        losses = table.select_one(".losses").get_text(strip=True)
        ot_losses = table.select_one(".ot-losses").get_text(strip=True)

        # 🟢 win (성공/실패 색상 구분)
        win_tag = table.select_one(".pct")
        if win_tag:
            win_class = win_tag.get("class", [])
            if "text-success" in win_class:
                win = f"{win_tag.get_text(strip=True)}(success)"
            elif "text-danger" in win_class:
                win = f"{win_tag.get_text(strip=True)}(danger)"
            else:
                win = win_tag.get_text(strip=True)
        else:
            win = None

        goalsfor = table.select_one(".gf").get_text(strip=True)
        goals_against = table.select_one(".ga").get_text(strip=True)

        # 🟢 diff (성공/실패 색상 구분)
        diff_tag = table.select_one(".diff")
        if diff_tag:
            diff_class = diff_tag.get("class", [])
            if "text-success" in diff_class:
                diff = f"{diff_tag.get_text(strip=True)}(success)"
            elif "text-danger" in diff_class:
                diff = f"{diff_tag.get_text(strip=True)}(danger)"
            else:
                diff = diff_tag.get_text(strip=True)
        else:
            diff = None
        
        all.append({
            "name": name,
            "year": year,
            "wins": wins,
            "losses": losses,
            "ot_losses": ot_losses,
            "win": win,
            "goalsfor": goalsfor,
            "goals_against": goals_against,
            "diff": diff
        })
    print(f"{page}페이지 크롤링 중 입니다")
        
print(json.dumps(all, ensure_ascii=False, indent=4))
print(len(all))