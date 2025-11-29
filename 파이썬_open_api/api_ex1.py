import requests

base_url = "https://restcountries.com/v3.1"

path = "/all"

solo = "/alpha"

url = base_url + solo

params = {
    # "fields" : "name,population,area",
    "codes" : "KR,JP,USA"
 }
response = requests.get(url=url, params=params)
#https://restcountries.com/v3.1?/all?fields=name,population
if response.status_code == 200: #요청에 성공했을때
    data = response.json()   # 응답 바디를 딕셔너리 형태로 변환
    print(data) #딕셔너리를 원소로 갖는 리스트
    for country in data:   
        print("=" * 20)
        name = country["name"]["common"]
        tid = country["tld"][0]
        map = country["maps"]["googleMaps"]
        populatiion = country["population"]
        area = country["area"]
        capital = country["capital"]
        translations = country.get("translations",{}).get("kor",{}).get("common","해당없음")
        print(f"나라 이름은:{name}, 인구수 :{populatiion} 면적은 {area}")
        print(f"나라 수도는:{capital}, tid :{tid} 나라위치 url: {map} 한국 발음:{translations}")
        
else: #요청에 실패했을때
    print(response.text)