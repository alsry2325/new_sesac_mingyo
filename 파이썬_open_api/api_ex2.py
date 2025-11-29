import requests
import json

base_url = "https://api.openweathermap.org/data/2.5/weather"

params = {
    "lat"  :"37.583328",
    "lon"   :"127.0",
    "appid" : "78ef33b244ef86476adf6e31367d646c",
    # "lang" : "kr"
}

response = requests.get(base_url,params=params)
if response.status_code == 200:
    data = response.json()
    weather = data["weather"][0]["description"]

print(json.dumps(data, ensure_ascii=False, indent=4))

