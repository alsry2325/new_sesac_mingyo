import pandas as pd

#딕셔너리로 생성하는 법
data = {

    "이름" : ["민교","영희","철수"],
    "나이" : [20,50,30],
    "도시" : ["서울", "부산", "대구"],
    "점수" : [90,50,40],

}
df = pd.DataFrame(data)
print(df)
#인덱스도 바꿀수있다
df = pd.DataFrame(data,index=["a","b","c"])
print(df)

print("="*50)

#2차원리스트로 생성(중첩 리스트) : 값이 없으면 none값으로 뜸

data = [

     ["민교","영희","철수"],
        [20,50,46],
     ["서울", "부산", "대구"],
     [90,50,40],

]

df = pd.DataFrame(data, columns=["이름","나이","도시"])

print(df)

print("="*30)


#외부 데이터로 생성

