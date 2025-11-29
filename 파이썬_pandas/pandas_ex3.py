import pandas as pd
from tabulate import tabulate
df = pd.read_csv("파이썬_pandas/seoul_park04.csv")


#비교연산자를 이용한 조건이 참인 데이터 행 추출
                # df[조건식]
# grouped = df[df["어른"]>20000] # 반드시 시리즈 형태로 추출후 비교해야한다
# print(tabulate(grouped, headers='keys', tablefmt='psql'))

print("=" *100)
# df["어른" > 1000] x 
#df["어른"] > 1000  x 


#논리연산자를 이용한 조건이 참인 데이터 행 추출

# df[("명제1") | ("명제 2")]
# result = df[(df["어른"]> 10000) & (df["어린이"] > 1000)] 
# print(result)


#Q1. 어른이 10000 초과이고 , 공휴일인 데이터를 조회
print("Q1. 어른이 10000 초과이고 , 공휴일인 데이터를 조회")
q1 = df[(df["어른"] > 10000) & (df["공휴일"] == "O")]
print(q1)
print("="*100)
#Q2. 어른이 10000 초과이거나, 어린이가 2000 초과인 데이터를 조회
print("Q2. 어른이 10000 초과이거나, 어린이가 2000 초과인 데이터를 조회")
q2 = df[(df["어른"] > 10000) | (df["어린이"] > 2000)]
print(q2)
print("="*100)
#Q3. 5월 5일(어린이날) 데이터를 조회
print("Q3. 5월 5일(어린이날) 데이터를 조회")
q3 = df[(df["월"] == 5) & (df["일"] == 5)]
print(q3)
print("="*100)

'''
데이터 정렬하기
'''
print("데이터 정렬하기")
test = df.sort_values("총입장객수")
print(df.sort_values("총입장객수")) # 오름차순 정렬
print(df.sort_values("총입장객수",ascending=False)) # 내림차순 정렬

'''
인덱스 재지정하기
'''
print("인덱스 재지정하기")
print(test.reset_index())


'''
데이터 삭제하기
'''
# print("데이터 삭제하기")
# test = df.drop(["총입장객수"],axis=1) #열 삭제
# print(test)

'''
컬럼이름 변경
'''
df = df.rename(columns={"총입장객수":"총계"}) #inplace=True를 하면 매개변수를 통해 바로 즉시 적용

print(df)


'''
결측치 처리
'''
print(df.shape)
print(df.isna().sum())
# df["청소년"]   = df["청소년"].fillna(int(df["청소년"].mean()))
df.dropna(subset=["청소년"],ignore_index=True,inplace=True)
print(df.shape)
print(df["청소년"])


# print(df.loc["인덱스이름","컬럼이름"]) # 대괄호
print(df.loc[(df["어른"]>1000)|(df["어린이"]>1000),["날짜","공휴일","어른","어린이"]])


print("Q4. 인덱스가 3이고 , 컬럼이 어른이 데이터 조회")
q4 = df.loc[3,["어른"]]
print(q4)
print("Q5. 인덱스가 3부터 6이고, 컬럼이 어른부터 외국인 데이터 조회")
q5 = df.loc[:6,"어른":"외국인"]
print(q5)
print("Q6. 어른이 1000 초과이고, 어린이가 1000 초과 인 행에서 날짜부터 총계 조회")
q6 = df.loc[(df["어른"] > 1000)&(df["어린이"] > 1000),"날짜":"요일"]
print(q6)

#예제 1 학생 성적 데이터
data = {
    "이름": ["철수", "영희", "민수", "지영", "수현", "지훈"],
    "학년": [1, 1, 2, 2, 3, 3],
    "국어": [80, 95, 70, 88, 90, 85],
    "영어": [85, 90, 75, 95, 92, 80],
    "수학": [78, 85, 68, 90, 95, 82]
}



df1 = pd.DataFrame(data)
result1 = df1[df1["영어"] > df1["국어"]]
print("영어 점수가 국어 점수보다 높은 학생")
print(result1)
print("="*100)
result2 = df1[(df1["학년"] >= 2) & (df1["영어"] >= 90)]
print("학년이 2학년 이상이면서 영어 점수가 90점 이상인 학생")
print(result2)
print("="*100)
result3 = df1[(df1["국어"]< 80)|(df1["수학"]< 80)]
print("국어가 80미만이거나 수학이 80 미만인 학생")
print(result3)
print("="*100)
result4 = df1[df1["이름"].str.contains("지")]
print("이름에 지 가 포함된 학생")
print(result4)
print("="*100)

data = {
    "이름": ["지민", "서준", "하늘", "민재", "예린", "도윤"],
    "국어": [88, 95, 76, 84, 92, 60],
    "수학": [90, 85, 70, 95, 98, 75],
    "영어": [82, 93, 65, 89, 91, 72],
    "출석률": [0.95, 0.90, 0.85, 1.00, 0.98, 0.70]
}   
df2  = pd.DataFrame(data)
print(df2)
print("Q1. 출석률이 0.9미만인 학생의 영어 점수 출력 :")
result5 = df2.loc[df2["출석률"] < 0.9,"영어"]
print(result5)

print("Q2. 2~4번째 행, 1~3번째 열데이터 출력 :")
result6 = df2.iloc[[2,4],[1,3]]

print(result6)
print("Q3. 상위 3명의 국어,수학 점수를 100점으로 변셩")

# df2.loc[:3,["국어","수학"]] = 100
df2.iloc[:3,[1,2]] = 100
print(df2) 

print("Q4. 마지막 학생의 모든 점수를 10점 올리기")
df2.iloc[-1,1:4] +=10
print(df2)

print("Q5. 국어가 90점 이상이고, 수학이 90이상인  학생의 영어 점수를 95, 출석률 1.0으로 변경")
df2.loc[(df2["국어"]>= 90)&(df2["수학"]>=90),["영어","출석률"]] = [95,1.0]
print(df2)
 
print("출석률이 0.8 미만인 학생의 영어 점수를 기존보다0.9배로 조정")
df2.loc[(df2["출석률"] < 0.8),"영어"] *= 0.9
print(df2)

df = pd.read_csv("파이썬_pandas/seoul_park04.csv")
df2 = pd.read_csv("파이썬_pandas/seoul_park_april.csv")
mm = pd.read_csv("파이썬_pandas/misemunji.csv")
print(df.head())
print(df2.head())

df_new = pd.concat([df,df2],axis=0,join="inner",ignore_index=True)
print(df_new.head())
print(df_new.shape)
print(df_new.columns)

print(df.head())
print(mm.head())
test1   = pd.merge(df,mm, on="날짜", how="inner") # df라는 데이터 프레임에 합치겠다 
#입장객 데이터에서 공통으로 있는 날짜
#미세먼지 데이터에서 공통으로 있는 날짜
#입장객 데이터를 기준으로 입장객 데이터의 날짜 기준으로 데이터를 받을수있다?
#입장객 데이터의 일부 날짜가 아예 날라가는것보다는 있는 상태로 두고,미세먼지


#예제2
data = {
    "지점" : ["서울","서울","부산","부산","대구","대구"],
    "제품" : ["TV",]
}

