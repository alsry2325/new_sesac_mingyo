import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_csv("파이썬_데이터_시각화/seoul_park.csv")

#요일 별 시각화
#요일별 그룹화 후 숫자형 데이터에 대해 평균
# df_week = df.groupby("요일").mean(numeric_only=True)
# print(df_week)
# #인덱스에 있는 요일 정보를 새컬럼으로 추가
# df_week = df_week.reset_index()

# #데이터 정렬 높은순으로 보여쥼
# df_sorted = df_week.sort_values("청소년")


# plt.title("서울대공원 어린이 입장객 분석")
# # plt.bar(df_week["요일"], df_week["청소년"])
# # plt.barh(df_week["요일"], df_sorted["청소년"])
# plt.xlabel("요일")
# plt.ylabel("어린이 입장객수")


# #seaborn으로 막대 그래프
# # sns.barplot(data=df,x="요일",y="청소년",errorbar=None)


# #그룹으로 묶기 : hue 매개변수
# sns.barplot(data=df,x="날씨",y="어린이",errorbar=None,hue="공휴일")


#파이차트 그리기

# data_2019 = df[df["연"] == 2019] [["어른","청소년","어린이"]].sum()
# #series 형태

# labels = data_2019.index.tolist()
# #인덱스를 가져와서 리스트 형태로 반환 [["어른","청소년","어린이"]]
# data = data_2019.tolist()
# #값을 가져와서 리스트 형태로 반환

# #파이 차트 함수: pie()
# plt.pie(x=data,labels=labels, autopct="%.1f%%")
# plt.legend(loc="lower left")
# #autopct: 가 범주의 실제 비율 값을 포맷 문자열 형식으로 출력
# #%: 포맷 문자열의 시작을 알림
# #.1f: 소수점 첫째자리까지 반올림
# #%%: 진짜 % 문자 그자체 처리(이스케이프 처리)
# plt.show()




#히스토그램 그리기
#토요일 데이터: "요일" 컬럼의 값이 토에 해당하는 데이터 추출

# df_sat = df[df["요일"] == "토"]
# #maplotlib으로 어린이 컬럼의 각 구간 빈도
# plt.hist(df_sat["어린이"], bins=10)
# #bins: 구간의 개수(=막대의 개수) 설정


# #seaborn으로 어른 컬럼의 각 구간의 빈도
# sns.histplot(data=df_sat,x="어른")


# df_feb=df[df['월']==2]
# sns.histplot(
#     data=df_feb, x='어린이',
#     hue='연',  #세부적 그룹화
#     binrange=(0,1000), # 구간범위 설정
#     bins=10, # 구간의 개수(막대의 개수)
#     multiple='dodge', #여러 그래프를 겹치지않게 
#     shrink=0.8  # 막대의 너비 조절 원래의 80%
# )

#2월 데이터에서 어린이 데이터의 구간마다의 빈도수를 연도별로 확인
# plt.show()


#2017년 4월의 어른 이용객의 히스토그램을 그리세요!
#구간은 20개로
# result = df[(df["연"] == 2017)&(df["월"] == 4)]

# sns.histplot(
#     data = result,
#     x="어른",
#     bins=20
# )


sns.countplot(data=df,x="날씨",hue="공휴일")
plt.show()