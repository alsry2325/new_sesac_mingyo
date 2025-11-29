import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_csv("파이썬_데이터_시각화/seoul_park.csv")

# # 월 컬럼으로 그룹화 진행
# group_month = df.groupby("월")
# #그룹마다 수치형 컬럼에 대해 평균을 구함(월별 수치형 컬럼들의 평균) 
# df_mon = group_month.mean(numeric_only=True)
# print(df_mon)                                           

# plt.title("서울대공원 입장객 분석")
# plt.plot(df_mon["어른"], label = "어른", marker=".", linestyle="-", color="red") #월별 어른 입장객 수의 평균
# plt.plot(df_mon["어린이"],label="어린이", marker="o", linestyle="--", color="blue") # 월별 어린이 입장객 수의 평균
# plt.plot(df_mon["청소년"],label ="청소년", marker="^", linestyle=":", color="red" ) #월별 청소년 입장객 수의 평균
plt.xlabel("월")
plt.ylabel("입장객수")
plt.xticks(range(1,13)) #x축 눈금 조절

# #범례 추가
plt.legend()
plt.grid(True,axis="x") #격자 추가 (x축에 대한; 세로 추가)
# plt.show() 


#data: 대상의 데이터프레임
#Seaborn 방식
# x:  그 데이터프레임에서의 x축에 넣을 컬럼의 이름  errorbar=None: 자동으로 나오는 오차 영역/막대 등 삭 제
sns.lineplot(data=df, x="월",y="어른",errorbar=None)

plt.show()