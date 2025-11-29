import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df= pd.read_csv("파이썬_데이터_시각화/seoul_park.csv")
#2018년 4월 데이터를 가져오겠다
df_2018_april = df[(df["연"] == 2018 )&(df["월"] == 4)]
print(df_2018_april["어린이"].info())
# plt.boxplot(df_2018_april["어린이"])

# plt.boxplot([
#     df_2018_april["어린이"],
#     df_2018_april["어른"],
#     df_2018_april["청소년"]])
# plt.xticks([1,2,3], labels=["어린이","어른","청소년"])
# plt.title("2018년 4월의 입장객수")

# df_2018 = df[df["연"] == 2018]

# sns.boxenplot(data=df_2018,y="어린이", x="월")


# df_2016 = df[df["연"] == 2016]
# sns.boxenplot(data=df_2016, x="월",y="어른")

# # df_2016["어른"]
# print(df_2016["어른"])

#산점도 그리기
# sns.scatterplot(data=df,x="어른",y="어린이")
# #이상치 때문에  범위가 극단적으로 나옴

# #xlim과 ylim으로 범위 제한
# plt.xlim(0,35000)
# plt.ylim(0,6000)

# #x축은 어른 y축은 청소년
# plt.scatter(df["어른"],df["청소년"])

#어른과 청소년의 관계
df_corr = df.corr(numeric_only=True)
print(df_corr)

sns.heatmap(df_corr,annot=True,fmt=".2f")
#annot=True: 실제 값도 히트맵에
#fmt: 출력되는 값의 형식 지정
plt.show()

 