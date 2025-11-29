import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

year = [2018, 2019, 2020, 2021, 2022, 2023]
python = [5.8, 8.17, 9.31, 11.87, 15.63, 14.51]
C = [13.59, 14.08, 16.72, 14.32, 12.71, 14.73]
java = [15.78, 15.04, 16.73, 11.23, 10.82, 13.23]
JS = [3.49, 2.51, 2.38, 2.44, 2.12, 2.1]

#선 그래프: plot()함수  : 연도에 따른 파이썬 점유율 변화

# plt.plot(year,python)
# plt.show() # 실제화면에  결과띄우기


# plt.title("파이썬 언어 점유율",fontsize=20)

# plt.xlabel("연도",fontsize=20)
# plt.ylabel("점유율",fontsize=20)


#3개의 선그래프
# plt.plot(year,python)
# plt.plot(year,C)
# plt.plot(year,java)   

# plt.legend(["Python","C","JAVA"])  # 범례를 표시해주는 함수 뭘 의미하는지 알려주는 설명상자
#하지만 위의 방법은 순서를 제대로 지정하지않으면 왜곡이 발생함 그래서
#아래방법을 선호
# plt.plot(year,python, label="Python")
# plt.plot(year,C, label="C")
# plt.plot(year,java, label="JAVA") 

# plt.legend()


#여러 개의 피규어 사용하기
# fig1 = plt.figure(1, figsize=(4,3)) #이 아래서 그려지는 요소는  figure 1번에 그려짐
# plt.plot(year,python, label="Python")
# plt.title("Figure 1")

# fig2 = plt.figure(2, figsize=(4,3))  # 이 아래서부터 그려지는 구성요소는 figure 2번에 그려짐
# plt.plot(year,python, label="Java")
# plt.title("Figure 2")
# plt.legend()

# fig2 = plt.figure(1) # 현재 도화지를  1번도화지로 활성화  이아래서부터 그려지는 것은 fig 1번도화지에 그려짐
# plt.plot(year,python, label="Java")
# plt.legend()

#하나의 도화지 (gigure) 안에서 공간을 여러개로 나누기 
'''
fig,(ax1,ax2) = plt.subplots(2,1, figsize=(10,6))#행,열
ax1.plot(year,python) #왼쪽공간
ax2.plot(year,java) # 오른쪽 공간
'''

#subplots() 생성 여러 케이스
''''
fig,(ax1,ax2) = plt.subplots(2,1,constrained_layout=True)#행,열
ax1.plot(year,python) #왼쪽공간
ax1.set_title("파이썬점유율") # 오른쪽 공간
ax1.set_xlabel("연도")
ax1.set_ylabel("점유율")

ax2.plot(year,python) #왼쪽공간
ax2.set_title("자바 점유율")
ax2.set_xlabel("연도")
ax2.set_ylabel("점유율")
'''

# fig,(ax1,ax2,ax3) = plt.subplots(1,3,constrained_layout=True)#행,열
# ax1.plot(year,python) #왼쪽공간
# ax1.set_title("파이썬점유율") # 오른쪽 공간
# ax1.set_xlabel("연도")
# ax1.set_ylabel("점유율")

# ax2.plot(year,python) #왼쪽공간
# ax2.set_title("자바 점유율")
# ax2.set_xlabel("연도")

# ax3.plot(year,python) #왼쪽공간
# ax3.set_title("c언어 점유율")
# ax3.set_xlabel("연도")


#seaborn  선 그래프
 
# sns.lineplot(x=year,y=python ,label ="py")
# sns.lineplot(x=year,y=java ,label = "Java")


plt.show()   # 실제 화면에 구현 실행
             
             

             
             
             
             
     