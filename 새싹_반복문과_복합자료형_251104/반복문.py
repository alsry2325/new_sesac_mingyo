#반복문
#조건이나 범위내에서 어떠한 명령을 반복적으로 수행하는것

#반복문의 종류

#for문

#시퀀스에서 원소를 하나씩 가져와서 명령을 실행  

''' 
for 변수 in 시퀀스:
    <수행할 명령>
    <수행할 명령>
'''
#1.시퀀스에서 앞에서  원소를 하나씩 꺼냄
#2.변수에 원소를 저장
#3. 수행할 명령 수행
#4.다음 원소를 꺼내서 시퀀스의 원소가 다 꺼내질 때까지 수행함

total = 0               
for i in [1,3,5]:   #1.첫번째 원소를 꺼내고 , 변수i에 저장
    total+=i        #2.명령 실행  3.다시 위로 이동
print(total) 

#for문의 반복 횟수

count = 0 
for i in [1,2,3,4,5]:   #i에 원소 하나씩 할당한다
    count+=1            #i에 원소 하나를 할당할때마다 +1를 해라
print(count)

prices = [10000,25000,80000,150000]
discount_ratio = 0.9

for price in prices:
    result = discount_ratio*price
    print(result)

a =[(1,3),(4,2),(5,3)] 
for (first,last) in a:
    result =  first+last
    print(result)

#range()함수
#연속되는 숫자를 만들어주는 함수
#range(start,end,step)
#시작숫자(a)와 끝 숫자(b)를 지정하면 a~ (b-1) 까지 숫자 생성

#for-range문
#n회동안 명령을 실행해라

nuum_list = [1,23,4,5,3]
for i in range(1,10):
    print(i)
for i in range(5,10): #b - a번 반복
    print(i) #5,6,7,8,9

count = 0
for i in range(5): #숫자만큼만 반복
    print(i)    #0,1,2,3,4

for i in range(5,10,2): #b - a번 반복
    print(i)    #5,7,9

nuum_list = [1,23,45,1223,1232]

#구간 반복
for i in range(len(nuum_list)):
    #len(num_list) == 5
    #range(5) 0부터 5미만까지 
    #각 값을 i에 저장한다
    print(i,nuum_list[i])


marks = [90, 25, 12, 45, 80]

for mark in range(len(marks)):

    if marks[mark] >= 60:
        print("합격입니다",mark+1,"번째 학생")
    else:
        print("불합격입니다",mark+1,"번째 학생")




prices = [3000,4500,10000,8000,20000]

quantities = [3,4,1,2,1]

total = 0

for i in range(len(prices)):
    total += prices[i] * quantities[i]
    print(total)


prices = [4000,25000,7500,12000,18500]
discounted = []

for i in range(len(prices)):

    if prices[i] >= 10000:
        result = prices[i]*0.9
        discounted.append(result)
    else: 
        discounted.append(prices[i])
print(int(result))


#continue키워드
#반복문에서 continue를 만나면 명령을 건너뛰고 넘어갑니다

for i in range(1,11):   #1부터 11미만까지 i라는 원소에 할당한다음
    if i % 2 == 0:  # i를 2로 나눴을때 0으로 떨어지면 건너뛰고 
        continue
    print(i)            #그러지않을때 i출력

#while문

#for문은 횟수기반이라면   while문은 조건기반으로 실행한다
# 1. 조건이 참이면 명령을 수행
#2.다시 위로 올라와서 조건 검사
#3.조건이 거짓이 될때까지 실행
i = 5
while i > 2:        #i가 2보다 크지 않을때까지 반복
    print(i)        
    i-=1            #돌때마다 -1
print("while문 종료")


balance = 10000 #초기금액
withdraw = 1500 #인출금액
 
while balance > withdraw:
    balance-=withdraw
    print("현재 잔액 :",balance)
    print("인출금액 :",withdraw)


numbers = [3,8,12,5,18,7]
target = 12
i = 0

while i < len(numbers):
    if numbers[i] == target:
        print(f"{i+1}번째에서 {target} 탐색완료")
        break
    i+= 1
    print(target,i)


#일부러 무한루프 만들기
# 일부러 조건을 항상 참으로 만들고 중간에 조건문 break통해 탈출

#while True:
 #   print("\n======쇼핑몰 메뉴=====")


    