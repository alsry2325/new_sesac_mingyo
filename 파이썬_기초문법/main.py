#문자열 자료형
print("안녕하세요")
print('이런식으로 표현한다')

#주석처리
#여러개는 '''
#한줄 주석처리

#출력은 print()
print('안녕',"이런식으로 , 콘마는 여러개 출력가능")

#이스케이프 시퀀스
'''
\n 은 줄바꿈 ,\t는 수평탭, \b는 백 스페이스 역할, \\ 문자 백 슬러시 \' 문자는 작은 따음표 \" 는 큰 따음표
'''
print("자유롭게 달리는  두 바퀴 위에서 \n전해지는 바람은 시원하고 \n거침없이 앞으로 나아간다")

#변수란?
#변할수있는 값으로, 자료형 값을 보관하는 곳

num=12
name="정민교"
print(num)
print(name)

#변수명 규칙
#의미하는 바로 표현
#snake_case사용 예시: min_gyo

#자료형연산자에는 
'''
+,-,*/ 있음

특수연산자에는
// 몫 나눗셈 연산
% 나머지 값
** 제곱연산자
'''
print(5//5) #1
print(7//2) #3
print(14%5) #4
print(2**3) #8

# %나머지 연산자는 꼭 중요함 순환하거나 주기적인 패턴에서 자주 사용함
# 시계계산, 요일계산,리스트안에 인덱스 순환,배수

hour = 23
after = 3
new_hour = (hour + after) % 24
print(new_hour)

#문자열은 이어붙이기가 가능
print("안녕"+"하셈")
#문자열 반복출력가능
print("안녕"*3)

#연산자의 확장 
total = 0
total = total + 2  #2
#줄이면 
total+=2 # 이렇게 가능 
print(total) # 결과는 4

#input()함수 
#컴퓨터에서 출력이 가능하게하는 함수
value = input("안에 입력하셈:")
print(value)
#하지만 출력하는값은 항상 문자열로 출력됨 
print(type(value))

#그래서 형변환을 해줘야함 
num1 = "123"
num2 = "44.4"

num1 = int(num1)
num2 = float(num2)

print(type(num1))
print(type(num2))

#논리 자료형
print(True)
print(False)
#타입은 bool 불리언값
#비교 연산자
'''
== 같다
< 오른쪽이 크다
> 왼쪽이 크다
>= 왼쪽이 크거나 같다
<= 오른쪽이 크거나 같다
!= 같지않다

'''
#논리 연산자
'''
and 는 그리고 둘다 참이여야 true
or는  이거나 둘중 하나만 참이면 true
not은 아니다 notTrue 이면 false

'''
print(3 == 4 and  4 <= 5 and 6 > 2 )
'''
3은 4랑 같나요? 그리고 4는 5보다 작거나 같나요? 육은 2보다 큰가요?
'''

print( 3 == 4 or 3 <= 5  or 4 < 5)
'''
3은 4랑 같거나 3은 5보다 작거나같나? 4는 5보다 크나?
''' 

#복합 연산자

result = (4+5)*2 > 12
'''
괄호 먼저 계산하고 곱하기 * 2
'''
print(result)

a=5
b=2
c=12

result= (a*b>8) and (c//b == 4)

print(result)


n = int(input("입력할 숫자의 개수를 입력하세요: "))

numbers = []  # 입력받은 숫자들을 저장할 리스트

for i in range(n):
    num = float(input(f"{i+1}번째 숫자 입력: "))
    numbers.append(num)

total = sum(numbers)
average = total / n

print(f"합계: {total}")
print(f"평균: {average}")
