#if문
# 만약에 특정 조건이 참이면 실행한다

'''
if 조건 : 
    <수행할 명령1>
    <수행할 명령2>

'''
#1.조건의 결과가 true 가 되는 판별
#2.if문의 마무리를 의미하는: 작성
#3.들여쓰기 
#4.조건에 따른 분기처리를 진행함

var1 = 1

if var1 > 0:
    print(var1,"양수입니다")

radius = int(input("반지름을 입력해라 : "))

if radius > 0:
    print("넓이",(3.14*radius**2))
    print("둘레",(2*3.14*radius))

num = int(input("정수를 입력해주세요. :"))

if num % 2 == 0:
    print(num,"짝수입니다")


# 
#만약에 조건이 맞지 않을때는 else 구문을 실행

num = int(input("정수를 입력하세요 :"))

if num % 2 == 0 :
    print(num,"은 정수입니다")
else:
    print(num,"은 홀수입니다") 


#if elif else 문
#그이상의 여러개의  조건을 걸고 싶을 때는 쓰는 문법
'''
if 조건1:
    <조건 1 명령>
elif 조건2:
    <조건 2 명령>
else:
    <else 명령>
'''

grade = 'B'

if grade == 'A':
    print("A등급입니다")
elif grade == 'B':
    print("B등급입니다")
elif grade == 'C':
    print("C등급 입니다")
else :
    print("D등급 입니다")


#중첩 if문
#조건을 세부적으로 걸고싶으면
num1 = 1
num2 = -1

if num1 > 0 :
    if num2 > 0 :
        print('둘다 양수입니다')
    elif num2 < 0 :
        print("num1은 양수,num2는 음수입니다")
elif num1 < 0 :
    if num2 > 0 :
        print('num1은 음수,num2는 양수입니다')
    elif num2 < 0 :
        print("둘다 음수입니다")


#비교 연산자 체이닝
#1. num이 10 이상이면서 ,90이하
#if 10 <= num and num <= 90 :
#2.num은 10부터 90이하
#if 10 <= num <= 90 :

a = ["H","h","f","l","l","o"]

# a리스트에서 h가 포함되면 삭제를하고 그렇지않으면 프린트 출력
if "h" in a :
    a.remove("h")
else:
    print("h가 없습니다")

#word라는 변수값이 5이상이면 두번 출력 그렇지 않으면 한번만 실행
word = "Python"

if len(word) >= 5:
    print(word *2)
else:
    print(word)

#word라는 변수값이  5이상이면서 P가 포함되어있으면 출력 그렇지않으면 프린트 실행
word = input("단어를 입력하세요")

if len(word) >= 5 and 'P' in word:
    print(word)
else:
    print("조건이 맞지않습니다")

#스코어 점수가 70이상이면서 출석점수가 90이상이면 통과 만약에 스코어 점수가 70이상이면서 출석점수가 90미만이면 출석미달 그외는 미흡
score = int(input("시험 점수를 입력하세요"))
attendance = int(input("출석 점수를 입력하세요."))

if score >= 70 and attendance >= 90:
    print("통과")
elif score >= 70 and attendance <90:
    print("조건부 통과(출석 미달)")
else:
    print("미흡")