#모듈이란?
#여러 함수, 변수, 클래스등을 하나의 파일로 묶은것
#모듈의 종류
#내장모듈, 사용자 정의모듈 
#내장: 기본적으로 제공하는 모듈
#사용자 정의 모듈: 사용자가 직접 만든.py파일

#모듈 불러오는법 import키워드를 사용해서 불러온다

#사용법
#모듈이름.함수/변수/클래스 형태로 사용

#내장모듈 종류

#수학/통계 math,statistics,random
#날짜/시간 datetime,time
#파일/시스템 os,sys,pathlib
#데이터 처리 관련 json,csv,re
#네트워크 관련 socket,http,urllib

#패키지
#여러개의 모듈을 하나의 디렉토리로 묶어 관리하는 단위(모듈들의 모음)
#사용법
#폴더이름.모듈이름 형태로 임포트 수행

#라이브러리
#기능과 목적  특정 분야 전체를 지원하는 도구


import math
import random
import datetime
import cal
from math import factorial

print(math.pi)  #원주율
print(math.e)   #자연상수
print(math.sqrt(16))   #제곱근
print(math.pow(2,4))    #거듭제곱
print(math.ceil(3.5))  #올리
print(math.floor(3.9)) # 내람


print(random.random())  #0~1 사이 난수
print(random.randint(1,10)) # 1~10 사이 정수 난수
print(random.choice(['가위','바위','보']))
print(random.sample(range(1,46),6))   #n개의 샘플 뽑기


now = datetime.datetime.now()
print("현재 시간",now)
today = datetime.date.today()
print("오늘 날짜",today)
future = today + datetime.timedelta(days=8,hours=2) #시간 차(간격) 계산
print("일주일 뒤",future)


print(cal.plus(1,2))

cal.Person("민교",15).eat("김밥")

#from improt 방식 메소드만 호출하면 된다
print(factorial(5))

#파일
#바이트의 묶음 저장된 의미가 있는 데이터 한 덩어리

#파일열기
# f = open(파일 경로 문자열)  관련된 작업이 끝나고 나면 close() 통해 파일을 닫아줘야함

#파일 열기 옵션
'''
f= open(파일 경로 문자열,열기 옵션)

r = 읽기전용
w = 쓰기 전용(파일이 존재하면 덮어쓰기)
x = 쓰기 전용(파일이 존재하면 오류)
a = 쓰기전용(파일 존재하면 기존내용 뒤에 이어씀)
t = 텍스트 모드 (읽기 및 쓰기)
b = 바이너리 모드
'''

#파일 읽기모드

f= open('새싹_모듈과파일처리/text.txt',encoding ='utf-8')
# print(f.read(4))
# f.close()

# 모든 줄 가져오기
print(f.readline())
print(f.readlines())

#파일 쓰기 write()
# f= open('새싹_모듈과파일처리/text.txt',encoding ='utf-8')

# f= open('새싹_모듈과파일처리/text.txt',"w")

# f.write("test\n")
# f.write("test\n")
# f.write("test\n")


# f.write('''
#         sdsdsd
#     s

#         ''')
# f = open("새싹_모듈과파일처리/text.txt","w")
# f = (f.read())
# f.close()

#파일 현재 위치가져오기 
#tell()메서드

f = open("새싹_모듈과파일처리/text.txt",encoding='utf-8')

print(f.read(3))

f.close()


f = open("새싹_모듈과파일처리/text.txt",encoding='utf-8') 
print(f.tell())  #0 현재 위치를 반환하는 메서드
print(f.seek(3)) #3 현재 위치를 변경하는 메서드
print(f.tell()+3) #6
print(f.read(1))
f.close

#whth 구문
#자원 객체(파일 객체, db세션)에 대상으로 한정된 구역에서 사용하도록 하는구문 
# 열고 작업 제대 닫아주는 동작을 보장한다


with(
    open("새싹_모듈과파일처리/text.txt",encoding='utf-8') as f1,
    open("새싹_모듈과파일처리/text.txt",encoding='utf-8') as f2
 ):
      content1 = f1.read()
      content2 = f2.read()

print(content1)
print(content2)


#예외처리
#발생할수있는 옝외에 대해 오류및 종류하는 갓 이외의 동작을 정의하는것

try:  #예외가 발생할 가능성있는 코드
    1/0
except ZeroDivisionError as e:  #예외 상황이 발생했을때 처리할 동작
    print("0으로 나눌수 없습니다",e)  #as e 예외의 정보를 확인가능

finally : #예외처리가 없이 항상 실행되는 동작
     print("예외처리 끝")

#파일 예외처리
try:
     f =open("없는파일.txt","r")
except FileNotFoundError as e:
     print("파일을 찾을수 없습니다",e)

#정수 변환 예외처리
try:
     num = int("avc")
except ValueError as e:
     print("정수로 변할수없습니다",e)

#인덱스 예외처리
# try:
#      arr = [1,2,3]
#      print(arr[5])
# except IndexError as e:
#      print("인덱스가 범위를 벗어났습니다",e)

# while True:
#      try:
#           number = int(input("숫자를 입력해보세요"))
#           break
#      except ValueError:
#           print("에러 숫자를 입력하세요")
# print("number:", number) 

try:
     my_list=[1,2,3,4,5]
     a,b = map(int,input("두개의 숫자를 입력하세요,").split())
     print(my_list[b])
except ValueError:
     print("입력이 옳지 않습니다")
except IndexError:
     print("b가 인덱스 범위를 초과했습니다")
except Exception :
     print("알수없는에러가 발생했습니다")
else:
     print("a+b = ",a+b)
finally:
     print("프로그램 종료")

