#함수란?
#입력에 따라 어떤 알을 수행하고 결과를 돌려주는것
# 기능을 수행하는 명령들의 모임


#함수의 종류

#내장함수
#기본적으로 탑재가 되어있는 함수
#집계함수들 종류 예

#max()
#최대값을 구하는 함수
print(max(10,20,13,554,212))
  
#min()
# 치솟값을 구하는 함수
print(min((1,2,3,4,5,6)))
#sum()
#시퀀스 자료의 합을 구해주는 함수



#사용자 정의 함수
#사용자가 직접 정의하는 함수
#1. def키워드를 사용하여 함수를 정의하겠다 선언
#2.매개변수(parameter): 함수 호출시 함수의 외부에서 내부로 값을 전달 할때 받는 함수 t의 자리가 매개변수
# 인자(argument): 함수 호출시 함수에 전달하는 값 
#3. 함수의 명령 정의
#4.return으로 함수를 반환
#5. 함수를 호출
def hello(t):
    a = 10 * t + 2
    return a

print(hello(2))

def plus(num3,num4):
    return num3 + num4
def sub(num1,num2):
    return num1 - num2
def mul(num1,num2):
    return num1 * num2
def div(num1,num2):
    return num1 / num2

print(plus(1,2))
print(sub(1,2))
print(mul(1,2))
print(div(1,2))


def test(num1,num2):
    plus = num1 + num2
    sub =  num1 - num2
    mul =  num1 * num2
    div =  num1 / num2
    
    return plus,sub,mul,div

print(test(3,5))


#반환 여부에 따라 차이가 발생한다
#리턴값이 없으면 none호출
def formula(a,b):
    c= (a**2)+(b**2)
    print("return :",c)
formula(2,4)
result = formula(2,4)

print(formula(2,4)) 

#함수 매개변수확장 : 디폴트 매개변수 매칭되는  인자없이 호출하면 디폴트값을 출력
def def_para(country = 'korea'):
    print("I am from",country)

def_para("India")
def_para("Brazil")
def_para()


#가변 매개변수 : 여러개의 겂을 한번에 받을수 있는 매개변수 전달받은 매개변수는 투플형태로 받은 값들을 묶는다
kor,eng,mat,sci = 98,77,85,12

def max_score(*args):
    print(args) #(98, 77, 85, 12)
    return max(args)

print(max_score(kor,eng,mat,sci))
print(sum((kor,eng,mat,sci)))

#매개변수 이름으로 인자 전달
def sum_score(kor,eng,math):
    return kor+eng+math

# print(sum_score(kor=70,eng=80,math=60))
print(sum_score(70,80,90))        


#함수의 unpaking/packing 활용법
#인자로 리스트 또는 투플에 * 붙여주면 언패킹 수행
def unpaking(a,b,c):
    print(a)
    print(b)
    print(c)

mylist = [10,20,30]

unpaking(*mylist)

#그리고 반환 부분에 여러개의 값을 반환 할수있다(투플 형태로 반환)
def three(x):
    return x, x**2,x**3

a,b,c = three(3) 
print(a,b,c) #언패킹
print(three(3))


#전역변수와 지역변수
#전역변수 : 전역적인 범위애서 사용가능한 변수
#지역변수 : 특정구문(for문이나 함수)안에서 정의한 함수 정의한 범위내에서만 쓸수있다


#전역변수
#gobal Namespace 영역에 {이름: 객체} 딕셔너리 형태로 저장되어있다
global_var ="Hi"
def my_func(a):
    print(a)
    print(global_var)

my_func(global_var)

print(global_var)


 