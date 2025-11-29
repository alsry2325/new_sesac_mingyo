#Tuple
#값을 바꿀수 없으며, 여러개의 값을 담을수 있는 순서가 있는 자료형 소괄호와 콤마(,)를 활용


#투플 사용법
# 실직적으로 투플을 구분하는 것은 콤마
# 때에 따라 가독성을 위해 소괄호 사용
#투플은 추가/삭제/변경은 안됨 오로지  값을 읽어오는 자료형
tuple_zero = ()
tuple_one = (1,)
tuple_multiple = (1,2,3,4,5)  


tp = (4,2,1,5,6)
tp2 = (100,200)
print(tp[1])
print(tp[2:4]) 
print(6 in tp)
print(len(tp))
print(tp + tp2)
print(tp2 * 3)

#tp.append(6)

#unpacking(언패킹)
#묶여있는 자료형의 원소들을 분해해서 각 변수들에 나누는것

#반대는 여러 값을 묶어서 하나의 변수에 저장
10
# 활용1 스왑: 두변수의 값을 서로 바꾸는 것
a,b,c,d = (1,2,3,4)
print(a)
print(b)
print(c)
print(d)

#활용2 input().split() 함수  : 여러값을 한줄에 받을수있다  

#result = input().split()
#print(result)
'''
a,b,c,d,e = input().split()
print(a,b,c,d,e)
print(type(a))
print(type(b))
print(type(c))

'''
#str.split(c) → List 
#문자열c를 기준으로  분할하여  각분할된 값을 리스트로 묶어서 반환한다 하지만 타입은 문자열타입이다

'''
a,b,c = int(input().split())
print(type(a))
print(type(b))
print(type(c))
'''
# 그래서 정수형으로 바꿔주려면 map()함수를 활용해야한다
#변환함수를 적용하여 변환된 새로운 리스트를 반환

# a,b,c = list(map(int,input().split())) 
a,b,c = map(int,input().split())
print(type(a))
print(type(b))
print(type(c))

#딕셔너리 Dictionary = 사전
# 단어와 뜻이 하나의 쌍을 이루는 자료형 key와 value 값을 가진다

#예시 
#중괄호와 콜론(:)을 사용하여 표현한다 key와 value가 한쌍이며 ,콤마로 구분함

person = {
    "name":"정민교",
    "age":29,
    "address":"서울시 용산구",
    "phone_number":"01012345678",
    "email":"test@test.com"
    }

#값 꺼내오기
# print(person["name"])
print(person[name])
print(person["age"])
print(person["address"])
print(person["phone_number"])
print(person["email"])

#값 추가하기
#딕셔너리[새로운key] = value 
#예시
person["gender"] = "M"
print(person)


#값 수정하기
#딕셔너리[기존key] = value
#예시
person["address"] = "주소 바꿈"
print(person)

#값 삭제하기
#del 딕셔너리[key]
del person["email"]
print(person)


#key:value의 포함 여부
#in 연산자를 활용하여 해당 key 딕셔너리가 있는지 검사
#예시
 
if "age" in person:   #해당 키가 있으면 값을 12로 바꾼다
    person["age"] =12

print(person)

level = {'low':1,'medium':5}

#예제1
print(level["medium"])
#예제2
if "low" in level:
    print("low있음")

#예제3
level["high"] = 10
print(level) 

#예제4
del level["low"]
print(level)

#딕셔너리 활용
#중첩 딕셔너리
company = {
    "name": "abc",
    "departments": {
        "HR":{
            "manager" :"이여희",
            "employees" : ["홍길동","김민수"]
        },
        "IT":{
            "manager" :"이여희",
            "employees" : ["홍길동","김민수","정민교"]
        }
    }
}

print(company["departments"]["HR"]["manager"])
#IT 부서의 직원의 수를 가지고 오고싶다
print(len(company["departments"]["IT"]["employees"]))

#딕셔너리 메서드 활용

print(person.keys())  #모든 key들의 묶음을 반환함
print(person.values()) #모든 value들을 묶음을 반환함
print(person.items())  #key와 value 쌍으로 묶어서 반환함

#for문과 함꼐 활용

for x in person:            #키값만 출력
    print(x)

for x in person.values():  #value값만 출력
    print(x)


for x,y in person.items():
    print(x,y)


#딕셔너리 활용 dict()함수
#다중 원소로 구성된 리스트나 투플을 딕셔너리로 변환하는 함수

li = [("name","Gildong"),('year',1999)]

info2 = dict(li)
print(info2)

#zip()함수
#동일한 인덱스끼리 하나의 투플로 묶는 역할
#[('name', 'john'), ('age', 30), ('year', 1996)]
title = ["name","age","year"]
value = ["john",30,1996]

print(list(zip(title,value)))

#zip()함수와 dict()함수 활용
#{'name': 'John', 'age': 30, 'year': 1996}
title = ['name','age','year']
value = ['John',30,1996]

print(dict(zip(title,value)))

names = []
scores = []

for i in range(5):
    n = input("이름을 입력하세요")
    c = int(input("점수를 입력허세요."))
    names.append(n)
    scores.append(c)

result = dict(zip(names,scores))
print(result)