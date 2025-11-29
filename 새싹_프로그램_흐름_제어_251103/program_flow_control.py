#list
#여러개의 값을 담을수있는 자료형 
#, 콘마로  구분한다

empty_list = []
num_list = [1,2,3,4,5]
mix_list = ["a",2,False,27.25] #서로 다른 자료형끼리 묶음가능

#시퀀스 자료형
#종류
'''
문자열(String)
리스트(list)
튜플(Tuple)
'''

#시퀀스 자료형에는 index 라는 것이 존재한다
#원소의 위치를 인덱스라고 부른다

num_list = [1,2,3,4,5] # 0,1,2,3,4

#원소에 접근하는 방법을 인덱싱 이라고 한다
#리스트[인덱스_번호] = 값
alpha = [1,2,23,13,45,2323]
print(alpha[2]) #23이 출력됨

#int값은 수정이 가능하다 

num_list = [2,2,3,4,5]
num_list[1] = num_list[1] ** 2 #자료형 연산도 가능 
num_list[4] = 23

print(num_list) #[2, 4, 3, 4, 23]

#하지만 String값은  Immutable타입이라 불가능하다

#슬라이싱 인덱스 범위에서 연속된 일부분을 가져오는 기법

#문자열/리스트[] = [시작인덱스:끝인덱스]

num_list = [1,2,3,4,5] 
print(num_list[1:3]) # 1부터 3미만의 값을 가져옴 [2, 3]

#음수의 인덱싱

a = "hollo world"
b = [4,5,1,2,3]

print(a[-1:]) #d 만 나옴
print(b[1:-4]) #빈값이 나오게 됨

#in 연산자
#해당 원소가 포함되어있는지 확인하는 함수
senteence = "Hollo world"
num_list = [1,2,3,4,5]
str_test = "Hollo "
num__test = [2,5]

print("E" in senteence) #False
print(str_test in senteence) #True
print(3 in num_list) #True
print(num__test in num_list) #False


#len() 함수
#자료형의 길이를 구하는 함수

num_list = [1,2,3,4,5,6]
world = "Python"

print(len(num_list))

#중첩 리스트

#리스트안에 리스트가 있는 구조 
#예시

nested =[[1,23,"sds",14],[1,2,3],[12,14,24,15,16]]

print(len(nested)) #3
print(len(nested[0])) #4 
nested_list = [
    ["a","b","c","d"],
    ["a","v"],
    ["c","s","x"],
    ["1","s","a","e","q"]
]

print(nested_list[1][1]) # v가 나옴 

#list메서드 종류

#list.append(x) : 리스트에 x를 가장 마지막에 위치에 추가가 된다 none
# none 결과값을 출력한다 의미없는 값

a = []
b = ["p","t","d"]

b.append(10)
b.append(13)
# 하나의 값만 추가가 가능하다
print(b)

b.append([1,2,56])

print(b)

#list.insert(i,x) : i번째 인덱스에 x를 삽입한다

a = [1,2,3,484,"12",[1,2,3,4]]
a[-1].insert(0,"배열 첫번째값 추가")
print(a)
a.insert(2,"test") #[1, 2, 'test', 3, 484, '12', ['배열 첫번째값 추가', 1, 2, 3, 4]]
print(a)

#list.remove(x) : 처음 검색된 원소x를 삭제한다

a.remove(3)  #3이라는 원소를 삭제한다
print(a) #[1, 2, 'test', 484, '12', ['배열 첫번째값 추가', 1, 2, 3, 4]]  

#리스트값에 없는 값을 선언하면 
#list.remove(x): x not in list 에러가 뜬다
#a.remove("ssssssc") 
#print(a)

#list.pop(i) : i번째  인덱스를 빼와서 반환한다 
#빼온값의 타입을 그대로 반환한다 오브젝트로 반환
a = [1,2,3,23,5,6]

print(a.pop(4))

#list.index(x)  : x에 해당하는 인덱스 검색 int값으로 반환

print(a.index(3)) #2가 나온다   

#list.count(x)  : 원소x의 개수를  int값으로 반환한다

a = ["a","b","a","A","C"] 

print(a.count("a"))  # a는 2개있어서 2개 반환
print(a.count("bdf")) # bdf값은 없어서 0를 반환한다
 

#list.sort() : 리스트 안의 겂들을 내부적으로 정렬하는 함수

a = [1,23,2,434,3,4,5]

a.sort() #오름차순으로 정렬이 됨
print(a)

a.sort(reverse=True) #내림차순 정렬하는 법
print(a)

#list.clear() : 모든원소들을 내부적으로 제거하는 기능

a = [5,4,23,23,231,2131,23,123,12,31,23,213,123,21,3,123,12,31,23,123,21,3,21,421,4235,2,5,342,52,45,324,643,6,436,43,6234,5,234,5,4325,233,4,214,24,23,4,234,236,43,6,2]
a.clear()

print(a)