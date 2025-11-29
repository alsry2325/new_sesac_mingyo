#객체지향 프로그래밍
#객체를 중심으로 프로그래밍하는 패러다임

#클래스
# 속성 ,동작을 갖고있는 자료형이다
#클래스는 설계도이고 이를 바탕으로 만들어진거를 객체라고한다  생성된 객체는 인스터스라고한다

#예시

#사람

#속성
#눈,손,발

#동작
#눈 깜빡이기,발 구부리기,손 접어보기

#클래스 정의 및 객체 생성하기
'''
1.클래스 이름은 PascalCase형식으로 짓는다
2. 생성자 정의하기
3. 메서드 정의하기
4. 객체 생성후, 객체.메서드()로 호출
인스턴스 변수: 클래스의 속성으로 정의된 변수
self키워드 : 객체 자기 자신을 의미하는 키워드  예를 들어 나는 이름은 민교이고 나이는 23살이다
'''
class Person:
    def __init__(self,name,age):  #생성자 
        self.name = name
        self.age = age
    
    def eat(self, food):  #메서드
        print(self.name,"가",food,"를(을) 먹는다")

p1 = Person("민교", 23) #생성자 호출

print(p1.name,p1.age)
p1.eat("김밥")


class Car:
    def __init__(self,name,speed):
        self.name = name
        self.speed = speed

    def get_name(self):
        return f"{self.name} s"
    
    def get_speed(self):
        return f"speed is {self.speed}"

c1 = Car("민교1",20)
c2 = Car("민교2",30)

print(c1.get_name(),c1.get_speed()) 
print(c2.get_name(),c2.get_speed())