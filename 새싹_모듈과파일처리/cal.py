

def plus(a,b):
    c = a + b
    return c

var1 = 12323

class Person:
    def __init__(self,name,age):  #생성자
        self.name = name
        self.age = age
    
    def eat(self, food):  #메서드
        print(self.name,"가",food,"를(을) 먹는다")
