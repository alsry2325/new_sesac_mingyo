# 학생 정보 리스트를 정의하세요.
# 함수들을 정의하세요.
students = []
def add_student(name, age, grade):
    # keys = {}
    # # students.append(name,age,grade)
    # # print(students)
    # keys["name"] = name
    # keys["age"] = age
    # keys["grade"] = grade
    # print(keys)
    # students.append(keys)
    # print(students)
    # print("추가 완료")
    students.append({'name' : name, 'age' : age, 'grade' : grade})
def get_all_students():
    print("전체 학생 목록")
    for i in students:
        print(i)
    
def find_student(name):

    # [{'name': '정민교1', 'age': 20, 'grade': 'a'}, {'name': '정민교2', 'age': 21, 'grade': 5}, {'name': '정민교3', 'age': 22, 'grade': 4}]
    for student in students:
        if  student["name"] == name:
            return student
    
def get_average_age():
    total = 0 
    for i in students:
        total += i["age"]

    print(total/len(students))
            
