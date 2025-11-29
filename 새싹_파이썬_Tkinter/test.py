from tkinter import *

# 창 열기
win = Tk()

win.title("test")
win.geometry("500x400")
#가로 x 세로 + x + y 

#Label 텍스트나 이미지 등을 화면에 출력하는법

l1 = Label(win, bg="green", fg="white",font=("Helvetica",16))
#딕셔너리로 텍스트를 넣을수있다
l1["text"] = "TkinterTest"
l1.pack(side=RIGHT,fill="x")
#글자 배경색,글꼴,글자색
l2 = Label(win,text="Python",bg="black", font=("Helvetica",16),fg="blue")
l2.pack(side=LEFT,fill="x")

l3 = Label(win,text="Java",bg="yellow", font=("Helvetica",16),fg="red")
l3.pack(side="top",fill="y",expand=YES)

#버튼 클릭이벤트
def click():
    print("클릭클릭클릭")
    l2["text"] = "클릭누르면 바뀌나?"

def click2():
    print(f"입력한 값 : {entry.get()}")
    l2["text"] = entry.get()

def ckeckbuttn():
    print(var1.get(),var2.get())

def check():
    print(g1.get())

def select():
    index = lst.curselection()[0]
    print(index,lst.get(index))
def select2():
    for index in lst2.curselection():
        print(index,lst2.get(index))


var1 = IntVar()
var2 = IntVar()
g1 = StringVar()

btn = Button(win, text="click",bg="red",fg="white",command=click,font=("Helvetica",14))
btn.pack(side="bottom")

entry = Entry(win,show="$")
entry.pack(fill=X)

btn2 = Button(win, text="click",bg="blue",fg="white",command=click2,font=("Helvetica",14))
btn2.pack(side="bottom")

chk1 = Checkbutton(win, text="영화",bg="blue",fg="white",variable=var1,command=ckeckbuttn,font=("Helvetica",14))
chk1.select()
chk1.pack(side="top")

chk2 = Checkbutton(win, text="음악",bg="blue",fg="white",variable=var2,command=ckeckbuttn,font=("Helvetica",14))
chk2.deselect()
chk2.pack(side="bottom")

rd1 = Radiobutton(win,text="라디오 버튼 만들기1",variable=g1,value='M',command=check)
rd1.pack(side=LEFT)
rd2 = Radiobutton(win,text="라이도 버튼 만들기2",variable=g1,value='F',command=check)
rd2.pack(side=LEFT)
rd1.select()

#한개만 출력 

lst = Listbox(win, height=3,selectmode=SINGLE)
lst.insert(0,"red")
lst.insert(1,"green")
lst.insert(2,"blue")
lst.pack(side="left")

listbutton = Button(win, text= "click" , command=select2)
listbutton.pack()


# 여러개 출력가능

lst2 = Listbox(win,height=3,selectmode="multiple")
color = ["red","green","blue"]
for item in color:
    lst2.insert(END,item)
lst2.pack()


btn3 = Button(win,text ="여러개 클릭가능", command=select2)
btn3.pack()


win.mainloop()
