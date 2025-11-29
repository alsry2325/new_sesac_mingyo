from tkinter import *
from PIL import Image


def choice():
    index =  lists.curselection()[0]
    if index  == 0:
        lbl["image"] = photo1
    elif index == 1:
        lbl["image1"] = photo2
    elif index == 2:
        lbl["image2"] = photo3
# 창 열기
win = Tk()

win.title("test")
win.geometry("300x400")
#가로 x 세로 + x + y 

animals = ["cat,dog,bear"]

lists = Listbox(win,height=3,selectmode="single")
for item in animals:
    lists.insert(END,item)

lists.pack(fill="both")

btn = Button(win,text ="click", command=choice)
btn.pack(fill="both")

photo1 = Image.open("새싹_파이썬_Tkinter/t1.webp")
photo2 = Image.open("새싹_파이썬_Tkinter/t2.avif")
photo3 = Image.open("새싹_파이썬_Tkinter/t3.jpg")


lbl = Label(win,text="My animal")
lbl.pack(fill="both",expand=YES)
win.mainloop()
