# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:36:02 2020
@author: MU-PING
"""
import os
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from math import floor, ceil
from collections import defaultdict

class Data():
    def __init__(self, filename):
        self.filename = filename
        self.color_dict={0:'blue',1:'red',2:'orange',3:'green'}
        self.read_data()
        self.draw()
        
    def read_data(self):
        self.data = []
        self.label=1                            #標籤種類
        self.label_checker = defaultdict(int)   #標籤映射
        
        with open(os.getcwd()+'/DataSet/'+ self.filename +".txt", 'r', encoding='UTF-8') as file:
            for data in file.readlines():
                each_data = data.replace("\n", "").split(" ")
                each_data = [float(i) for i in each_data]
                each_data[-1] = int(each_data[-1])
                if(self.label_checker[each_data[-1]]==0): #代表出現新標籤
                    self.label_checker[each_data[-1]] = self.label
                    self.label += 1
                each_data[-1] = self.label_checker[each_data[-1]]-1    
                self.data.append(each_data)
                
        self.dimension = len(self.data[0])-1        
        self.data = np.array(self.data)
        
        #讀完資料設置製圖的最大x y z 軸
        self.data_transpose = self.data.transpose() #用於取x y (z) 軸最大值
        self.max_x = ceil(max(self.data_transpose[0])+0.5)
        self.min_x = floor(min(self.data_transpose[0])-0.5)
        self.max_y = ceil(max(self.data_transpose[1])+0.5)
        self.min_y = floor(min(self.data_transpose[1])-0.5)
        
    def draw(self):
        plt.figure(num='fig1')
        plt.clf() #清除圖片 參考：https://reurl.cc/9XnRrj

        if(self.dimension==2):
            plt.xlim(self.min_x, self.max_x)
            plt.ylim(self.min_y, self.max_y)
            for i in self.data:
                plt.plot(i[0], i[1], 'o', ms = 4 , color = self.color_dict[int(i[-1]%self.label)], alpha=.6) #畫圖 ms：折點大小
            plt.draw()  
            
        elif(self.dimension==3):
            self.max_z = ceil(max(self.data_transpose[2]))
            self.min_z = floor(min(self.data_transpose[2]))
            ax = Axes3D(fig1) #3D圖
            ax.set_xlim(self.min_x, self.max_x)
            ax.set_ylim(self.min_y, self.max_y)
            ax.set_zlim(self.min_z, self.max_z)
            for i in self.data:  #ax.plot版本不一樣語法不一樣
                ax.plot([i[0]], [i[1]], [i[2]], 'o', ms=4 , color = self.color_dict[int(i[-1]%self.label)], alpha=.6)
            plt.draw()    
    
class MLP():
    def __init__():
        pass
        
def create_data(event):
    data = Data(event.widget.get())


window = tk.Tk()
window.geometry("1200x720")
window.resizable(False, False)
window.title("感知機訓練器")

file=["2Ccircle1", "2Circle1", "2Circle2", "2CloseS", "2CloseS2", 
      "2CloseS3", "2cring", "2CS", "2Hcircle1", "2ring", 
      "4satellite-6", "5CloseS1", "8OX", "C3D", "C10D",
      "IRIS", "Number", "perceptron1", "perceptron2", "perceptron3",\
      "perceptron4", "wine", "xor"]
    
# 左半邊Panel
l_panel = tk.Frame(window)
l_panel.grid(row=0, column=0, sticky=tk.NW)

#設定框_訓練資料集
panel1 = tk.Frame(l_panel)
panel1.grid(row=0, sticky=tk.NW, padx=8, pady=3)
tk.Label(panel1, font=("微軟正黑體", 10, "bold"), text="選擇訓練資料集").grid(row=0, sticky=tk.W, pady=5)
data_combobox = ttk.Combobox(panel1, value=file, state="readonly") #readonly為只可讀狀態
data_combobox.grid(row=1, sticky=tk.W, padx=12)
data_combobox.bind("<<ComboboxSelected>>", create_data)

#設定框_學習率
panel2 = tk.Frame(l_panel)
panel2.grid(row=1, sticky=tk.NW, padx=8, pady=3)
lr = tk.StringVar()#學習率
lr.set("0.1")
tk.Label(panel2, font=("微軟正黑體", 10, "bold"), text="學習率").grid(row=0, sticky=tk.W, pady=5)
tk.Entry(panel2, width=10, textvariable=lr).grid(row=1, padx=12)

#設定框_收斂條件
panel3 = tk.Frame(l_panel)
panel3.grid(row=2, sticky=tk.NW, padx=8, pady=5)
convergence = tk.IntVar() #判斷收斂條件
epoch_accuracy = tk.StringVar() #訓練次數、正確率
epoch_accuracy.set("2")
convergence_condition_text = tk.Label(panel3, font=("微軟正黑體", 10, "bold"), text="收斂條件").grid(row=0, sticky=tk.W)
tk.Radiobutton(panel3, font=("微軟正黑體", 10, "bold"), text="Epoch", variable=convergence, value=0).grid(row=1, column=0, sticky=tk.W)
tk.Radiobutton(panel3, font=("微軟正黑體", 10, "bold"), text="Accuracy", variable=convergence, value=1).grid(row=1, column=1, sticky=tk.W)
tk.Entry(panel3, width=10, textvariable=epoch_accuracy).grid(row=2, padx=12)


#設定框_設定神經元
panel4 = tk.Frame(l_panel)
panel4.grid(row=3, sticky=tk.NW, padx=8, pady=5)
hidden_layer=tk.IntVar()
hidden_layer.set(1)
hidden_layer_num=tk.IntVar()
hidden_layer_num.set(10)
tk.Label(panel4, font=("微軟正黑體", 10, "bold"), text="隱藏層層數：").grid(row=0, sticky=tk.W)
tk.Label(panel4, font=("微軟正黑體", 10, "bold"), text="隱藏層神經元：").grid(row=2, sticky=tk.W)
hidden_layer_input=tk.Entry(panel4, width=10, textvariable=hidden_layer)
hidden_layer_input.grid(row=1, sticky=tk.W, padx=12, pady=5)
hidden_layer_num_input=tk.Entry(panel4, width=10, textvariable=hidden_layer_num)
hidden_layer_num_input.grid(row=3, sticky=tk.W, padx=12, pady=5)

#設定框_開始訓練按鈕
panel5 = tk.Frame(l_panel)
panel5.grid(row=4, sticky=tk.NW, padx=8, pady=5)
btn_reset = tk.Button(panel5, text='重新設定')
btn_reset.grid(row=4, column=0, sticky=tk.E)
btn_reset.configure()
btn = tk.Button(panel5, text='開始訓練')
btn.grid(row=4, column=1, sticky=tk.E, padx=30)

#設定框_資料圖 num用於管理不同figure
image1 = tk.Frame(l_panel)
image1.grid(row=5, sticky=tk.NW, padx=8, pady=5)
tk.Label(image1, font=("微軟正黑體", 11, "bold"), text="資料分布圖").grid(row=0, columnspan=6, pady=2)
btn1 = tk.Button(image1, text='全部資料(1)')
btn1.grid(row=1, column=0, columnspan=2, pady=4)
btn2 = tk.Button(image1, text='訓練資料(3/4)')
btn2.grid(row=1, column=2, columnspan=2, pady=4)
btn3 = tk.Button(image1, text='測試資料(1/4)')
btn3.grid(row=1, column=4, columnspan=2, pady=4)
fig1 = plt.figure(num='fig1', figsize=(4,4))
canvas1 = FigureCanvasTkAgg(fig1, image1)  # A tk.DrawingArea.
canvas1.get_tk_widget().grid(row=2, column=0,columnspan=6, padx=5)

# 左半邊Panel
r_panel = tk.Frame(window)
r_panel.grid(row=0, column=1, sticky=tk.NW)

#設定框_數據表格
table = tk.Frame(r_panel)
table.grid(row=0, column=0, columnspan=2, sticky=tk.N)
tk.Label(table, font=("微軟正黑體", 12, "bold"), text="資料表格").grid(row=0, column=0, columnspan=2, pady=4) 
tree=ttk.Treeview(table, column=("Input","train_or_test","Output"),show="headings", height=14)
tree.column("Input", width=450, minwidth=450, stretch=False)
tree.column("train_or_test", width=200, minwidth=200, stretch=False)
tree.column("Output", width=200, minwidth=200, stretch=False)
tree.heading("Input",text="資料",anchor=tk.CENTER)
tree.heading("train_or_test",text="訓練/測試",anchor=tk.CENTER)
tree.heading("Output", text="輸出",anchor=tk.CENTER)
tree.grid(row=1, column=0)
ybar = ttk.Scrollbar(table, orient ="vertical", command = tree.yview) 
ybar.grid(row=1, column=1, sticky=tk.S + tk.W + tk.E + tk.N) 
xbar = ttk.Scrollbar(table, orient ="horizontal", command = tree.xview) 
xbar.grid(row=2, column=0, sticky=tk.S +tk. W + tk.E + tk.N) 
tree.configure(yscrollcommand = ybar.set)
tree.configure(xscrollcommand = xbar.set)

#設定框_Loss圖
image2 = tk.Frame(r_panel)
image2.grid(row=1, column=0 ,sticky=tk.N, pady=20)
fig2 = plt.figure(num='fig2', figsize=(6,4.5))
plt.title("Loss( RMSE )")
canvas2 = FigureCanvasTkAgg(fig2, image2)  # A tk.DrawingArea.
canvas2.get_tk_widget().grid(padx=5)

#設定框_Accuracy圖
image3 = tk.Frame(r_panel)
image3.grid(row=1, column=1 ,sticky=tk.N, pady=20)
fig3 = plt.figure(num='fig3', figsize=(6,4.5))
plt.title("Accuracy")
canvas3 = FigureCanvasTkAgg(fig3, image3)  # A tk.DrawingArea.
canvas3.get_tk_widget().grid(padx=5)

window.mainloop()