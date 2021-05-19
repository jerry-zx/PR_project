from graphics import *
import numpy as np
from main import predict_observe
import matplotlib.pyplot as plt

data_set = []
data_label = []

print("input every instance in one line, end with the corresponding label")
k = 0
while True:
    line = input("instance %s:" % str(k))
    k += 1
    line = line.split(' ')
    if len(line) < 3:
        break
    else:
        for i in range(len(line)):
            line[i] = float(line[i])
        data_set.append(line[:2])
        data_label.append(int(line[-1]))


data = np.array(data_set)
label = np.array(data_label)

num_data = len(data)
num_label = len(set(label))
ones = np.ones((num_data, 1), dtype=np.float)
data = np.concatenate((data, ones), axis=1)  # 增广化  (num_data, dim+1)

# initialize the params
weight = np.zeros_like(data)  # (num_data, dim+1)
k = 0  # number of iterations
n_c = 0  # number of continuous instances that are classified correctly
alpha = 1  # length of a step

# train
while n_c < num_data:
    cur_x = data[k % num_data]  # current instance  (1, dim+1)
    g = np.dot(cur_x, weight.T)  # (1, num_data)
    for i in range(num_data):
        if i == k % num_data:
            if g[i] <= 0:  # if classified correctly
                weight[i, :] = weight[i, :] + alpha * cur_x
                n_c = 0
            else:
                n_c += 1
        else:
            if g[i] >= 0:
                weight[i, :] = weight[i, :] - alpha * cur_x
                n_c = 0
            else:
                n_c += 1
    k += 1

print(weight)  # Notice: It is NOT the decision boundaries!

width = 2 * max(max(np.abs(data_set[:, 0])), max(np.abs(data_set[:, 0])))
height = 2 * max(max(np.abs(data_set[:, 1])), max(np.abs(data_set[:, 1])))
win = GraphWin(200, 200)
win.setCoords(-width / 2, -height / 2, width / 2, height / 2)
"""
w_1 = 2
w_2 = 3
b = 0  # 保证w_1*x_1+w_2*x_2+b>0是positive
x_1 = (-w_2 * height / 2 - b) / w_1
x_2 = (w_2 * height / 2 - b) / w_1

polyon_1 = Polygon(Point(x_1, height / 2), Point(x_2, -height / 2), Point(width / 2, -height / 2),
                   Point(width / 2, height / 2))
polyon_2 = Polygon(Point(x_1, height / 2), Point(x_2, -height / 2), Point(-width / 2, -height / 2),
                   Point(-width / 2, height / 2))
polyon_1.setFill("yellow")
polyon_2.setFill("green")
polyon_1.draw(win)
polyon_2.draw(win)

predict_observe(w_1, w_2, b, width, 2, win)
"""
win.getMouse()