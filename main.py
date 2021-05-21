from tkinter import * 
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

def train(train_data, label_num, epoch, alpha=1):
    feat_dim = train_data.shape[1]
    weights = np.ones((label_num, label_num-1, feat_dim))
    for i in range(label_num-1):
        for j in np.arange(i+1,label_num):
            data_i = train_data[np.where(train_data[:,2]==i)[0]]
            data_j = train_data[np.where(train_data[:,2]==j)[0]]
            data = np.append(data_i, data_j,axis=0)
            data, label=np.split(data,[-1],axis=1)
            num_data = len(data)
            ones = np.ones((num_data, 1), dtype=np.float)
            data = np.concatenate((data, ones), axis=1)  # 增广化  (num_data, dim+1)
            w = np.zeros(data.shape[1])  
            k = 0  # 迭代步数
            n_c = 0  # number of continuous instances that are classified correctly
            while n_c < num_data and k < epoch:
                cur_x = data[k % num_data]
                g = np.dot(cur_x, w)
                if label[k % num_data][0] == i and g <=0:
                    w = w + alpha * cur_x
                    n_c = 0
                elif label[k % num_data][0] ==j and g >=0:
                    w = w - alpha * cur_x
                    n_c = 0
                else:
                    n_c += 1
                k += 1
            weights[i][j-1] = w
            weights[j][i] = -w
    return weights

def test(test_data, weights):
    num_data = len(test_data)
    data, label = np.split(test_data,[-1],axis=1)
    ones = np.ones((num_data, 1), dtype=np.float)
    data = np.concatenate((data, ones), axis=1)  # 增广化  (num_data, dim+1)
    predict = np.dot(weights,data.transpose(1,0)).transpose(2,0,1)
    
    correct = 0
    for i in range(num_data):
        predict_i = (predict[i] > 0).all(axis=1)
        predict_label = np.where(predict_i == True)[0]
        if len(predict_label) == 0:
            print("样本位于不确定区域")
        else:
            print("预测类别为%d" % predict_label[0])
        try:
            if predict_label[0] == label[i]:
                correct += 1
        except:  # 不确定区域的样本
            continue
    return correct/num_data

def visualize(width, height, weights):
    coordinates = np.array(list(product(np.arange(-width/2, width/2,width/100), np.arange(-height/2, height/2,height/100))))
    num_data = len(coordinates)
    ones = np.ones((num_data, 1), dtype=np.float)
    data = np.concatenate((coordinates, ones), axis=1)  # 增广化  (num_data, dim+1)
    predict = np.dot(weights,data.transpose(1,0)).transpose(2,0,1)
    label = -1 * np.ones(shape=num_data)
    has_label = np.where((predict > 0).all(axis=2)==True)
    label[has_label[0]] = has_label[1]
    
    for i in np.arange(-1, weights.shape[0]):
        x = np.where(label == i)[0]
        if i == -1:
            plt.scatter(coordinates[x,0], coordinates[x,1], label = '不确定区域', marker='s')
        else:
            plt.scatter(coordinates[x,0], coordinates[x,1], label = '类别'+str(i), marker='s')
    plt.legend()
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("分类结果可视化")
    plt.show()

def main():
    global train_data, test_data
    train_data, test_data = [], []
    
    root = Tk()
    root.title("多类别二维分类器")
    # 设置标签
    Label(root, text='类别个数：').grid(row=0, column=0)
    Label(root, text='训练最大轮数：').grid(row=1, column=0)
    Label(root, text="输入x1").grid(row=0,column=2)
    Label(root, text="输入x2").grid(row=0,column=3)
    Label(root, text="输入label").grid(row=0,column=4)
    # 输入框
    e1 = Entry(root)
    e2 = Entry(root)
    e3 = Entry(root)
    e4 = Entry(root)
    e5 = Entry(root)
    # 位置
    e1.grid(row=0, column=1, padx=10, pady=5)
    e2.grid(row=1, column=1, padx=10, pady=5)
    e3.grid(row=1, column=2, padx=10, pady=5)
    e4.grid(row=1, column=3, padx=10, pady=5)
    e5.grid(row=1, column=4, padx=10, pady=5)

    def continue_train():    
        n1 = e3.get()  
        n2 = e4.get()  
        n3 = e5.get()
        train_data.append([float(n1),float(n2),int(n3)])
        e3.delete(0, 'end') 
        e4.delete(0, 'end')
        e5.delete(0, 'end')

    def inputs_train():
        Button(root, text='继续输入', width=10, command=continue_train).grid(row=4, column=4, sticky=E, padx=10, pady=5)
        Button(root, text='完成输入', width=10, command=continue_train).grid(row=5, column=4, sticky=E, padx=10, pady=5)
    
    def continue_test():    
        n1 = e3.get()  
        n2 = e4.get()
        n3 = e5.get()
        test_data.append([float(n1),float(n2),int(n3)])
        e3.delete(0, 'end')    
        e4.delete(0, 'end')
        e5.delete(0, 'end')

    def inputs_test():
        Button(root, text='继续输入', width=10, command=continue_test).grid(row=4, column=4, sticky=E, padx=10, pady=5)
        Button(root, text='完成输入', width=10, command=continue_test).grid(row=5, column=4, sticky=E, padx=10, pady=5)

    Button(root, text='输入训练样本', width=10, command=inputs_train).grid(row=2, column=1, sticky=E, padx=10, pady=5)
    Button(root, text='输入测试样本', width=10, command=inputs_test).grid(row=3, column=1, sticky=E, padx=10, pady=5)
    Button(root, text='训练并测试', width=10, command=root.quit).grid(row=4, column=1, sticky=E, padx=10, pady=5)
    mainloop()

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    label_num =int(e1.get())
    epoch =int(e2.get())
    
    weights = train(train_data, label_num, epoch, alpha=1)
    
    acc = test(test_data, weights)
    print("分类器ACC指标:%f" % acc)
    
    width = 2*max(max(np.abs(train_data[:,0])),max(np.abs(test_data[:,0])))
    height = 2*max(max(np.abs(train_data[:,1])),max(np.abs(test_data[:,1])))
    
    visualize(width, height, weights)

if __name__ == '__main__':
    main()
