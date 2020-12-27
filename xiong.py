import math
import matplotlib.pyplot as plt
import itertools
import numpy as np
from numpy import random
from scipy.optimize import linear_sum_assignment
import collections
 
 
# 任务分配类
class TaskAssignment:
 
    # 类初始化，需要输入参数有任务矩阵以及分配方式，其中分配方式有两种，全排列方法all_permutation或匈牙利方法Hungary。
    def __init__(self, task_matrix, mode):
        self.task_matrix = task_matrix
        self.mode = mode
        if mode == 'all_permutation':
            self.min_cost, self.best_solution = self.all_permutation(task_matrix)
        if mode == 'Hungary':
            self.min_cost, self.best_solution = self.Hungary(task_matrix)
 
    # 全排列方法
    def all_permutation(self, task_matrix):
        number_of_choice = len(task_matrix)
        solutions = []
        values = []
        for each_solution in itertools.permutations(range(number_of_choice)):
            each_solution = list(each_solution)
            solution = []
            value = 0
            for i in range(len(task_matrix)):
                value += task_matrix[i][each_solution[i]]
                solution.append(task_matrix[i][each_solution[i]])
            values.append(value)
            solutions.append(solution)
        min_cost = np.min(values)
        best_solution = solutions[values.index(min_cost)]
        return min_cost, best_solution
 
    # 匈牙利方法
    def Hungary(self, task_matrix):
        b = task_matrix.copy()
        # 行和列减0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        for i in range(len(b)):
            row_min = np.min(b[i])
            for j in range(len(b[i])):
                b[i][j] -= row_min
        for i in range(len(b[0])):
            col_min = np.min(b[:, i])
            for j in range(len(b)):
                b[j][i] -= col_min
        line_count = 0
        # 线数目小于矩阵长度时，进行循环
        while (line_count < len(b)):
            line_count = 0
            row_zero_count = []
            col_zero_count = []
            for i in range(len(b)):
                row_zero_count.append(np.sum(b[i] == 0))
            for i in range(len(b[0])):
                col_zero_count.append((np.sum(b[:, i] == 0)))
            # 划线的顺序（分行或列）
            line_order = []
            row_or_col = []
            for i in range(len(b[0]), 0, -1):
                while (i in row_zero_count):
                    line_order.append(row_zero_count.index(i))
                    row_or_col.append(0)
                    row_zero_count[row_zero_count.index(i)] = 0
                while (i in col_zero_count):
                    line_order.append(col_zero_count.index(i))
                    row_or_col.append(1)
                    col_zero_count[col_zero_count.index(i)] = 0
            # 画线覆盖0，并得到行减最小值，列加最小值后的矩阵
            delete_count_of_row = []
            delete_count_of_rol = []
            row_and_col = [i for i in range(len(b))]
            for i in range(len(line_order)):
                if row_or_col[i] == 0:
                    delete_count_of_row.append(line_order[i])
                else:
                    delete_count_of_rol.append(line_order[i])
                c = np.delete(b, delete_count_of_row, axis=0)
                c = np.delete(c, delete_count_of_rol, axis=1)
                line_count = len(delete_count_of_row) + len(delete_count_of_rol)
                # 线数目等于矩阵长度时，跳出
                if line_count == len(b):
                    break
                # 判断是否画线覆盖所有0，若覆盖，进行加减操作
                if 0 not in c:
                    row_sub = list(set(row_and_col) - set(delete_count_of_row))
                    min_value = np.min(c)
                    for i in row_sub:
                        b[i] = b[i] - min_value
                    for i in delete_count_of_rol:
                        b[:, i] = b[:, i] + min_value
                    break
        row_ind, col_ind = linear_sum_assignment(b)
        min_cost = task_matrix[row_ind, col_ind].sum()
        best_solution = list(task_matrix[row_ind, col_ind])
        return min_cost, best_solution

def loadData(filename):
    with open(filename,'r') as f:
        dicts = eval(f.read())
        #for key, value in dicts.items():
        #    print(key,value)
        sorted_dict = sorted(dicts.items(), key=None ,reverse=False)
        #print(sorted_dict)
        return sorted_dict


def loss_create_cost_matrix (datasheet_video,datasheet_audio):
    a = datasheet_audio.copy()
    v = datasheet_video.copy()

    M = int(len(a)/4)
    N = int(len(v)/4)
    
    #print(M,N)
    L = np.zeros((M,N))
    #print(L)
    for i in range(M):
        for j in range(N):
            # print(a[2+4*i])
            # print(v[2+4*i])
            if a[2+4*i] != v[2+4*j]:
                L[i][j] += 150
            if a[3+4*i] != v[3+4*j]:
                L[i][j] += 90
            L[i][j] += 200 * (1 - np.cos(a[1+4*i]-v[1+4*j]))**2
    
    return L
    

def create_matrix():
    # 首先读取两个大矩阵，其目的是为了构造audio-video cost matrix
    v_a = loadData('./CollisionRecognition/val_vedio_angle.txt')
    v_c = loadData('./CollisionRecognition/val_vedio_category.txt')
    v_i = loadData('./CollisionRecognition/val_vedio_ismove.txt')
    #print(len(v_a))

    #print(v_a[0][0])
    audio_a = loadData('./CollisionRecognition/val_audio_angle.txt')
    audio_c = loadData('./CollisionRecognition/val_audio_category.txt')
    audio_i = loadData('./CollisionRecognition/val_audio_ismove.txt') 
      
    # 数据录入后开始拓展
    v_combine = []
    audio_combine = []
    for i in range(len(v_a)):
        v_combine.append(v_a[i][0])
        v_combine.append(v_a[i][1])
        v_combine.append(v_c[i][1])
        v_combine.append(v_i[i][1])
    #print(v_combine)
    for j in range(len(audio_a)):
        audio_combine.append(audio_a[j][0])
        audio_combine.append(audio_a[j][1])
        audio_combine.append(audio_c[j][1])
        audio_combine.append(audio_i[j][1])
    return v_combine, audio_combine
  
b=create_matrix()
L=loss_create_cost_matrix(b[0], b[1])

# L=np.ceil(L)
'''
# 生成开销矩阵
rd = random.RandomState(1000)
task_matrix = rd.randint(0, 100, size=(6, 6))
'''
'''
# 用全排列方法实现任务分配
ass_by_per = TaskAssignment(task_matrix, 'all_permutation')
'''
# 用匈牙利方法实现任务分配
ass_by_Hun = TaskAssignment(L, 'Hungary')

#print('cost matrix = ', '\n', task_matrix)
'''
print('全排列方法任务分配：')
print('min cost = ', ass_by_per.min_cost)
print('best solution = ', ass_by_per.best_solution)
'''
print('匈牙利方法任务分配：')
print('min cost = ', ass_by_Hun.min_cost)
print('best solution = ', ass_by_Hun.best_solution)

def find_index(cost_matrix, result):

    index_list = []
    #print(len(result))
    for k in range(len(result)):
        for i in range(500):
            if result[k] == cost_matrix[k][i] and i < 500:
                index_list.append(i)
                break
        
    return index_list

def match_name(matrix, index_list,matrix_audio):
    match_dict = {}
    match_num = 0
    for k in range(len(index_list)):
        match_dict[matrix_audio[4 * k]] = matrix[4 * index_list[k]]
        a = matrix_audio[4 * k].split('/')
        b = matrix[4 * index_list[k]].split('/')
        if a[3] == b[3] and a[4] == b[4]:
            match_num += 1
    return match_dict , float(match_num) / float(len(match_dict))


index = find_index(L,ass_by_Hun.best_solution)
name, acc = match_name(b[0],index,b[1])
print(name)
print(acc)
print(len(name))


print(index)
print(len(index))
c = collections.Counter(index)
dictc = dict(c)
print(dictc)




