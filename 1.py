import math 
import numpy as np

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
            if v[2+4*i] != a[2+4*j]:
                L[i][j] += 20
            if v[3+4*i] != a[3+4*j]:
                L[i][j] += 20
            L[i][j] += (v[1+4*i]-a[1+4*j]) * (v[1+4*i]-a[1+4*j])
    
    return L
    

def create_matrix():
    # 首先读取两个大矩阵，其目的是为了构造audio-video cost matrix
    v_a = loadData('./CollisionRecognition/task2_vedio_angle.txt')
    v_c = loadData('./CollisionRecognition/task2_vedio_category.txt')
    v_i = loadData('./CollisionRecognition/task2_vedio_ismove.txt')
    #print(len(v_a))

    #print(v_a[0][0])
    audio_a = loadData('./CollisionRecognition/task2_audio_angle.txt')
    audio_c = loadData('./CollisionRecognition/task2_audio_category.txt')
    audio_i = loadData('./CollisionRecognition/task2_audio_ismove.txt') 
      
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

print(len(L[0]))