
def two_list2dict(file, A, B):
    result = {}
    for i in range(len(A)):
        result[A[i]] = B[i]
    # print(result)
    with open(file, 'w') as f:
        f.write(str(result))
