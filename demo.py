def dort(x):
    result = [0] * len(x)
    a = len(x)-1
    left = 0
    right = len(x)-1
    while (left <= right):
        if abs(x[left]) < abs(x[right]):
            result[a] = x[right] * x[right]
            right = right-1
        else:
            result[a] = x[left] * x[left]
            left = left +1 
        a = a - 1
    return result 
result = [-7,-3,2,3,11]
print(dort(result)) 

