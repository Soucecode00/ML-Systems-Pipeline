def target(arr):
    left = 1
    for right in range(len(arr)-1):
        if arr[right] != arr[right+1]:
            arr[left] = arr[right+1]
            left=left+1
    return left
nums = [1,1,2,2,2,2,3,4,4,5]
print(target(nums))