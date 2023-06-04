def binary_search(nums, L, R, target):
    if (L > R):
        return -1
    mid = (L + R) // 2
    if nums[mid] == target:
        return mid
    if nums[mid] < target:
        return binary_search(nums, mid + 1, R, target)
    if nums[mid] > target:
        return binary_search(nums, 0, mid-1, target)


def search(nums, target):
    return binary_search(nums,0,len(nums)-1,target)

def removeDuplicates(s):
    res=[s[0]]
    size=1
    for i in range(1,len(s)):
        if res[size-1]==s[i]:
            res.pop(size-1)
            size-=1
        else:
            res.append(s[i])
            size+=1
    return ''.join(res)


def findShortestSubArray(nums):
    dic = {}
    maxi = 0
    res = len(nums)
    for i in range(len(nums)):
        if nums[i] not in dic:
            dic[nums[i]] = [1, i]
        else:
            dic[nums[i]][0] += 1

        if dic[nums[i]][0] == maxi:
            if (i - dic[nums[i]][1]) < res:
                res = i - dic[nums[i]][1]
        if dic[nums[i]][0] > maxi:
            maxi = dic[nums[i]][0]
            res = i +1- dic[nums[i]][1]

    return res
print(findShortestSubArray([2,1]))

