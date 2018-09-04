# -*- coding: utf-8 -*-
'''
# NO.49 使用lambda来创建匿名函数
MAXIMUM = lambda x,y :  (x > y) * x + (x < y) * y
MINIMUM = lambda x,y :  (x > y) * y + (x < y) * x
True*2 = 2   False*2 = 0

# NO.61 打印出杨辉三角形（要求打印出10行如下图）
print(1)
l = [1,1]
for i in range(2,10):
    print(l)
    ll = [1]
    for j in range(1,i):
        try:
            ll.append(l[j-1]+l[j])
        except:
            print('The error: ',i,j)
    ll.append(1)
    l = ll[:]
'''
# NO.69 有n个人围成一圈，顺序排号。从第一个人开始报数（从1到3报数），
# 凡报到3的人退出圈子，问最后留下的是原来第几号的那位。
n = 69
while n>1:

