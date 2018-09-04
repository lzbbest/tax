# -*- coding: utf-8 -*-
'''
# NO.24 有一分数序列：2/1，3/2，5/3，8/5，13/8，21/13...求出这个数列的前20项之和。
up = [2,3]
down = [1,2]
sum = 3.5
for i in range(2,20):
    up.append(up[i-1]+down[i-1])
    down.append(up[i-1])
    sum += up[i]/down[i]
print(sum)

# NO.25 求1+2!+3!+...+20!的和
sum,t = 0,1
for i in range(1,21):
    t = t*i
    sum +=t
print(sum)

# NO.26 利用递归方法求5!
def fact(j):
    sum = 0
    if j == 0:
        sum = 1
    else:
        sum = j * fact(j - 1)
    return sum
print(fact(5))
'''
# NO.30 一个5位数，判断它是不是回文数。即12321是回文数，个位与万位相同，十位与千位相同
a,b=98765,12321
def func(x):
    x = list(str(x))
    if x[0] == x[4] and x[1] == x[3]:
        print('T')
    else:
        print('F')
func(a)









