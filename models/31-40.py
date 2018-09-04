# -*- coding: utf-8 -*-
'''
# NO.35 文本颜色设置
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
print bcolors.WARNING + "警告的颜色字体?" + bcolors.ENDC
'''
# NO.36  求100之内的素数
for i in range(2,101):
    flag = True
    for j in range(2,i):
        if i%j ==0:
            flag = False
            break
    if flag:
        print(i)
# NO.40 将一个数组逆序输出。
a=[1,43,5,6,3]
n = len(a)
[print(a[n-i-1]) for i in range(n)]


