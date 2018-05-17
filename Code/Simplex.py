
# coding: utf-8

# In[1]:

# min c`x
# Ax=b
# x>=0

# A: The constraint matrix 
# b: The rhs vector 
# c: The vector of cost coefficients 
# C: The indices of the basic variables corresponding to an initial basic feasible solution


# In[2]:

import numpy as np
import copy
from sklearn.svm import SVC


# In[3]:

#a vector is by default a column vector

def mySimplex(A, b, c, C):
    for i in range(len(C)):
        C[i] = C[i]-1

    print("A = \n", A, A.shape)
    print("b = \n", b, b.shape)
    print("c = \n", c, c.shape)
    print("C = \n", C, C.shape)

    ##1

    x = np.zeros(c.shape)
    # print("x = \n", x, x.shape)

    B = A[:, C]
    print("B = \n", B, B.shape)

    Binv = np.linalg.inv(B)
    print("Binv = \n", Binv, Binv.shape)

    x[C] = np.matmul(Binv, b)
    print("x = \n", x, x.shape)

    k = 1

    while (1):
        print("-------------------------------------------------------")
        print("\nIteration ", k)
        print("\n")
        k = k+1

        ##2
        cB = c[C]
        print("cB = \n", cB, cB.shape)
        print("Binv = \n", Binv)

        p = (np.matmul(cB.transpose(), Binv)).transpose()
        print("p = \n", p, p.shape)

        flag = 0
        for j in range((A.shape)[1]):
            if j not in C:
                cj_ = c[j] - np.matmul(p.transpose(), A[:, j])
                if (cj_ < 0):
                    flag = 1
                    break

        if (flag == 0):
            print("\n\n", x, "is the optimal ans and the cost is", np.matmul(c.transpose(), x)[0][0], "\n")
            break

        Aj = np.array([A[:, j]]).transpose()
        print("j = \n", j)
        print("Aj = \n", Aj, Aj.shape)

        ##3

        u = np.matmul(Binv, Aj)
        print("u = \n", u, u.shape)

        flag = 0
        for i in range(u.shape[0]):
            if (u[i] >= 0):
                flag = 1
        if (flag == 0):
            print("\n\n", x + "is the optimal ans and the cost is -inf \n")
            break

        ##4

        minim = 1e9+7
        l = 1e9+7 #index at which we get min

        for i in range(u.shape[0]):
            if (u[i] > 0):
                val = x[C[i]]/u[i]
                if (minim > val):
                    minim = val
                    l = i

        theta = minim
        print("theta = \n", theta)
        print("l = \n", l)

        ##5

        B_new = B
        B_new[:,l] = A[:,j]
        print("B_new = \n", B_new, B_new.shape)

        y = np.zeros(x.shape)
        y[j] = theta

        for i in range(u.shape[0]):
            if i != l:
                y[C[i]] = x[C[i]] - np.matmul(theta, u[i])

        print("x_new = \n", y, y.shape)

        ##6

        B_inv = np.zeros(Binv.shape)
        for i in range(Binv.shape[0]):
            rep = Binv[:, l]
            tmp = Binv[i] - ((u[i] - rep[i])/u[l])*Binv[l];
            B_inv[i] = tmp
        print("Binv_new = \n", B_inv, B_inv.shape)

        C[l] = j
        print("C = \n", C, C.shape)

        #Assign original vars
        B = B_new
        x = y
        Binv = B_inv

    a = x[0:30, :]
    b = x[31, :]
    y_found = np.zeros((688, 1))
    count = 0
    print(a.shape, x_true.shape, b.shape)
    for i in range(688):
    	y_found[i] = np.matmul(a.transpose(), x_true[i, :]) + b
    	if (y_found[i] == y_true[i]):
    		count += 1
    np.save("Q_Labels.npy", y_found)
    print("Accuracy =", (count/(688)))

# In[4]:

n = 31
m = 688

x_true = np.load('DB_Vecs.npy')
y_true = np.load('DB_Labels.npy')
x_find = np.load('Q_vecs.npy')

A = np.zeros((m,n))

for i in range(m):
    # print(x_true[i])
    tmp = np.insert(x_true[i], x_true[i].shape[0], 1)
    A[i, :] = tmp
    # print(A[i, :].shape)
    if (y_true[i] == 1):
        for s in range(A[i, :].shape[0]):
            A[i, s] = -A[i, s]

print(A, "\n")
# exit()

b = np.array([[s for s in y_true]]).transpose()
print(b, "\n")
# exit()
s = (688+31, 1)
c = np.ones(s)
for i in range(n):
	c[i] = 0
print(c)

# print('''min c`x
#     subject to:
#         Ax=b
#         x>=0''')
# print("Enter no. of coeff (n): ")
# n = int(input())
# print("Enter no. of constraints (m): ")
# m = int(input())

# A = np.zeros((m,n))

# print("Enter A: ")
# for i in range(m):
#     A[i, :] = [int(x) for x in input().split()]
# print(A, "\n")

# print("Enter b: ")
# b = np.array([[int(x) for x in input().split()]]).transpose()
# print(b, "\n")

# print("Enter c: ")
# c = np.array([[int(x) for x in input().split()]]).transpose()
# print(c, "\n")

# print("Enter C(Col for basis matrix): ")
# C = np.array([int(x) for x in input().split()])

C = np.array([i+n+1 for i in range(m)])
print(C, "\n")
# exit()

# In[5]:

#Primal Problem
def findPrimal(A, b, c, C):
    I = np.identity(A.shape[0])
    A = np.concatenate([A, I], axis=1)
    # c = np.concatenate([c, np.zeros((A.shape[0], 1))], axis=0)

    # for i in range(n, c.shape[0]):
    # 	c[i] = 1e9+7

#     print(A)
    # print(c)
    # print(c.shape)
    mySimplex(A, b, c, C)

# In[6]:

def findDual(A, b, c, C):
    print('''max p`b
subject to
    p`A = c`''')
    
#     p = np.zeros((A.transpose().shape[0], 1))
#     print(p)

    dualA = A.transpose()
    dualb = c
    dualc = np.array([[-1*(i[0]) for i in b]]).transpose()
    
    # print(dualA)
#     print(dualb)
#     print(dualc)
    
    I = np.identity(dualA.shape[0])
    dualA = np.concatenate([dualA, I], axis=1)
    dualc = np.concatenate([dualc, np.zeros((dualA.shape[0], 1))], axis=0)
#     print(dualA)
#     print(dualc)
    C = np.array([(i+m+1) for i in range(dualA.shape[0])])
    
    mySimplex(dualA, dualb, dualc, C)


# In[9]:

while(1):
    print("Primal(0) / Dual(1): ")

    option = int(input())

    if (option == 0):
        findPrimal(A, b, c, copy.deepcopy(C))
    elif (option == 1):
        findDual(A, b, c, copy.deepcopy(C))

