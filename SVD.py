import numpy as np

# Created by Raviv Herrera 
#A = np.array([[4, 1, 3], [8, 3, -2]])
#A = np.array([[1, 1], [0, 1], [-1, 1]])
#A = np.array([[2, 2], [-1, 1]])
A = np.array([[122, 534, 2], [1, 640, 9], [4, 1, 1]])
def mySVD(A):
    if type(A) != np.ndarray:
        Exception("Data must be a np.ndarray type")
    else:
        S = np.zeros((A.shape[0], A.shape[1]))
        singular_values, v = np.linalg.eig(np.dot(A.transpose(), A))
        singular_values_sorted = np.sort(singular_values)[::-1] ** 0.5
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == j:
                    S[i, j] = singular_values_sorted[i]

        V = v[:, singular_values.argsort()[::-1]]
        su, u = np.linalg.eig(np.dot(A, A.transpose()))
        U = u[:, su.argsort()[::-1]]

        return U, S, V.T


U, S, V = mySVD(A)
print("mySVD U is : \n", np.round(U, 3), f"\n\n mySVD {chr(931)} is : \n", np.round(S, 3), "\n\n mySVD V' is : \n",
      np.round(V, 3), "\n")
u1, s1, v1 = np.linalg.svd(A)
print("NUMPY U is : \n", np.round(u1, 3), f"\n\n NUMPY {chr(931)} is : \n", np.round(s1, 3), "\n\n NUMPY V' is : \n",
      np.round(v1, 3))
"\n\n mySVD V' is : \n", np.round(V.T, 3)
#Let us try the results : 
print("The Matrix A is : \n", np.round(np.dot(U, S).dot(V), 3))
