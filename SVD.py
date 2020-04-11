import numpy as np
import sympy as sy

#Three examples to try : 
# A = np.array([[1, 1], [0, 1], [-1, 1]])
# A = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0]])
A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])


def svd_Implem(A):
    if type(A) != np.ndarray:
        raise Exception("Must be a np.ndarray type!!")
    A_trans = A.transpose()
    singular_values = np.sort(np.linalg.eigvals(np.dot(A_trans, A)))[::-1] ** 0.5
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    k = 0
    for i in range(Sigma.shape[0]):
        for j in range(Sigma.shape[1]):
            if i == j:
                Sigma[i, j] = singular_values[k]
                k += 1

    V_trans = np.linalg.eig(Sigma[:len(singular_values), :])[1][::-1]
    U = np.zeros((A.shape[0], A.shape[0]))
    for i in range(len(singular_values)):
        U[:, i] = 1 / singular_values[i] * np.dot(A, V_trans[:, i])

    u_rest = np.asarray(sy.Matrix(A_trans).nullspace())
    for i in range(A.shape[0] - len(singular_values)):
        temp = np.sum(u_rest[i] ** 2) ** 0.5
        U[:, len(singular_values) + i] = 1 / temp * u_rest[i]

    return U, Sigma, V_trans


#Let us try the results : 

U, Sigma, V_trans = svd_Implem(A)
ans = np.mat(U) * np.mat(Sigma) * np.mat(V_trans)
print(ans)
