# SVD-Implemantation
A python implemantation of the Singular Values Decomposition (SVD) 
A pure implemantation without the built-in function from linalg library svd().

## Theory 
A little bit theory :
The SVD is a remarkable matrices decomposition using to several uses such as :
1) Rigid transformation .
2) In Machine Learning field - PCA algorithm to reduce number of features .
3) Statistics field .

# The 3 returned matrices form the decomposition are :
1) U - An orthogonal matrix , considered as a rotation matrix.
2) Σ - A diagonal matrix , considered as a stretching matrix and contains the singular values of the original matrix in it's diagonal .
3) V^t - An orthogonal matrix , considered as a rotation matrix as well as the U matrix . 

The fact that the Σ matrix holds the singular values in it's main diagonal from the biggest value to the smallest value ,in addition to the fact that U and V^t are both orthogonal we can clearly get the top features or in other words the heavyest components and so to achive the top features of a data , that's all behind the SVD takes place in PCA algorithm .   

## The function it self : 

svd_Implem(A)

takes one argument -> matrix we want to decompose . 
and returns the 3 matrices mentioned above . 
