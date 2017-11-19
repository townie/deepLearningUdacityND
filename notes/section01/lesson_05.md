# Matrix Math and Numpy Refresher


## Scalar

single value
ie. 1
Zero dimensions
ad

## Vector
a single row or column data
ie row vector [ 1, 2, ,3]
or column vector

```
[
1
2
3
]
```

## Matrix


```
[
1 2 3
2 4 5
3 6 7
]
```

Index in books start at 1... but programing of course starts at 0


# Scalar Math

# addition
[ 1 2 3 4 5 ] + 2 = [3 4 5 6 7]

[ 1 2 3 4 5 ] - 2 = [-1 0 1 2 3]

[ 1 2 3 4 5 ] % 10 = [0.1 0.2 0.3 0.4 0.5]


# as long as they are the same size you can add across

[1 2   + [1 2   = [2 4
3 4]      3 4]     6 8]




[ 1 2 3 4 5 ] * 2 = [2 4 6 8 10]



# Matrix Mulitplication

# element wise multiplication
must be the same size
[1 2   *  [1 2   = [1 4
3 4]      3 4]     9 16]


## if you matrix are the same size you cannt do matrix product!


Caluclate the product of matrix

but lets start on a vector

# dot product

[ 0 2 4 6 ] [ 1 7 13 19]

Apply multiplcation element wise and then add

0 X 1
2 x 7
4 X 13
6 X 19

0 + 14 + 52 + 114 = 180

output is a scalar


# calculating the dot product on unbalanced matrix

ROWS in first matrix
COLUMNS in second matrix

so a [2X4] matrix dot'd with a [4X3] = output matrix == [2X3]

# NUMBER OF ROWS MUST EQUAL TH E NUMBER OF COULUNS

left # of rows == right # columns



Matrix mulitplication is not commutative

Important Reminders About Matrix Multiplication
The number of columns in the left matrix must equal the number of rows in the right matrix.
The answer matrix always has the same number of rows as the left matrix and the same number of columns as the right matrix.
Order matters. Multiplying A•B is not the same as multiplying B•A.
Data in the left matrix should be arranged as rows., while data in the right matrix should be arranged as columns.







