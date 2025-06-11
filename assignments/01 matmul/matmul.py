"""
Solves the problem statement defined in Assignment 1 (Matrix Multiplication).

Author:
    Kaushik G Iyer (EE23B135)
"""

from typing import List, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")
Matrix = Sequence[Sequence[T]]

Number = Union[int, float, complex]


def validate_matrix(matrix: Matrix[Number]) -> Tuple[int, int]:
    """
    Checks whether the given matrix is a valid matrix (with numeric elements).

    Parameters
    ----------
    matrix : Matrix
        The matrix to be validated. A matrix is defined as follows:
            - A matrix is a finite iterable of rows with the same size.
            - A matrix must have atleast 1 row and 1 column.

        Every element of `matrix` must be numeric (either a float, an int or a complex)

    Returns
    -------
    tuple[int, int]
        The dimensions of `matrix`

    Raises
    ------
    ValueError
        - If the object is not a matrix
    TypeError
        - If the matrix contains non numeric elements
    """
    # Ensure that the given object is a list or tuple
    if not isinstance(matrix, (list, tuple)):
        raise ValueError("Value provided is not a valid matrix!")

    # Every row in it must also be a list or tuple
    if not all(isinstance(row, (list, tuple)) for row in matrix):
        raise ValueError("Value provided is not a valid matrix!")

    rows = len(matrix)
    # A matrix must have atleast 1 row
    if rows == 0:
        raise ValueError("Matrix has invalid dimensions!")

    cols = len(matrix[0])
    # A matrix must have atleast 1 column
    # All rows must have the same number of elements
    if cols == 0 or not all(len(row) == cols for row in matrix):
        raise ValueError("Matrix has invalid dimensions!")

    # Every element must be numeric (Either a float, an int or a complex)
    if not all(
        all(isinstance(value, (int, float, complex)) for value in row)  # type: ignore
        for row in matrix
    ):
        raise TypeError("Value provided contains elements that are not numeric!")

    # The type ignore above is because pylance felt the check to be uneccessary
    # (Since the input to this function is a Matric[Number] anyways)

    # The dimensions of the given matrix
    return rows, cols


def matrix_multiply(matrix1: Matrix[Number], matrix2: Matrix[Number]) -> Matrix[Number]:
    """
    Multiplies two matrices of suitable dimensions

    Parameters
    ----------
    matrix1 : Matrix[Number]
        The lhs of the multiplication (say with dimensions `N x M`)

    matrix2 : Matrix[Number]
        The rhs of the multiplication (say with dimensions `M x P`)

    Returns
    -------
    Matrix[Number]
        The resultant matrix (with dimensions `N x P`)

    Raises
    ------
    ValueError
        - If either of the objects are not a matrix
        - If the matrices have incompatible dimensions for multiplication
    TypeError
        - If either of the matrices contain non numeric elements
            (refer to validate_matrix for more information on this)
    """
    # Validate that both matrices are indeed matrices with numeric elements
    dimension1 = validate_matrix(matrix1)
    dimension2 = validate_matrix(matrix2)

    # To multiply two matrices they must have dimension of the form `N x M` and `M X P`
    if dimension1[1] != dimension2[0]:
        raise ValueError("Matrices have incompatible dimensions!")

    # The dimensions of the resultant matrix
    resultant_dimension = (dimension1[0], dimension2[1])

    # The index over which we iterate through while doing the multiplication
    iteration_index = dimension1[1]

    # The resultant matrix (Initially with all values set to 0)
    result: List[List[Number]] = [
        [0] * resultant_dimension[1] for _ in range(resultant_dimension[0])
    ]

    # NOTE:
    # If matrix1 or matrix2 had complex elements, result would contain complex elements
    # If there were no complex elements but, float elements result would contain floats
    # If there were no complex or float elements, result would just contain ints :)

    # Your standard matrix multiplication algorithm
    for i in range(resultant_dimension[0]):
        for j in range(resultant_dimension[1]):
            # Set the value of result[i][j]
            for k in range(iteration_index):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    # As for float errors and all, no rounding is done here.
    # It is up to the user who uses these resultant matrices
    # to deal with floating point errors.
    # They may check approximate equality(like numpy.isclose) instead of strict equality

    return result
