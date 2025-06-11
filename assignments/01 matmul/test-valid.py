import random
from numpy import matrix

from matmul import matrix_multiply
for _ in range(1, 102):
    print(_)
    m = random.randint(5, 50)
    n = random.randint(5, 50)
    p = random.randint(5, 40)
    A = (
        [[
            random.randint(5, 100) // 10 for _ in range(p)
        ] for _ in range(n)]
    )
    B = (
        [[
            random.randint(100, 300) for _ in range(m)
        ] for _ in range(p)]
    )

    res = matrix_multiply(A, B)
    
    if not ((matrix(res) == matrix(A) * matrix(B)).all()):
        print(A)
        print(B)
        break