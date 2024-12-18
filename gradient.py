import numpy as np


class Matrix:
    def __init__(self, size, experimental=False):
        self.size = size
        if experimental:
            self.matrix = np.array([[1, 2, 3],
                                    [3, 4, 5],
                                    [1, 2, 5]])
        else:
            self.matrix = np.random.rand(size[0], size[1])
        self.grad = None

    def __mul__(self, other):
        if isinstance(other, Vector):
            self.grad = np.tile(other.vector.T, (self.size[1], 1))
            return np.dot(self.matrix, other.vector)

    def __rmul__(self, other):
        if isinstance(other, Vector):
            return np.dot(other.vector.T, self.matrix)
            

class Vector:
    def __init__(self, size, experimental=False):
        self.size = size
        if experimental:
            self.vector = np.array([1, 2, 3])
        else:
            self.vector = np.random.rand(size)
        self.grad = None

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            self.grad = np.sum(other.matrix, axis=-1)
            return np.dot(other.matrix, self.vector)
        elif isinstance(other, Vector):
            self.grad = other.vector
        else: 
            self.grad = np.full(self.size, other)
            return other * self.vector
    
    def __mul__(self, other):
        if isinstance(other, Matrix):
            return np.dot(self.vector.T, other.matrix)
        return np.dot(self.vector.T, other)
    

if __name__ == "__main__":
    vec = Vector(3, experimental=True)
    v = 8
    mat1 = Matrix((3, 3), experimental=True)
    print(f"Scalar * Vector: {v * vec}, Vector grad: {vec.grad}")
    print(f"Vector * Matrix: {vec * mat1}, Vector grad: {vec.grad}")
    print(f"Matrix * Vector: {mat1 * vec}, Matrix grad: {mat1.grad}")
