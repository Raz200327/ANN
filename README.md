# TensorGrad

A lightweight automatic differentiation system inspired by PyTorch's autograd and Standford's gradient notes (https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf), implementing basic matrix and vector operations with gradient tracking.

## Features
- Matrix-Vector multiplication with gradient computation
- Vector-Matrix multiplication with gradient computation
- Scalar-Vector multiplication with gradient computation
- NumPy backend for efficient computation

## Quick Start
```python
# Create vector and matrix
vec = Vector(3)
mat = Matrix((3, 3))

# Perform operations
result1 = 8 * vec       # Scalar-Vector multiplication
result2 = vec * mat     # Vector-Matrix multiplication
result3 = mat * vec     # Matrix-Vector multiplication

# Access gradients
print(vec.grad)         # Vector gradient
print(mat.grad)         # Matrix gradient
```

## Dependencies
- NumPy

Work in progress - more operations and features coming soon.