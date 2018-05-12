from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""

    # describe
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    # schedule
    s = tvm.create_schedule(C.op)

    # compile
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""

    # describe
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.compute(A.shape, lambda *i: A(*i) + const_k)

    # schedule
    s = tvm.create_schedule(B.op)

    # compile
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)

    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""

    # describe
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.compute(A.shape, lambda *i: A(*i) * const_k)

    # schedule
    s = tvm.create_schedule(B.op)

    # compile
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)

    return f


def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""

    # describe
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    zero = tvm.const(0, A.dtype)
    B = tvm.compute(A.shape, lambda *i: tvm.max(A(*i), zero))

    # schedule
    s = tvm.create_schedule(B.op)

    # compile
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)

    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    # 1 if > 0 else 0

    # describe
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    zero = tvm.const(0, A.dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.select(A(*i) > zero, B(*i), zero))

    # schedule
    s = tvm.create_schedule(C.op)

    # compile
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

    return f


def make_matrix_mul_AB(shapeA, shapeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    # describe
    assert shapeA[1] == shapeB[0]
    k = tvm.reduce_axis((0, shapeA[1]), name="k")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    C = tvm.compute((shapeA[0], shapeB[1]), lambda i, j: tvm.sum(A(i, k) * B(k, j), axis=k))
    
    # schedule
    s = tvm.create_schedule(C.op)

    # compile
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

    return f

def make_matrix_mul_ATB(shapeA, shapeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    # describe
    assert shapeA[0] == shapeB[0]
    k = tvm.reduce_axis((0, shapeA[0]), name="k")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    C = tvm.compute((shapeA[1], shapeB[1]), lambda i, j: tvm.sum(A(k, i) * B(k, j), axis=k))
    
    # schedule
    s = tvm.create_schedule(C.op)

    # compile
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

    return f

def make_matrix_mul_ABT(shapeA, shapeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    # describe
    assert shapeA[1] == shapeB[1]
    k = tvm.reduce_axis((0, shapeA[1]), name="k")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    C = tvm.compute((shapeA[0], shapeB[0]), lambda i, j: tvm.sum(A(i, k) * B(j, k), axis=k))
    
    # schedule
    s = tvm.create_schedule(C.op)

    # compile
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

    return f

def make_matrix_mul_ATBT(shapeA, shapeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    # describe
    assert shapeA[0] == shapeB[1]
    k = tvm.reduce_axis((0, shapeA[0]), name="k")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    C = tvm.compute((shapeA[1], shapeB[0]), lambda i, j: tvm.sum(A(k, i) * B(j, k), axis=k))
    
    # schedule
    s = tvm.create_schedule(C.op)

    # compile
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

    return f

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    
    if (transposeA == False) and (transposeB == False):
        return make_matrix_mul_AB(shapeA, shapeB, tgt, tgt_host,
                    func_name, dtype)
    elif (transposeA == True) and (transposeB == False):
        return make_matrix_mul_ATB(shapeA, shapeB, tgt, tgt_host,
                    func_name, dtype)
    elif (transposeA == False) and (transposeB == True):
        return make_matrix_mul_ABT(shapeA, shapeB, tgt, tgt_host,
                    func_name, dtype)
    else: # if (transposeA == True) and (transposeB == True):
        return make_matrix_mul_ATBT(shapeA, shapeB, tgt, tgt_host,
                    func_name, dtype)


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f