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

def matmul_AB_computation(shapeA, shapeB, bn, dtype="float32"):
    # describe
    assert shapeA[1] == shapeB[0]
    k = tvm.reduce_axis((0, shapeA[1]), name="k")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    packedB = tvm.compute((shapeB[1] / bn, shapeB[0], bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    MATMUL = tvm.compute((shapeA[0], shapeB[1]), lambda i, j: tvm.sum(A(i, k) * packedB(j / bn, k, j % bn), axis=k), name="MATMUL")

    return A, B, packedB, MATMUL

def matmul_ATB_computation(shapeA, shapeB, bn, dtype="float32"):
    # describe
    assert shapeA[0] == shapeB[0]
    k = tvm.reduce_axis((0, shapeA[0]), name="k")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    # packedB = tvm.compute((shapeB[1] / bn, shapeB[0], bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    # MATMUL = tvm.compute((shapeA[1], shapeB[1]), lambda i, j: tvm.sum(A(k, i) * packedB(j / bn, k, j % bn), axis=k), name="MATMUL")
    MATMUL = tvm.compute((shapeA[1], shapeB[1]), lambda i, j: tvm.sum(A(k, i) * B(k, j), axis=k), name="MATMUL")

    return A, B, MATMUL

def matmul_ABT_computation(shapeA, shapeB, bn, dtype="float32"):
    # describe
    assert shapeA[1] == shapeB[1]
    k = tvm.reduce_axis((0, shapeA[1]), name="k")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    # packedB = tvm.compute((shapeB[1], shapeB[0] / bn, bn), lambda x, y, z: B[y * bn + z, x], name="packedB")
    # MATMUL = tvm.compute((shapeA[0], shapeB[0]), lambda i, j: tvm.sum(A(i, k) * packedB(k, j / bn, j % bn), axis=k), name="MATMUL")
    MATMUL = tvm.compute((shapeA[0], shapeB[0]), lambda i, j: tvm.sum(A(i, k) * B(j, k), axis=k), name="MATMUL")

    return A, B, MATMUL

def matmul_ATBT_computation(shapeA, shapeB, bn, dtype="float32"):
    # describe
    assert shapeA[0] == shapeB[1]
    k = tvm.reduce_axis((0, shapeA[0]), name="k")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    # packedB = tvm.compute((shapeB[1], shapeB[0] / bn, bn), lambda x, y, z: B[y * bn + z, x], name="packedB")
    # MATMUL = tvm.compute((shapeA[1], shapeB[0]), lambda i, j: tvm.sum(A(k, i) * packedB(k, j / bn, j % bn), axis=k), name="MATMUL")
    MATMUL = tvm.compute((shapeA[1], shapeB[0]), lambda i, j: tvm.sum(A(k, i) * B(j, k), axis=k), name="MATMUL")

    return A, B, MATMUL

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    bn = 32
    packedB = None

    if transposeA is False and transposeB is False:
        A, B, packedB, C = matmul_AB_computation(shapeA, shapeB, bn, dtype)
    elif transposeA is True and transposeB is False:
        A, B, C = matmul_ATB_computation(shapeA, shapeB, bn, dtype)
    elif transposeA is False and transposeB is True:
        A, B, C = matmul_ABT_computation(shapeA, shapeB, bn, dtype)
    else: # transposeA is True and transposeB is True:
        A, B, C = matmul_ATBT_computation(shapeA, shapeB, bn, dtype)

    # schedule
    s = tvm.create_schedule(C.op)

    if transposeA is False and transposeB is False:
        # allocate write cache
        CC = s.cache_write(C, 'global')

        ## TILE
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

        # write cache is computed at yo
        s[CC].compute_at(s[C], yo)

        # new inner axes
        xc, yc = s[CC].op.axis

        k, = s[CC].op.reduce_axis
        ko, ki = s[CC].split(k, factor=4)

        s[CC].reorder(ko, xc, ki, yc)
        s[CC].unroll(ki)
        s[CC].vectorize(yc)

        ## PARALLELIZE
        s[C].parallel(xo)

        ## PACK
        x, y, z = s[packedB].op.axis
        s[packedB].vectorize(z)
        s[packedB].parallel(x)
    # TODO: optimize other cases

    # compile
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

    return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    # adapted from http://docs.tvmlang.org/tutorials/optimize/opt_conv_cuda.html

    stride = 1
    padding = 0

    # extract dimensions
    batch, inChannel, H, W = shapeX
    filter_outChannel, filter_inChannel, filter_H, filter_W = shapeF
    assert inChannel == filter_inChannel

    # output shape is (batch, filter_outChannel, out_H, out_W)
    assert (H + 2 * padding - filter_H) % stride == 0
    assert (W + 2 * padding - filter_W) % stride == 0
    out_H = (H + 2 * padding - filter_H) // stride + 1
    out_W = (W + 2 * padding - filter_W) // stride + 1
    
    X = tvm.placeholder(shapeX, dtype=dtype, name="X")
    F = tvm.placeholder(shapeF, dtype=dtype, name="F")

    # pad input
    # assumed pad = 0 so not implemented

    # reduction vars
    rc = tvm.reduce_axis((0, inChannel), name='rc')
    ry = tvm.reduce_axis((0, filter_H), name='ry')
    rx = tvm.reduce_axis((0, filter_W), name='rx')

    # compute
    OUT = tvm.compute(
        (batch, filter_outChannel, out_H, out_W),
        lambda nn, ff, yy, xx: tvm.sum(
            X[nn, rc, yy * stride + ry, xx * stride + rx] * F[ff, rc, ry, rx],
            axis=[ry, rx, rc]
        ),
        name="OUT"
    )

    # schedule
    s = tvm.create_schedule(OUT.op)

    # compile
    f = tvm.build(s, [X, F, OUT], tgt, target_host=tgt_host, name=func_name)

    return f

def softmax_computation(X, shape, dtype="float32"): 
    km = tvm.reduce_axis((0, shape[1]), name="km")
    MAX = tvm.compute((shape[0],), lambda i: tvm.max(X(i, km), axis=km), name="MAX")
    
    E_X = tvm.compute(shape, lambda i, j: tvm.exp(X(i, j) - MAX(i)), name="E_X")
    
    ks = tvm.reduce_axis((0, shape[1]), name="ks")
    SUM = tvm.compute((shape[0],), lambda i: tvm.sum(E_X(i, ks), axis=ks), name="SUM")
    
    SOFTMAX = tvm.compute(shape, lambda i, j: E_X(i, j) / SUM(i), name="SOFTMAX")

    return MAX, E_X, SUM, SOFTMAX

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    X = tvm.placeholder(shape, dtype=dtype, name="X")
    _, _, _, SOFTMAX = softmax_computation(X, shape, dtype)

    # schedule
    s = tvm.create_schedule(SOFTMAX.op)

    # compile
    f = tvm.build(s, [X, SOFTMAX], tgt, target_host=tgt_host, name=func_name)

    return f

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    X = tvm.placeholder(shape, dtype=dtype, name="X")
    _, _, _, SOFTMAX = softmax_computation(X, shape, dtype)

    # cross_entropy = np.mean(
        # -np.sum(y_ * np.log(autodiff.softmax_func(y)), axis=1), keepdims=True)

    Y = tvm.placeholder(shape, dtype=dtype, name="Y")

    MUL_LOG = tvm.compute(shape, lambda *i: Y(*i) * tvm.log(SOFTMAX(*i)), name="MUL_LOG")

    k1 = tvm.reduce_axis((0, shape[1]), name="k1")
    SUM1 = tvm.compute((shape[0],), lambda i: tvm.sum(-MUL_LOG(i, k1), axis=k1), name="SUM1")

    k2 = tvm.reduce_axis((0, shape[0]), name="k2")
    SUM2 = tvm.compute((1,), lambda _: tvm.sum(SUM1(k2), axis=k2), name="SUM2")

    OUT = tvm.compute((1,), lambda i: SUM2(i) / shape[0])

    # schedule
    s = tvm.create_schedule(OUT.op)

    # compile
    f = tvm.build(s, [X, Y, OUT], tgt, target_host=tgt_host, name=func_name)

    return f

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
