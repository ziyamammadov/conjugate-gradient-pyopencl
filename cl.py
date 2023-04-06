import numpy as np
import pyopencl as cl
IS_COMPLEX = True
LOCAL_SIZE = 256
WAVE_SIZE = 32
MAX_ITERATIONS = 10
FOLDER_PATH = './kernel/complex/' if IS_COMPLEX == True else './kernel/real/'
NP_TYPE = np.dtype(np.csingle if IS_COMPLEX else np.intc)
VAL_SIZE = NP_TYPE.itemsize
INCLUDE_FILE = "-I ./kernel/complex " if IS_COMPLEX else ""


def create_kernels(n_rhs):
    global axpy_kernel, aypx_kernel, spmv_kernel, sub_kernel, dot_kernel, ctx, queue
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    options = [
        f"{INCLUDE_FILE} -D N_RHS={n_rhs} -D WAVE_SIZE={WAVE_SIZE} -D WG_SIZE={LOCAL_SIZE}"]

    with open(f'{FOLDER_PATH}axpy.cl', 'r') as f:
        axpy_kernel = cl.Program(ctx, f.read()).build(options=options).axpy

    with open(f'{FOLDER_PATH}aypx.cl', 'r') as f:
        aypx_kernel = cl.Program(ctx, f.read()).build(options=options).aypx

    with open(f'{FOLDER_PATH}spmv.cl', 'r') as f:
        spmv_kernel = cl.Program(ctx, f.read()).build(options=options).spmv

    with open(f'{FOLDER_PATH}sub.cl', 'r') as f:
        sub_kernel = cl.Program(ctx, f.read()).build(options=options).sub

    with open(f'{FOLDER_PATH}vdot.cl', 'r') as f:
        dot_kernel = cl.Program(ctx, f.read()).build(options=options).vdot


def CG(size, non_zeros, a_values, b_values, a_pointers, a_cols, x, n_rhs, n_iterations):
    work_groups = 1 + ((size - 1) // LOCAL_SIZE)
    global_size = size if work_groups == 1 else work_groups * LOCAL_SIZE
    local_size = global_size if work_groups == 1 else LOCAL_SIZE
    rows_per_wg = (LOCAL_SIZE // WAVE_SIZE)
    spmv_work_groups = 1 + ((size - 1) // rows_per_wg)
    spmv_global_size = spmv_work_groups * LOCAL_SIZE
    spmv_local_size = LOCAL_SIZE

    # Allocate device memory and copy host arrays to device
    mf = cl.mem_flags
    int_size = np.dtype(np.intc).itemsize
    a_values_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=non_zeros * VAL_SIZE, hostbuf=a_values)
    a_cols_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=non_zeros * int_size, hostbuf=a_cols)
    a_pointers_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=(size + 1) * int_size, hostbuf=a_pointers)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                      size=n_rhs * size * VAL_SIZE,
                      hostbuf=b_values)
    x_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                      size=n_rhs * size * VAL_SIZE,
                      hostbuf=x)
    r_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * VAL_SIZE)
    d_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * VAL_SIZE)
    q_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * VAL_SIZE)
    dot_res_buf = cl.Buffer(
        ctx, mf.READ_WRITE, size=n_rhs * work_groups * VAL_SIZE)
    const_buf = cl.Buffer(ctx, mf.READ_ONLY, size=n_rhs * VAL_SIZE)
    dot_loc_mem = cl.LocalMemory(n_rhs * local_size * VAL_SIZE)
    spmv_cl_loc_mem = cl.LocalMemory(n_rhs * spmv_local_size * VAL_SIZE)

    # y = A * x                   (spmv)
    waitKSpmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                            a_pointers_buf, a_cols_buf, x_buf, q_buf, spmv_cl_loc_mem)

    # r = b - y                   (sub)
    waitKSub = sub_kernel(queue, (global_size,), (local_size,),
                          b_buf, q_buf, r_buf, np.intc(size), wait_for=[waitKSpmv])
    b_buf.release()

    # d = r                       (copy)
    waitCopy = cl.enqueue_copy(queue, d_buf, r_buf, wait_for=[waitKSub])

    # deltaNew = r^T * r               (dot)
    waitKDot = dot_kernel(queue, (global_size,), (local_size,), r_buf,
                          r_buf, dot_loc_mem, dot_res_buf, np.intc(size), wait_for=[waitCopy])

    h_dot_res = np.zeros(n_rhs * work_groups, dtype=NP_TYPE)
    delta_new = np.zeros(n_rhs, dtype=NP_TYPE)
    alpha = np.zeros(n_rhs, dtype=NP_TYPE)
    cl.enqueue_copy(queue, h_dot_res, dot_res_buf, wait_for=[waitKDot])

    # Reduction
    for r in range(n_rhs):
        for i in range(work_groups):
            delta_new[r] += h_dot_res[r * work_groups + i]

    delta_new = np.nan_to_num(delta_new, nan=0, posinf=0, neginf=0)[0]
    delta_old = delta_new
    # print(f'Delta new: {delta_new}')

    for iteration in range(n_iterations):
        # q = A * d (spmv)
        # q = np.zeros(n_rhs*size, dtype = NP_TYPE)
        if iteration == 0:
            waitKSpmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                                    a_pointers_buf, a_cols_buf, d_buf, q_buf, spmv_cl_loc_mem)
        else:
            waitKSpmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                                    a_pointers_buf, a_cols_buf, d_buf, q_buf, spmv_cl_loc_mem, wait_for=[waitKAypx])
            
        # cl.enqueue_copy(queue, q, q_buf, wait_for = [waitKSpmv])
        # print(f'Q res: {q[2600]}')

        # dq = d * q(dot)
        waitKDot = dot_kernel(queue, (global_size,), (local_size,), d_buf,
                              q_buf, dot_loc_mem, dot_res_buf, np.intc(size), wait_for=[waitKSpmv])
        cl.enqueue_copy(queue, h_dot_res, dot_res_buf,
                        wait_for=[waitKDot]).wait()
        h_dq = np.zeros(n_rhs, dtype=NP_TYPE)
        for r in range(n_rhs):
            # Reduction
            for i in range(work_groups):
                h_dq[r] += h_dot_res[r * work_groups + i]
        h_dq = np.nan_to_num(h_dq, nan=0, posinf=0, neginf=0)[0]
        alpha = delta_new/h_dq

        # print(f'Alpha: {alpha}')

        cl.enqueue_copy(queue, const_buf, np.csingle(
            alpha), is_blocking=True).wait()

        # x = alpha * d + x(axpy)
        axpy_kernel(queue, (global_size,), (local_size,), d_buf,
                    x_buf, const_buf, np.intc(1), np.intc(size))
        # print(f'X: {x}')

        # r = - alpha * q + r(axpy)
        waitKAxpy = axpy_kernel(queue, (global_size,), (local_size,), q_buf,
                                r_buf, const_buf, np.intc(0), np.intc(size))
        delta_old = delta_new
        # print(f'R: {x}')

        # deltaNew = r ^ T * r(dot)
        waitKDot = dot_kernel(queue, (global_size,), (local_size,), r_buf,
                              r_buf, dot_loc_mem, dot_res_buf, np.intc(size), wait_for=[waitKAxpy])
        cl.enqueue_copy(queue, h_dot_res, dot_res_buf, wait_for=[waitKDot])

        # Reduction
        delta_new = np.zeros(n_rhs, dtype=NP_TYPE)
        for r in range(n_rhs):
            for i in range(work_groups):
                delta_new[r] += h_dot_res[r * work_groups + i]
        delta_new = np.nan_to_num(delta_new, nan=0, posinf=0, neginf=0)[0]

        # print(f'Delta new {iteration}: {delta_new}')
        # print(f'Delta old {iteration}: {delta_old}')

        beta = delta_new/delta_old

        # print(f'Beta: {beta}')

        # d = beta * d + r(aypx)
        cl.enqueue_copy(queue, const_buf, np.csingle(
            beta), is_blocking=True)
        waitKAypx = aypx_kernel(queue, (global_size,), (local_size,), r_buf,
                                d_buf, const_buf, np.intc(size))
        # print(f'D: {x}')

    cl.enqueue_copy(queue, x, x_buf).wait()
    a_values_buf.release()
    a_cols_buf.release()
    a_pointers_buf.release()
    x_buf.release()
    r_buf.release()
    q_buf.release()
    d_buf.release()
    dot_res_buf.release()
    const_buf.release()
    queue.flush()
    # queue.finish()
    return x
