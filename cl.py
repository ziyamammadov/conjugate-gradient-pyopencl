
import numpy as np
import pyopencl as cl
import os
IS_COMPLEX = True
WAVE_SIZE = 32
LOCAL_SIZE = 8*WAVE_SIZE
FOLDER_PATH = './kernel/complex/' if IS_COMPLEX else './kernel/real/'
INCLUDE_FILE = "-I ./kernel/complex " if IS_COMPLEX else ""

def load_and_build_kernel(ctx, kernel_name, options):    
    with open(f'{FOLDER_PATH}{kernel_name}.cl', 'r') as f:
        kernel = cl.Program(ctx, f.read()).build(options=options).__getattr__(kernel_name)
    return kernel

def initialize_cl_environment():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    return ctx, queue

def initialize_cl_environment_with_device(device):
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)
    return ctx, queue

def get_gpu_devices():
    platforms = cl.get_platforms()
    devices = []
    for platform in platforms:
        devices += platform.get_devices(device_type=cl.device_type.GPU)
    return devices

def load_and_build_kernels(ctx, n_rhs):
    options = [f"{INCLUDE_FILE} -D N_RHS={n_rhs} -D WAVE_SIZE={WAVE_SIZE} -D WG_SIZE={LOCAL_SIZE}"]

    return {
        'axpy': load_and_build_kernel(ctx, 'axpy', options),
        'aypx': load_and_build_kernel(ctx, 'aypx', options),
        'spmv': load_and_build_kernel(ctx, 'spmv', options),
        'sub': load_and_build_kernel(ctx, 'sub', options),
        'vdot': load_and_build_kernel(ctx, 'vdot', options)
    }

def CG(ctx, queue, kernels, size, non_zeros, a_values, b_values, a_pointers, a_cols, x, n_rhs, n_iterations, device=None):
    if n_rhs == 1:
        kernels = load_and_build_kernels(ctx, 1)
    
    axpy_kernel = kernels['axpy']
    aypx_kernel = kernels['aypx']
    spmv_kernel = kernels['spmv']
    sub_kernel = kernels['sub']
    dot_kernel = kernels['vdot']
        
    work_groups = 1 + ((size - 1) // LOCAL_SIZE)
    global_size = size if work_groups == 1 else work_groups * LOCAL_SIZE
    local_size = global_size if work_groups == 1 else LOCAL_SIZE
    rows_per_wg = (LOCAL_SIZE // WAVE_SIZE)
    spmv_work_groups = 1 + ((size - 1) // rows_per_wg)
    spmv_global_size = spmv_work_groups * LOCAL_SIZE
    spmv_local_size = LOCAL_SIZE
    np_type = np.dtype(np.csingle if IS_COMPLEX else np.intc)
    val_size = np_type.itemsize
    
    # print(f'Local size: {local_size}')
    # print(f'Global size: {global_size}')
    # print(f'Size of matrix: {size}')
    # print(f'SPMV Local size: {spmv_local_size}')
    # print(f'SPMV Global size: {spmv_global_size}')
    
    # Allocate device memory and copy host arrays to device
    mf = cl.mem_flags
    int_size = np.dtype(np.intc).itemsize
    a_values_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=non_zeros * val_size, hostbuf=a_values)
    a_cols_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=non_zeros * int_size, hostbuf=a_cols)
    a_pointers_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=(size + 1) * int_size, hostbuf=a_pointers)
    b_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                      size=n_rhs * size * val_size,
                      hostbuf=b_values)
    x_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                      size=n_rhs * size * val_size,
                      hostbuf=x)
    r_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * val_size)
    d_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * val_size)
    q_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * val_size)
    dot_res_buf = cl.Buffer(
        ctx, mf.READ_WRITE, size=n_rhs * work_groups * val_size)
    const_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * val_size)
    
    # CL local memories
    spmv_cl_loc_mem = cl.LocalMemory(n_rhs * spmv_local_size * val_size)
    dot_cl_loc_mem = cl.LocalMemory(n_rhs * local_size * val_size)

    # y = A * x                   (spmv)
    wait_spmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                            a_pointers_buf, a_cols_buf, x_buf, q_buf, spmv_cl_loc_mem)
    # r = b - y                   (sub)
    wait_sub = sub_kernel(queue, (global_size,), (local_size,),
                          b_buf, q_buf, r_buf, np.intc(size), wait_for=[wait_spmv])
    b_buf.release()
    # d = r                       (copy)
    wait_copy = cl.enqueue_copy(queue, d_buf, r_buf, wait_for=[wait_sub])

    # deltaNew = r^T * r               (dot)
    wait_dot = dot_kernel(queue, (global_size,), (local_size,), r_buf, r_buf, dot_cl_loc_mem,
                          dot_res_buf, np.intc(size), wait_for=[wait_copy])

    h_dot_res = np.zeros(n_rhs * work_groups, dtype=np_type)
    delta_new = np.zeros(n_rhs, dtype=np_type)
    delta_old = np.zeros(n_rhs, dtype=np_type)
    alpha = np.zeros(n_rhs, dtype=np_type)
    cl.enqueue_copy(queue, h_dot_res, dot_res_buf, wait_for=[wait_dot])

    # Reduction
    for r in range(n_rhs):
        for i in range(work_groups):
            delta_new[r] += h_dot_res[r * work_groups + i]
    delta_old = delta_new
    
    # print(f'Delta new: {delta_new}')

    for iteration in range(n_iterations):
        # q = A * d (spmv)
        if iteration == 0:
            wait_spmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                                    a_pointers_buf, a_cols_buf, d_buf, q_buf, spmv_cl_loc_mem)
        else:
            wait_spmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                                    a_pointers_buf, a_cols_buf, d_buf, q_buf, spmv_cl_loc_mem, wait_for=[waitKAypx])

        # dq = d * q(dot)
        wait_dot = dot_kernel(queue, (global_size,), (local_size,), d_buf, q_buf, dot_cl_loc_mem,
                              dot_res_buf, np.intc(size), wait_for=[wait_spmv])
            
        h_dot_res = np.zeros(n_rhs * work_groups, dtype=np_type)
        cl.enqueue_copy(queue, h_dot_res, dot_res_buf,
                        wait_for=[wait_dot])
        h_dq = np.zeros(n_rhs, dtype=np_type)
        for r in range(n_rhs):
            # Reduction
            for i in range(work_groups):
                h_dq[r] += h_dot_res[r * work_groups + i]
        alpha = delta_new/h_dq

        # print(f'Alpha: {alpha}')

        cl.enqueue_copy(queue, const_buf, np.csingle(
            alpha))

        # x = alpha * d + x(axpy)
        axpy_kernel(queue, (global_size,), (local_size,), d_buf,
                    x_buf, const_buf, np.intc(1), np.intc(size))
        # print(f'X: {x}')

        # r = - alpha * q + r(axpy)
        wait_axpy = axpy_kernel(queue, (global_size,), (local_size,), q_buf,
                                r_buf, const_buf, np.intc(0), np.intc(size))
        delta_old = delta_new
        # print(f'R: {x}')

        # deltaNew = r ^ T * r(dot)
        wait_dot = dot_kernel(queue, (global_size,), (local_size,), r_buf, r_buf, dot_cl_loc_mem,
                              dot_res_buf, np.intc(size), wait_for=[wait_axpy])
        cl.enqueue_copy(queue, h_dot_res, dot_res_buf, wait_for=[wait_dot])

        # Reduction
        delta_new = np.zeros(n_rhs, dtype=np_type)
        for r in range(n_rhs):
            for i in range(work_groups):
                delta_new[r] += h_dot_res[r * work_groups + i]

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

    cl.enqueue_copy(queue, x, x_buf)
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
    queue.finish()
    return x


def conjugate_gradient_multi_gpu(ctx, queue, kernels, size, non_zeros, a_values, b_values, a_pointers, a_cols, x, n_rhs, n_iterations, device):
        
    axpy_kernel = kernels['axpy']
    aypx_kernel = kernels['aypx']
    spmv_kernel = kernels['spmv']
    sub_kernel = kernels['sub']
    dot_kernel = kernels['vdot']
        
    work_groups = 1 + ((size - 1) // LOCAL_SIZE)
    global_size = size if work_groups == 1 else work_groups * LOCAL_SIZE
    local_size = global_size if work_groups == 1 else LOCAL_SIZE
    rows_per_wg = (LOCAL_SIZE // WAVE_SIZE)
    spmv_work_groups = 1 + ((size - 1) // rows_per_wg)
    spmv_global_size = spmv_work_groups * LOCAL_SIZE
    spmv_local_size = LOCAL_SIZE
    np_type = np.dtype(np.csingle if IS_COMPLEX else np.intc)
    val_size = np_type.itemsize
    
    # print(f'Local size: {local_size}')
    # print(f'Global size: {global_size}')
    # print(f'Size of matrix: {size}')
    # print(f'SPMV Local size: {spmv_local_size}')
    # print(f'SPMV Global size: {spmv_global_size}')
    
    # Allocate device memory and copy host arrays to device
    mf = cl.mem_flags
    int_size = np.dtype(np.intc).itemsize
    a_values_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=non_zeros * val_size, hostbuf=a_values)
    a_cols_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=non_zeros * int_size, hostbuf=a_cols)
    a_pointers_buf = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=(size + 1) * int_size, hostbuf=a_pointers)
    b_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                      size=n_rhs * size * val_size,
                      hostbuf=b_values)
    x_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                      size=n_rhs * size * val_size,
                      hostbuf=x)
    r_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * val_size)
    d_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * val_size)
    q_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * size * val_size)
    dot_res_buf = cl.Buffer(
        ctx, mf.READ_WRITE, size=n_rhs * work_groups * val_size)
    const_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * val_size)
    
    # CL local memories
    spmv_cl_loc_mem = cl.LocalMemory(n_rhs * spmv_local_size * val_size)
    dot_cl_loc_mem = cl.LocalMemory(n_rhs * local_size * val_size)

    # y = A * x                   (spmv)
    wait_spmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                            a_pointers_buf, a_cols_buf, x_buf, q_buf, spmv_cl_loc_mem)
    # r = b - y                   (sub)
    wait_sub = sub_kernel(queue, (global_size,), (local_size,),
                          b_buf, q_buf, r_buf, np.intc(size), wait_for=[wait_spmv])
    b_buf.release()
    # d = r                       (copy)
    wait_copy = cl.enqueue_copy(queue, d_buf, r_buf, wait_for=[wait_sub])

    # deltaNew = r^T * r               (dot)
    wait_dot = dot_kernel(queue, (global_size,), (local_size,), r_buf, r_buf, dot_cl_loc_mem,
                          dot_res_buf, np.intc(size), wait_for=[wait_copy])

    h_dot_res = np.zeros(n_rhs * work_groups, dtype=np_type)
    delta_new = np.zeros(n_rhs, dtype=np_type)
    delta_old = np.zeros(n_rhs, dtype=np_type)
    alpha = np.zeros(n_rhs, dtype=np_type)
    cl.enqueue_copy(queue, h_dot_res, dot_res_buf, wait_for=[wait_dot])

    # Reduction
    for r in range(n_rhs):
        for i in range(work_groups):
            delta_new[r] += h_dot_res[r * work_groups + i]
    delta_old = delta_new
    
    # print(f'Delta new: {delta_new}')

    for iteration in range(n_iterations):
        # q = A * d (spmv)
        if iteration == 0:
            wait_spmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                                    a_pointers_buf, a_cols_buf, d_buf, q_buf, spmv_cl_loc_mem)
        else:
            wait_spmv = spmv_kernel(queue, (spmv_global_size,), (spmv_local_size,), np.intc(size), a_values_buf,
                                    a_pointers_buf, a_cols_buf, d_buf, q_buf, spmv_cl_loc_mem, wait_for=[waitKAypx])

        # dq = d * q(dot)
        wait_dot = dot_kernel(queue, (global_size,), (local_size,), d_buf, q_buf, dot_cl_loc_mem,
                              dot_res_buf, np.intc(size), wait_for=[wait_spmv])
            
        h_dot_res = np.zeros(n_rhs * work_groups, dtype=np_type)
        cl.enqueue_copy(queue, h_dot_res, dot_res_buf,
                        wait_for=[wait_dot])
        h_dq = np.zeros(n_rhs, dtype=np_type)
        for r in range(n_rhs):
            # Reduction
            for i in range(work_groups):
                h_dq[r] += h_dot_res[r * work_groups + i]
        alpha = delta_new/h_dq

        # print(f'Alpha: {alpha}')

        cl.enqueue_copy(queue, const_buf, np.csingle(
            alpha))

        # x = alpha * d + x(axpy)
        axpy_kernel(queue, (global_size,), (local_size,), d_buf,
                    x_buf, const_buf, np.intc(1), np.intc(size))
        # print(f'X: {x}')

        # r = - alpha * q + r(axpy)
        wait_axpy = axpy_kernel(queue, (global_size,), (local_size,), q_buf,
                                r_buf, const_buf, np.intc(0), np.intc(size))
        delta_old = delta_new
        # print(f'R: {x}')

        # deltaNew = r ^ T * r(dot)
        wait_dot = dot_kernel(queue, (global_size,), (local_size,), r_buf, r_buf, dot_cl_loc_mem,
                              dot_res_buf, np.intc(size), wait_for=[wait_axpy])
        cl.enqueue_copy(queue, h_dot_res, dot_res_buf, wait_for=[wait_dot])

        # Reduction
        delta_new = np.zeros(n_rhs, dtype=np_type)
        for r in range(n_rhs):
            for i in range(work_groups):
                delta_new[r] += h_dot_res[r * work_groups + i]

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

    cl.enqueue_copy(queue, x, x_buf)
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
    queue.finish()
    # result_queue.put((gpu_id, x))
    # print(f'X is equal: {x}')
    
    return x
