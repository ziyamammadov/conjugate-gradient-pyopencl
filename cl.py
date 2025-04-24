import numpy as np
import pyopencl as cl

IS_COMPLEX = True
WAVE_SIZE = 32
LOCAL_SIZE = 8 * WAVE_SIZE
FOLDER_PATH = './kernel/complex/' if IS_COMPLEX else './kernel/real/'
INCLUDE_FILE = "-I ./kernel/complex " if IS_COMPLEX else ""
DEBUG_MODE = True

def debug_print(*args):
    if DEBUG_MODE:
        print(*args)

def load_and_build_kernel(ctx, kernel_name, devices, options):
    with open(f'{FOLDER_PATH}{kernel_name}.cl', 'r') as f:
        kernel = cl.Program(ctx, f.read()).build(options=options, devices=devices).__getattr__(kernel_name)
    return kernel

def initialize_cl_ctx(devices):
    return cl.Context(devices=devices)

def initialize_cl_queue_with_device(ctx, device):
    return cl.CommandQueue(ctx, device)

def get_gpu_devices():
    platforms = cl.get_platforms()
    devices = []
    for platform in platforms:
        devices += platform.get_devices(device_type=cl.device_type.GPU)
    return devices

def load_and_build_kernels(ctx, devices):
    options = [f"{INCLUDE_FILE} -D WAVE_SIZE={WAVE_SIZE} -D WG_SIZE={LOCAL_SIZE}"]
    return {
        'tol': load_and_build_kernel(ctx, 'tol', devices, options),
        'axpy': load_and_build_kernel(ctx, 'axpy', devices, options),
        'aypx': load_and_build_kernel(ctx, 'aypx', devices, options),
        'spmv': load_and_build_kernel(ctx, 'spmv', devices, options),
        'sub': load_and_build_kernel(ctx, 'sub', devices, options),
        'vdot': load_and_build_kernel(ctx, 'vdot', devices, options),
    }

def conjugate_gradient_multi_gpu(ctx, queue, kernels, size, non_zeros, a_values, b_values, a_pointers, a_cols, x, n_rhs, n_iterations, tol=1e-6, check_tolerance=False):
    axpy_kernel = kernels['axpy']
    aypx_kernel = kernels['aypx']
    spmv_kernel = kernels['spmv']
    sub_kernel = kernels['sub']
    dot_kernel = kernels['vdot']
    tol_kernel = kernels['tol']

    work_groups = 1 + ((size - 1) // LOCAL_SIZE)
    global_size = work_groups * LOCAL_SIZE
    rows_per_wg = (LOCAL_SIZE // WAVE_SIZE)
    spmv_work_groups = 1 + ((size - 1) // rows_per_wg)
    spmv_global_size = spmv_work_groups * LOCAL_SIZE

    np_type = np.dtype(np.csingle if IS_COMPLEX else np.float32)
    val_size = np_type.itemsize
    np_size = np.int32(size)
    np_rhs = np.int32(n_rhs)
    mf = cl.mem_flags
    
    a_values_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_values)
    a_cols_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_cols)
    a_pointers_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_pointers)

    b_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b_values)
    x_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    r_buf = cl.Buffer(ctx, mf.READ_WRITE, size=x.nbytes)
    d_buf = cl.Buffer(ctx, mf.READ_WRITE, size=x.nbytes)
    q_buf = cl.Buffer(ctx, mf.READ_WRITE, size=x.nbytes)
    
    dot_res_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * work_groups * val_size)
    const_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs * val_size)
    
    tolerance_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, size=np.dtype(np.float32).itemsize, hostbuf=np.array([tol**2], dtype=np.float32))
    converged_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_rhs *np.dtype(np.int32).itemsize)
    converged = np.zeros(n_rhs, dtype=np.int32)
    
    spmv_loc_mem = cl.LocalMemory(n_rhs * LOCAL_SIZE * val_size)
    dot_loc_mem = cl.LocalMemory(n_rhs * LOCAL_SIZE * val_size)

    # y = A * x                   (spmv)
    spmv_kernel(queue, (spmv_global_size,), (LOCAL_SIZE,), np_size, a_values_buf, a_pointers_buf, a_cols_buf, x_buf, q_buf, spmv_loc_mem, np_rhs)
    # r = b - y                   (sub)
    sub_kernel(queue, (global_size,), (LOCAL_SIZE,), b_buf, q_buf, r_buf, np_size, np_rhs)
    # d = r                       (copy)
    cl.enqueue_copy(queue, d_buf, r_buf)
    # delta_new = r^T * r         (dot)
    dot_kernel(queue, (global_size,), (LOCAL_SIZE,), r_buf, r_buf, dot_loc_mem, dot_res_buf, np_size, np_rhs)
    h_dot_res = np.empty(n_rhs * work_groups, dtype=np_type)
    cl.enqueue_copy(queue, h_dot_res, dot_res_buf)
    delta_new = np.array([np.sum(h_dot_res[r * work_groups:(r + 1) * work_groups]) for r in range(n_rhs)], dtype=np_type)
    # delta_old = delta_new      (copy)
    delta_old = np.copy(delta_new)

    for iteration in range(n_iterations):
        # check tolerance
        if check_tolerance and iteration % 10 == 0:
            wait_tol = tol_kernel(queue, (n_rhs,), None, dot_res_buf, tolerance_buf, converged_buf, np_rhs, np.int32(work_groups))

            cl.enqueue_copy(queue, converged, converged_buf, wait_for=[wait_tol], is_blocking=False)
            
            if np.all(converged):
                debug_print(f'Converged after {iteration} iterations')
                break
        
        # q = A * d              (spmv)
        spmv_kernel(queue, (spmv_global_size,), (LOCAL_SIZE,), np_size, a_values_buf, a_pointers_buf, a_cols_buf, d_buf, q_buf, spmv_loc_mem, np_rhs)
        # dq = d * q             (dot)
        dot_kernel(queue, (global_size,), (LOCAL_SIZE,), d_buf, q_buf, dot_loc_mem, dot_res_buf, np_size, np_rhs)
        cl.enqueue_copy(queue, h_dot_res, dot_res_buf)
        
        dq = np.array([np.sum(h_dot_res[r * work_groups:(r + 1) * work_groups]) for r in range(n_rhs)], dtype=np_type)
        
        alpha = delta_new / dq
        cl.enqueue_copy(queue, const_buf, alpha)

        # x = x + alpha * d       (axpy)
        axpy_kernel(queue, (global_size,), (LOCAL_SIZE,), d_buf, x_buf, const_buf, np.int32(1), np_size, np_rhs)
        # r = r - alpha * q       (axpy)
        axpy_kernel(queue, (global_size,), (LOCAL_SIZE,), q_buf, r_buf, const_buf, np.int32(0), np_size, np_rhs)

        delta_old[:] = delta_new[:]
        
        # delta_new = r^T * r    (dot)
        dot_kernel(queue, (global_size,), (LOCAL_SIZE,), r_buf, r_buf, dot_loc_mem, dot_res_buf, np_size, np_rhs)
        cl.enqueue_copy(queue, h_dot_res, dot_res_buf)
        delta_new = np.array([np.sum(h_dot_res[r * work_groups:(r + 1) * work_groups]) for r in range(n_rhs)], dtype=np_type)

        beta = delta_new / delta_old
        cl.enqueue_copy(queue, const_buf, beta)
        
        # d = r + beta * d       (aypx)
        aypx_kernel(queue, (global_size,), (LOCAL_SIZE,), r_buf, d_buf, const_buf, np_size, np_rhs)

    cl.enqueue_copy(queue, x, x_buf)
    queue.finish()
    return x