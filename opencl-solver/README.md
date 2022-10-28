# cg-opencl
### Conjugate Gradient solver in OpenCL

- [X] Real numbered CG
  - [X] Host code
  - [X] KERNEL axpy
  - [X] KERNEL aypx
  - [X] KERNEL coomv (_updated with `spmv.cl`_)
  - [X] KERNEL csrmv or other type of substitute that doesnt include atomics (`spmv.cl`)
  - [X] KERNEL sub
  - [X] KERNEL dot
- [X] Complex numbered CG
  - [X] Host code
  - [X] Kernels
- [X] Multiple RHS on a single run (`SpMV` -> `SpMM` with a tall matrix)
- [ ] Considerations for different platforms (automatic parameter choice)
- [ ] Proper C library structure

#### Instructions
* invoke `clcg::cg` with the following parameters
  * `size` - N of input matrix
  * `nonZeros` - Number of non zeros in the matrix. 
  * `aValues`, `aPointers`, `aCols` - The matrix in CSR format (check report for correct array structures)
  * `b` - the right-hand side vector (of size `size`)
  * `x` - pointer to the output solution vector, can be initialized with a close
  
  An example of usage is in `main.c`.
  The example makes use of the ['BeBOP Sparse Matrix Converter'](http://bebop.cs.berkeley.edu/smc/) library for IO and expanding sparse notations, reading in a matrix in ['Matrix Market'](https://math.nist.gov/MatrixMarket/formats.html) format.
  
#### Compilation
To use BeBOP SMC, follow the compilation instructions at the link above. Place the three directories into a directory `bebop` of the root of the repository.

To compile this project: `cmake` followed by `make`
