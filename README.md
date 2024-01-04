This library is a general purpose library for linear algebra.

**All Features are still Under Construction**

# Features
### Linear algebric vectors
* `SizedVec`: representing a vector whose size is determined at the compile time. (the data will be allocated at the stack area and is expected to apply to vectors relatively small size)
* `DynVec`: representing a vector whose size can be determined at the execution time. (the data will be allocated to the heap memory)

### Vector-Vector operations
* `add`: element to element in-place addition
* `+`: element to element addition
* `sub`: element to element in-plane subtraction
* `-`: element to element subtraction
* `axpy`: performs vector-scalar product and adds the result into the given vector
* `scale`: performs vector-scalar product (in-place)
* `scaled`: performs vector-scalar product
* `*`: the same operation as `scaled`
* `dot`: performs the vector-vector inner products