# The MIT License (MIT)
#
# Modifications Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

import numpy as np
import cupy as cp

def ndarray_aligned(shape, dtype, align=16):

    # get total size
    nelem = np.prod(shape)
    elemsize = dtype(1).itemsize
    size = nelem * elemsize
    
    # extend to fit alignment
    alloc_size = (size + align - 1)
    
    # allocate memory
    buf = cp.cuda.Memory(alloc_size)
    # get base pointer
    ptr = buf.ptr
    # compute offset
    off = 0 if (ptr%align==0) else (align - ptr % align)
    aligned_ptr = cp.cuda.MemoryPointer(buf, off)
    
    return cp.ndarray(shape, dtype=dtype, memptr=aligned_ptr)
