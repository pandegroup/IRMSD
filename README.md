IRMSD: Fast structural RMSD computation
=====

IRMSD is a Python library for computing the optimal root-mean-square-deviation
between pairs of structures (e.g., protein conformations). It is based on the
Theobald QCP method, and because of an efficient matrix multiply routine, is
several-fold faster than other RMSD packages using off-the-shelf naive or
generic high-performance matrix multiplies. In particular:

- With one thread, IRMSD is 4x the speed of the original Theobald code and
  over 3x as fast as Theobald code modified to use GotoBLAS.
- IRMSD features automatic parallelization over multiple RMSD computations;
  with four threads, it is twice the speed of similarly parallelized
  Theobald code using GotoBLAS.
- IRMSD reaches the machine theoretical peak arithmetic performance when
  using multiple threads and small structures that fit into L1 cache.
- IRMSD reaches the machine limit of memory bandwidth when dealing with
  very large structures that do not fit into cache.

IRMSD also fixes a small numerical instability in the Theobald QCP method that
manifests as a lack of convergence in approximately one in 10^9 RMSD
computations.

**If you use this code in computations that result in publication, please cite our
paper: [upcoming].**

# Installation

IRMSD is distributed as a Python package with a C extension module. Installing
it requires a C compiler with C99 support. For parallel RMSD computation, a
compiler supporting OpenMP is required. The setup script attempts to
automatically detect OpenMP support and all required header directories. To
install IRMSD, just run:

`python setup.py install`

from the `python/` directory of the library.

After installation, you may run `python/IRMSD/test.py` to run the regression
test suite.

Note: The version of clang in current versions of OS X does not support OpenMP,
so parallel execution will not work on that platform.

# Non-Python languages

No wrappers for non-Python languages are currently distributed. However, the
computational core of the module is isolated into `theobald_rmsd.c`, so the
library should be usable from C/C++ code, and in principle native wrappers
could be written for other languages. `rmsdcalc.c` wraps `theobald_rmsd.c` for
Python; `python/IRMSD/__init__.py` provides a convenient object-oriented wrapper
around the low-level routines in `rmsdcalc`.

# Usage

The recommended usage of IRMSD routines is through the `IRMSD.Conformations`
class.

## Input Data Format

IRMSD supports structural data given in either of two typical orderings,
which we denote as "atom" and "axis" major order. If possible, use axis-major
ordering, as it performs better for typical structure sizes. However, if your
program already has a large amount of atom-major data in memory, IRMSD can
compute on it in that layout.

Assume that our structure has two atoms:

```
    atom1.{x, y, z} = {1, 2, 3}
    atom2.{x, y, z} = {10, 11, 12}
```

Axis-major ordering has all the x-coordinates, followed by all the
y-coordinates, followed by all the z-coordinates; in our example, the linear
axis-major ordering would be `1 10 2 11 3 12`.

Atom-major ordering has each atom's coordinates in atom order. In the example,
linear atom-major ordering would be `1 2 3 10 11 12`.

IRMSD requires that the data input be a three-dimensional NumPy array of type
float32 (single-precision floating point), in either atom- or axis-major order.
The first dimension always corresponds to different structures. The second
coordinate is the atom number for atom-major ordering and the axis number (0 for
x, 1 for y, 2 for z) for axis-major. The last coordinate is the axis number for
atom-major order or the atom number for axis-major.

IRMSD further mandates that each structure be padded out to have a "padded"
number of atoms that is a multiple of four, and that all such padding elements
be equal to zero. Finally, the array must be aligned to a multiple of 16 bytes
in memory.

The final representation for our example structure, in atom-major order, might
look like:

```python
[[[1 2 3]
  [10 11 12]
  [0 0 0]
  [0 0 0]]]
```

Or, in axis-major order:
```python
[[[1 10 0 0]
  [2 11 0 0]
  [3 12 0 0]]]
```

Note that the array can be interpreted as a vertically-stacked (`np.vstack`) set
of 2-dimensional arrays of structure coordinates.

If your input data do not intrinsically conform to these specifications, the
`IRMSD.align_array` function takes an input 3-d array of coordinates, along with
a specified majority, and returns a copy that satifies the alignment and padding
requirements for IRMSD. If you would prefer to avoid copies by filling in a
properly allocated array in your input routines, `IRMSD._allocate_aligned_array`
takes a desired shape and majority, and returns an array meeting the alignment
and padding specifications; padding elements are initialized to zero, but other
elements are left uninitialized.

## RMSD Computation

Given a properly-formatted input array, first construct an `IRMSD.Conformations`
object.

```python
from IRMSD import align_array
from IRMSD import Conformations
calculator = Conformations(align_array(my_conformations), my_majority)
```

(Note that the example code performs a data copy, in `align_array`. This may be
undesirable, for very large data sets; in this case, use
`IRMSD._allocate_aligned_array` and store data into the returned array in your
input routines.)

Given two `Conformations` objects `C1` and `C2`, you can compute the RMSDs
between all conformations in `C1` and a single chosen conformation in `C2` in
one function call:

```python
# Compute RMSD between every conf in C1 and the first conformation in C2
rmsds = C1.rmsds_to_reference(C2, 0)

# Compute RMSD between every conf in C1 and the tenth conformation in C2
rmsds = C1.rmsds_to_reference(C2, 9)

# Compute RMSD between every conf in C1 and the first conformation in C1
rmsds = C1.rmsds_to_reference(C1, 0)
```

Note that the special case of computing all RMSDs against one in the same
conformation object is supported. Underneath the hood, `rmsds_to_reference`
will automatically center the structures if needed and runs a modified
Theobald QCP method to compute RMSDs in single precision. If OpenMP support
was available at installation time, the computation is parallelized over all
conformations in `C1` over all available cores; the number of threads can be
adjusted with the `OMP_NUM_THREADS` environment variable.

RMSDs are returned as a one-dimensional, single-precision NumPy array.


**NB: the array passed into the `Conformations` constructor may be modified by
the routines in IRMSD!**


# Licensing

IRMSD is released under a modified BSD license, as follows:

```
COPYRIGHT NOTICE

Written by Imran S. Haque

Copyright (c) 2011 Stanford University.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of Stanford University nor the names of its contributors
      may be used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
