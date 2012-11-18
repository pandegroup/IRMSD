import distutils.ccompiler
from distutils.core import setup
from distutils.extension import Extension
import distutils.sysconfig
import numpy as np

# If your machine supports only SSE2 but not SSSE3, change the following
# definition to True.
SSE2_ONLY = False


def detect_openmp():
    compiler = distutils.ccompiler.new_compiler()
    print "Attempting to autodetect OpenMP; ignore anything between the lines."
    print "-------------------------------------------------------------------"
    hasopenmp = compiler.has_function('omp_get_num_threads')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = compiler.has_function('omp_get_num_threads')
        needs_gomp = hasopenmp
    print "-------------------------------------------------------------------"
    print
    if hasopenmp:
        print "Detected compiler OpenMP support"
    else:
        print "Did not detect OpenMP support"
    return hasopenmp, needs_gomp


def main():
    py_include = distutils.sysconfig.get_python_inc()
    np_include = np.get_include()
    openmp_enabled, needs_gomp = detect_openmp()

    compiler_args = ['-msse2' if SSE2_ONLY else '-mssse3',
                     '--std=gnu99', '-O3', '-funroll-loops']
    if openmp_enabled:
        compiler_args.append('-fopenmp')
    compiler_libraries = ['gomp'] if needs_gomp else []

    rmsd_ext = Extension('IRMSD.rmsdcalc',
                         ['IRMSD/rmsdcalc.c', 'IRMSD/theobald_rmsd.c'],
                         extra_compile_args=compiler_args,
                         libraries=compiler_libraries,
                         include_dirs=[np_include, py_include])

    setup(name='IRMSD',
          version='1.0',
          description='Python implementation of IRMSD fast RMSD',
          long_description='Python implementation of IRMSD fast RMSD',
          author='Imran S. Haque',
          author_email='ihaque@cs.stanford.edu',
          url='https://github.com/simtk/IRMSD',
          packages=['IRMSD'],
          ext_modules=[rmsd_ext],
          license='BSD',
          platforms=['Any'],
          classifiers=['Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Software Development :: Libraries :: Python Modules'])

if __name__ == "__main__":
    main()
