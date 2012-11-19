from distutils.ccompiler import new_compiler
from distutils.core import setup
from distutils.extension import Extension
import distutils.sysconfig
import os
import sys
import tempfile
import shutil

import numpy as np

# If your machine supports only SSE2 but not SSSE3, change the following
# definition to True.
SSE2_ONLY = False


# From http://stackoverflow.com/questions/
#            7018879/disabling-output-when-compiling-with-distutils
def hasfunction(cc, funcname):
    tmpdir = tempfile.mkdtemp(prefix='irmsd-install-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            f.write('int main(void) {\n')
            f.write('    %s();\n' % funcname)
            f.write('}\n')
            f.close()
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            # This will have to be changed if we ever have to check
            # for a function on Windows.
            devnull = open('/dev/null', 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir)
            cc.link_executable(objects, os.path.join(tmpdir, "a.out"))
        except:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)


def detect_openmp():
    compiler = new_compiler()
    print "Attempting to autodetect OpenMP support...",
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = hasfunction(compiler, 'omp_get_num_threads')
        needs_gomp = hasopenmp
    print
    if hasopenmp:
        print "Compiler supports OpenMP"
    else:
        print "Did not detect OpenMP support; parallel RMSD disabled"
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
    compiler_defs = [('USE_OPENMP', None)] if openmp_enabled else []

    rmsd_ext = Extension('IRMSD.rmsdcalc',
                         ['IRMSD/rmsdcalc.c', 'IRMSD/theobald_rmsd.c'],
                         extra_compile_args=compiler_args,
                         define_macros=compiler_defs,
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
