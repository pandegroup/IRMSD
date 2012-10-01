// Copyright 2011 Stanford University
//
// MSMBuilder is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//


#include "Python.h"
#include "arrayobject.h"
#include <stdint.h>
#include <stdio.h>
#include "theobald_rmsd.h"
#include <omp.h>


#define CHECKARRAYTYPE(ary,name) if (PyArray_TYPE(ary) != NPY_FLOAT32) {\
                                     PyErr_SetString(PyExc_ValueError,name" was not of type float32");\
                                     return NULL;\
                                 } 
#define CHECKARRAYCARRAY(ary,name) if ((PyArray_FLAGS(ary) & NPY_CARRAY) != NPY_CARRAY) {\
                                       PyErr_SetString(PyExc_ValueError,name" was not a contiguous well-behaved array in C order");\
                                       return NULL;\
                                   } 


static PyObject *_getMultipleRMSDs_axis_major(PyObject *self, PyObject *args) {
    float *AData, *BData, *GAData, *distances;
    int nrealatoms=-1, npaddedatoms=-1, rowstride=-1, truestride=-1;
    npy_intp dim2[2], *arrayADims;
    PyArrayObject *ary_coorda, *ary_coordb, *ary_Ga, *ary_distances;
 
    if (!PyArg_ParseTuple(args, "iiiOOOf",&nrealatoms,&npaddedatoms,&rowstride,
                          &ary_coorda, &ary_coordb, &ary_Ga, &G_y)) {
      return NULL;
    }

  
    // Get pointers to array data
    AData  = (float*) PyArray_DATA(ary_coorda);
    BData  = (float*) PyArray_DATA(ary_coordb);
    GAData  = (float*) PyArray_DATA(ary_Ga);

    // TODO add sanity checking on Ga
    // TODO add sanity checking on structure dimensions A vs B

    arrayADims = PyArray_DIMS(ary_coorda);

    // Do some sanity checking on array dimensions
    //      - make sure they are of float32 data type
    CHECKARRAYTYPE(ary_coorda,"Array A");
    CHECKARRAYTYPE(ary_coordb,"Array B");

    if (ary_coorda->nd != 3) {
        PyErr_SetString(PyExc_ValueError,"Array A did not have dimension 3");
        return NULL;
    }
    if (ary_coordb->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Array B did not have dimension 2");
        return NULL;
    }
    // make sure stride is 4 in last dimension (ie, is C-style and contiguous)
    CHECKARRAYCARRAY(ary_coorda,"Array A");
    CHECKARRAYCARRAY(ary_coordb,"Array B");

    // Create return array containing RMSDs
    dim2[0] = arrayADims[0];
    dim2[1] = 1;
    ary_distances = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_FLOAT);
    distances = (float*) PyArray_DATA(ary_distances);

    truestride = npaddedatoms * 3;

    #pragma omp parallel for
    for (int i = 0; i < arrayADims[0]; i++) {
        float msd = msd_axis_major(nrealatoms, npaddedatoms, rowstride,
                                   (AData + i*truestride), BData, GAData[i], G_y);
        distances[i] = sqrtf(msd);
    }
    
    return PyArray_Return(ary_distances);
}

static PyObject *_getMultipleRMSDs_atom_major(PyObject *self, PyObject *args) {
    float *AData, *BData, *GAData, *distances;
    int nrealatoms=-1, npaddedatoms=-1;
    npy_intp dim2[2], *arrayADims;
    PyArrayObject *ary_coorda, *ary_coordb, *ary_Ga, *ary_distances;
 
    if (!PyArg_ParseTuple(args, "iiOOOf",&nrealatoms,&npaddedatoms,
                          &ary_coorda, &ary_coordb, &ary_Ga, &G_y)) {
      return NULL;
    }

  
    // Get pointers to array data
    AData  = (float*) PyArray_DATA(ary_coorda);
    BData  = (float*) PyArray_DATA(ary_coordb);
    GAData  = (float*) PyArray_DATA(ary_Ga);

    // TODO add sanity checking on Ga
    // TODO add sanity checking on structure dimensions A vs B

    arrayADims = PyArray_DIMS(ary_coorda);

    // Do some sanity checking on array dimensions
    //      - make sure they are of float32 data type
    CHECKARRAYTYPE(ary_coorda,"Array A");
    CHECKARRAYTYPE(ary_coordb,"Array B");

    if (ary_coorda->nd != 3) {
        PyErr_SetString(PyExc_ValueError,"Array A did not have dimension 3");
        return NULL;
    }
    if (ary_coordb->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Array B did not have dimension 2");
        return NULL;
    }
    // make sure stride is 4 in last dimension (ie, is C-style and contiguous)
    CHECKARRAYCARRAY(ary_coorda,"Array A");
    CHECKARRAYCARRAY(ary_coordb,"Array B");

    // Create return array containing RMSDs
    dim2[0] = arrayADims[0];
    dim2[1] = 1;
    ary_distances = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_FLOAT);
    distances = (float*) PyArray_DATA(ary_distances);

    #pragma omp parallel for
    for (int i = 0; i < arrayADims[0]; i++) {
        float msd = msd_atom_major(nrealatoms, npaddedatoms,
                                   (AData + i*npaddedatoms*3), BData, GAData[i], G_y);
        distances[i] = sqrtf(msd);
    }
    
    return PyArray_Return(ary_distances);
}

static PyMethodDef _rmsd_methods[] = {
  {"getMultipleRMSDs_axis_major", (PyCFunction)_getMultipleRMSDs_axis_major, METH_VARARGS, "Theobald RMSD calculation on axis-major centered structures."},
  {"getMultipleRMSDs_atom_major", (PyCFunction)_getMultipleRMSDs_atom_major, METH_VARARGS, "Theobald RMSD calculation on atom-major centered structures."},
  {NULL, NULL, 0, NULL}
};

DL_EXPORT(void) initrmsdcalc(void)
{
  Py_InitModule3("rmsdcalc", _rmsd_methods, "Core routines for IRMSD fast Theobald RMSD calculation.");
  import_array();
}
