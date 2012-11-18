import numpy as np
import rmsdcalc


class Conformations(object):
    """Structure to store coordinates and compute RMSDs between conformations.

    Conformations wraps a 3-dimensional numpy array of coordinates,
    transparently handling structure centering and G (matrix trace) computation
    required to use the IRMSD fast-Theobald RMSD routines.

    Note that `Conformations` will modify the array of coordinates it is given,
    when those structures are centered!
    """
    def __init__(self, coordinates, major, natoms):
        """Initialize a `Conformations` object.

        Arguments:
            coordinates: an M x N x P numpy ndarray of type float32. See
                         `major` for definition of dimensions.

            major: 'atom' or 'axis'. Specifies the storage format of
                    M x N x P ndarray `coordinates`:
                'axis': M = # conformations; N = # dimensions;
                        P = # padded atoms
                'atom': M = # conformations; N = # padded atoms;
                        P = # dimensions
                note that 'dimensions' must be 3 (points in 3D space)

            natoms: the number of actual, not padding, atoms in each structure
                    in the array

        There are special restrictions on `coordinates` to use the IRMSD
        routines:
            1. If the number of atoms `natoms` is not a multiple of 4, then
            npaddedatoms should be the next multiple of 4 that is larger than
            natoms, and the corresponding 'padding atoms' in the coordinates
            array must be all-zero.
            2. The coordinates array must be aligned to a 16-byte boundary.
            
        """
        if major not in ('atom', 'axis'):
            raise ValueError("Must specify atom or axis major coordinates")
        if coordinates.dtype != np.float32:
            raise ValueError("IRMSD can only handle single-precision float")

        self.natoms = natoms
        self.nconfs = coordinates.shape[0]

        if major == 'axis':
            self.ndims = coordinates.shape[1]
            self.npaddedatoms = coordinates.shape[2]
            self.axis_major = True
            self.atom_major = False
        else:
            self.ndims = coordinates.shape[2]
            self.npaddedatoms = coordinates.shape[1]
            self.axis_major = False
            self.atom_major = True
        
        self.major = major
        self.cords = coordinates
        self._G = None
        self._centered = False
        assert self.axis_major ^ self.atom_major

        if self.ndims != 3:
            raise ValueError("IRMSD only supports operation in 3 dimensions")
        if self.npaddedatoms % 4 != 0:
            raise ValueError("(Padded) number of atoms must be a multiple of 4"
                             " to use fast IRMSD routines")
        if self.cords.ctypes.data % 16 != 0:
            raise ValueError("Coordinate array must be aligned to a 16-byte "
                             "boundary to use fast IRMSD routines")
                
        return

    def center(self):
        """Transform conformations so that each is centered about 0.

        This function is automatically called if necessary for an alignment to
        proceed, but is exposed in case the user wants to center structures at a
        different time.

        Modifies data in-place since it might be very large.
        """
        for ci in xrange(self.nconfs):
            if self.atom_major:
                centroid = np.mean(confs[ci, self.natoms, :], axis=0)
                repcent = np.tile(centroid, (confs.shape[1], 1))
                confs[ci, :self.natoms, :] -= repcent
            else:
                centroid = np.mean(confs[ci, :, self.natoms], axis=1)
                repcent = np.tile(centroid, (1, confs.shape[1]))
                confs[ci, :, :self.natoms] -= repcent
        self._centered = True
        return
    
    @property
    def G(self):
        """Conformation traces

        For a structure S made of column vectors Sx, Sy, Sz representing the x,
        y, and z coordinates of each atom in the structure, G(S) = tr(S'S) =
        dot(x,x) + dot(y,y) + dot(z,z). This quantity is related to the radius
        of gyration and is needed in the Theobald RMSD computation.
        """
        if self._G is None:
            self._compute_g()
        return self._G

    def _compute_g(self):
        if not self._centered:
            self.center()
        self._G = np.zeros((self.nconfs,), dtype=np.float32)
        for i in xrange(self.nconfs):
            for j in xrange(self.ndims):
                if self.axis_major:
                    self._G[i] += np.dot(self.cords[i,j,:], self.cords[i,j,:])
                else:
                    self._G[i] += np.dot(self.cords[i,:,j], self.cords[i,:,j])
        return
    
    def rmsds_to_reference(self, otherconfs, refidx):
        """Compute RMSD of all conformations to a reference conformation.

        For each conformation in this Conformations object, compute the RMSD
        to a particular 'reference' conformation in another Conformations object
        `otherconfs`, identified by index `refidx`. Return a 1-D numpy array of
        the RMSDs.

        The underlying computation uses OpenMP and so will automatically
        parallelize across cores; it parallelizes over independent conformations
        in this Conformations object.

        To be compared, two Conformations objects must have the same atom/axis
        majority, the same number of atoms, and the same number of padded atoms.
        If the structures have not been centered or G values have not been
        calculated, they will be transparently computed here before the RMSDs
        are computed, and the results saved for future invocations.
        """
        if otherconfs.major != self.major:
            raise ValueError("Cannot align two conformation sets of differing "
                             "atom/axis majority")
        if otherconfs.natoms != self.natoms:
            raise ValueError("Cannot align two conformation sets of differing "
                             "number of atoms")
        if otherconfs.npaddedatoms != self.npaddedatoms:
            raise ValueError("Cannot align two conformation sets of differing "
                             "number of padded atoms")
        

        ref_structure = otherconfs.cords[refidx, :, :]
        # Getting G will implicitly center structures if needed
        G = self.G
        ref_G = otherconfs.G[refidx]
        
        if self.axis_major:
            return rmsdcalc.getMultipleRMSDs_axis_major(self.natoms,
                    self.npaddedatoms, self.npaddedatoms, self.cords,
                    ref_structure, G, ref_G)
        else:
            return rmsdcalc.getMultipleRMSDs_atom_major(self.natoms,
                    self.npaddedatoms, self.cords, ref_structure, G, ref_G)
