__author__ = 'martinez'

try:
    import pexceptions
except:
    import pyseg.pexceptions
import numpy as np
try:
    import disperse_io
except:
    import pyseg.pexceptions
from pyto.io import ImageIO
from abc import *

##########################################################################################
#   Class for holding the Geometry of a VertexMCF
#
#
class GeometryMCF:

    # coords: array (shape=Nx3) with the coordinates of this geometry in the original tomogram
    # list: array (shape=N) with the densities of this coordinates
    def __init__(self, coords, densities):
        self.__image = None
        self.__mask = None
        self.__size = None
        self.__resolution = 1 # nm / voxel
        self.__offset = None
        self.__total_density = None
        self.__avg_density = None
        self.__density = None
        self.__build_image(coords, densities)

    ########## Implemented functionality area

    # In nm/voxel, by default 1
    def set_resolution(self, resolution=1):
        self.__resolution = resolution

    def get_resolution(self):
        return self.__resolution

    # Size of the subvol which holds the geometry
    def get_size(self):
        return self.__size

    # Offset of the subvol which holds the geometry
    def get_offset(self):
        return self.__offset

    # Return density mean
    def get_mean(self):
        return np.mean(self.__density)

    # Return the standard deviation
    def get_std(self):
        return np.std(self.__density)

    # Top-left-front and Bottom-down-back corners is a 6 length Tuple
    def get_bound(self):
        return (self.__offset[0],
                self.__offset[1],
                self.__offset[2],
                self.__offset[0] + self.__size[0],
                self.__offset[1] + self.__size[1],
                self.__offset[2] + self.__size[2])

    # lbl: if not None all voxels are set to lbl
    def get_numpy_image(self, lbl=None):

        if lbl is None:
            return self.__image
        else:
            return self.__mask * lbl

    # lbl: if not None all voxels are set to lbl
    def get_numpy_mask(self, lbl=None):
        if lbl is None:
            return self.__mask
        else:
            return self.__mask * lbl

    # Return geometry volume in nm
    def get_volume(self):
        dv = self.__resolution * self.__resolution * self.__resolution
        return self.__mask.sum() * dv

    # Prints the geometry into a numpy volume
    # vol: numpy array big enough for holding the geometry
    # lbl: label used for printing, if None (default) density is printed
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    def print_in_numpy(self, vol, lbl=None, th_den=None):

        o = self.__offset
        s = self.__size
        w = o + s
        if (w[0] > vol.shape[0]) or (w[1] > vol.shape[1]) or (w[2] > vol.shape[2]):
            error_msg = 'The volume cannot hold the geometry.'
            raise pexceptions.PySegInputWarning(expr='print_in_numpy (Geometry)', msg=error_msg)

        if th_den is not None:
            mean = self.get_mean()
            std = self.get_std()
            self.__mask[self.__image > (mean + th_den*std)] = False
        subvol = self.get_numpy_image(lbl)
        mask_arr = self.get_array_mask()
        vol[mask_arr[:, 0], mask_arr[:, 1], mask_arr[:, 2]] = subvol[self.__mask]

    def get_mask_sum(self):
        return self.__mask.sum()

    # Return an array which indexes mask foreground voxels
    # mode: 1 (default) in manifold coordinates system, otherwise in local geometry system
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    def get_array_mask(self, mode=1, th_den=None):

        # Building the meshgrid
        if mode == 1:
            x_a = np.arange(self.__offset[0], self.__offset[0]+self.__size[0])
            y_a = np.arange(self.__offset[1], self.__offset[1]+self.__size[1])
            z_a = np.arange(self.__offset[2], self.__offset[2]+self.__size[2])
        else:
            x_a = np.arange(0, self.__size[0])
            y_a = np.arange(0, self.__size[1])
            z_a = np.arange(0, self.__size[2])
        mg_y, mg_x, mg_z = np.meshgrid(y_a, x_a, z_a)

        if th_den is None:
            hold_mask = self.__mask
        else:
            mean = self.get_mean()
            std = self.get_std()
            hold_mask = np.copy(self.__mask)
            hold_mask[self.__image > (mean + th_den*std)] = False

        # Building coordinates array
        x = mg_x[hold_mask]
        y = mg_y[hold_mask]
        z = mg_z[hold_mask]
        if (len(x.shape) != 1) or (x.shape != y.shape) or (x.shape != z.shape):
            error_msg = 'Unexpected state.'
            raise pexceptions.PySegTransitionError(expr='get_array_mask (GeometryMCF)', msg=error_msg)
        mask_array = np.zeros(shape=(x.size, 3), dtype=np.int)
        mask_array[:, 0] = x.astype(np.int)
        mask_array[:, 1] = y.astype(np.int)
        mask_array[:, 2] = z.astype(np.int)

        return mask_array

    def get_total_density(self):
        if self.__total_density is None:
            self.__total_density = np.sum(self.__density)
        return self.__total_density

    # It only works if input density map is in range [0, 1]
    def get_total_density_inv(self):
        return np.sum(1 - self.__density)

    # It this value has been already computed this function is faster than mean_density
    def get_avg_density(self):
        if self.__total_density is None:
            self.__avg_density = np.mean(self.__density)
        return self.__avg_density

    # Return all densities in one dimensional array
    def get_densities(self):
        return self.__density

    # Extend the geometry with another
    # geom: this geometry is added to the current geometry
    def extend(self, geom):

        if geom.get_resolution() != self.__resolution:
            error_msg = 'Input geometry resolution does not match current geomtry resolution.'
            raise pexceptions.PySegTransitionError(expr='extend (GeometryMCF)', msg=error_msg)

        # Compute new geometry size and offset
        hold_offset = np.zeros(shape=3, dtype=np.int)
        off_img_s = np.zeros(shape=3, dtype=np.int)
        off_img_g = np.zeros(shape=3, dtype=np.int)
        if self.__offset[0] < geom.__offset[0]:
            hold_offset[0] = self.__offset[0]
            off_img_g[0] = geom.__offset[0] - self.__offset[0]
        else:
            hold_offset[0] = geom.__offset[0]
            off_img_s[0] = self.__offset[0] - geom.__offset[0]
        if self.__offset[1] < geom.__offset[1]:
            hold_offset[1] = self.__offset[1]
            off_img_g[1] = geom.__offset[1] - self.__offset[1]
        else:
            hold_offset[1] = geom.__offset[1]
            off_img_s[1] = self.__offset[1] - geom.__offset[1]
        if self.__offset[2] < geom.__offset[2]:
            hold_offset[2] = self.__offset[2]
            off_img_g[2] = geom.__offset[2] - self.__offset[2]
        else:
            hold_offset[2] = geom.__offset[2]
            off_img_s[2] = self.__offset[2] - geom.__offset[2]
        hold_size = np.zeros(shape=3, dtype=np.int)
        hold_s = self.__offset + self.__size
        hold_g = geom.__offset + geom.__size
        if hold_s[0] > hold_g[0]:
            hold_size[0] = hold_s[0]
        else:
            hold_size[0] = hold_g[0]
        if hold_s[1] > hold_g[1]:
            hold_size[1] = hold_s[1]
        else:
            hold_size[1] = hold_g[1]
        if hold_s[2] > hold_g[2]:
            hold_size[2] = hold_s[2]
        else:
            hold_size[2] = hold_g[2]
        hold_size -= hold_offset

        # Create the extended container arrays
        hold_density = np.concatenate((self.__density, geom.__density))
        hold_image = np.zeros(shape=hold_size, dtype=self.__image.dtype)
        hold_mask = np.zeros(shape=hold_size, dtype=self.__mask.dtype)
        xsl, ysl, zsl = off_img_s[0], off_img_s[1], off_img_s[2]
        xsh, ysh, zsh = off_img_s[0]+self.__size[0], off_img_s[1]+self.__size[1], \
                        off_img_s[2]+self.__size[2]
        hold_mask[xsl:xsh, ysl:ysh, zsl:zsh] = self.__mask
        hold_image[hold_mask] = self.__image[self.__mask]
        hold_mask_2 = np.zeros(shape=hold_size, dtype=geom.__mask.dtype)
        xgl, ygl, zgl = off_img_g[0], off_img_g[1], off_img_g[2]
        xgh, ygh, zgh = off_img_g[0]+geom.__size[0], off_img_g[1]+geom.__size[1], \
                        off_img_g[2]+geom.__size[2]
        hold_mask_2[xgl:xgh, ygl:ygh, zgl:zgh] = geom.__mask
        hold_image[hold_mask_2] = geom.__image[geom.__mask]

        # Update object state
        self.__offset = hold_offset
        self.__size = hold_size
        self.__density = hold_density
        self.__image = hold_image
        self.__mask = hold_mask + hold_mask_2
        self.__avg_density = None
        self.__total_density = None

    ###### Internal methods

    def __build_image(self, coords, densities):

        # Get subvol borders
        self.__offset = np.min(coords, axis=0)
        self.__size = np.max(coords, axis=0)
        self.__size = self.__size - self.__offset + 1
        self.__density = densities

        # Creates image and mask
        self.__mask = np.zeros(shape=self.__size, dtype=np.bool)
        self.__image = np.zeros(shape=self.__size, dtype=densities.dtype)

        # Fill up image and mask
        self.__total_density = 0
        for i in range(len(densities)):
            x, y, z = coords[i]
            x, y, z = (x, y, z) - self.__offset
            self.__mask[x, y, z] = True
            den = densities[i]
            self.__image[x, y, z] = den
            self.__total_density += den
        self.__avg_density = self.__total_density / len(densities)

##########################################################################################
#   Abstract class for working as interface to geometries
#
#
class Geometry(metaclass=ABCMeta):

    # For Abstract Base Classes in python
    def __init__(self, manifold, density):

        if (not isinstance(manifold, np.ndarray)) and (not isinstance(density, np.ndarray)):
            error_msg = 'Booth manifold and density must be np.ndarray objects.'
            raise pexceptions.PySegInputError(expr='__init___ (Geometry)', msg=error_msg)
        self.__manifold = manifold
        self.__density = density
        self.__image = None
        self.__mask = None
        self.__size = None
        self.__resolution = 1 # nm / voxel
        self.__offset = None
        self.__build_image()
        self.__total_density = None

    ########## Implemented functionality area

    # In nm/voxel, by default 1
    def set_resolution(self, resolution=1):
        self.__resolution = resolution

    def get_manifold(self):
        return self.__manifold

    def get_density(self):
        return self.__density

    # Size of the subvol which holds the geometry
    def get_size(self):
        return self.__size

    # Offset of the subvol which holds the geometry
    def get_offset(self):
        return self.__offset

    # Top-left-front and Bottom-down-back corners is a 6 length Tuple
    def get_bound(self):
        return (self.__offset[0],
                self.__offset[1],
                self.__offset[2],
                self.__offset[0] + self.__size[0],
                self.__offset[1] + self.__size[1],
                self.__offset[2] + self.__size[2])

    # lbl: if not None all voxels are set to lbl
    def get_numpy_image(self, lbl=None):

        if lbl is None:
            return self.__image
        else:
            return self.__mask * lbl

    # lbl: if not None all voxels are set to lbl
    def get_numpy_mask(self, lbl=None):
        return self.__mask

    # Return an array which indexes mask foreground voxels
    # mode: 1 (default) in manifold coordinates system, otherwise in local geometry system
    def get_array_mask(self, mode=1):

        # Building the meshgrid
        if mode == 1:
            x_a = np.arange(self.__offset[0], self.__offset[0]+self.__size[0])
            y_a = np.arange(self.__offset[1], self.__offset[1]+self.__size[1])
            z_a = np.arange(self.__offset[2], self.__offset[2]+self.__size[2])
        else:
            x_a = np.arange(0, self.__size[0])
            y_a = np.arange(0, self.__size[1])
            z_a = np.arange(0, self.__size[2])
        mg_y, mg_x, mg_z = np.meshgrid(y_a, x_a, z_a)

        # Buiding coordinates array
        x = mg_x[self.__mask]
        y = mg_y[self.__mask]
        z = mg_z[self.__mask]
        if (len(x.shape) != 1) or (x.shape != y.shape) or (x.shape != z.shape):
            error_msg = 'Unexpected state.'
            raise pexceptions.PySegTransitionError(expr='get_array_mask (Geometry)', msg=error_msg)
        mask_array = np.zeros(shape=(x.size, 3), dtype=np.int)
        mask_array[:, 0] = x.astype(np.int)
        mask_array[:, 1] = y.astype(np.int)
        mask_array[:, 2] = z.astype(np.int)

        return mask_array

    # Eliminates a voxel in the geometry
    def delete_voxel(self, x, y, z):
        xi = int(x)
        yi = int(y)
        zi = int(z)
        self.__mask[xi, yi, zi] = True
        self.__mask[xi, yi, zi] = 0

    # lbl: if not None all voxels are set to lbl
    def get_vtk_image(self, lbl=None):

        if lbl is None:
            return disperse_io.numpy_to_vti(self.__image, self.__offset, self.__resolution*[1, 1, 1])
        else:
            return disperse_io.numpy_to_vti(self.__mask*lbl, self.__offset, self.__resolution*[1, 1, 1])

    # lbl: if not None all voxels are set to lbl
    def save_mrc_image(self, file_name, lbl=None):

        mrc_image = ImageIO()
        if lbl is None:
            mrc_image.setData(self.__image)
        else:
            mrc_image.setData(self.__image * lbl)
        mrc_image.writeMRC(file=file_name, length=self.__resolution*self.__size, nstart=self.__offset)

    # Return density mean
    def get_mean(self):
        return np.mean(self.__density[self.__mask])

    # Return the standard deviation
    def get_std(self):
        return np.std(self.__density[self.__mask])

    # Apply an external to the geometry. The mask must be a numpy array big enough for embedding
    # the geometry with format: 1-fg, 0-bg
    def apply_ext_mask(self, mask):

        # Cropping the mask
        subvol = mask[self.__offset[0]:(self.__offset[0]+self.__size[0]),
                        self.__offset[1]:(self.__offset[1]+self.__size[1]),
                        self.__offset[2]:(self.__offset[2]+self.__size[2])]

        self.__mask = self.__mask * subvol
        self.__image = self.__image * subvol

    # Prints the geometry into a numpy volume
    # vol: numpy array big enough for holding the geometry
    # lbl: label used for printing, if None (default) density is printed
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    def print_in_numpy(self, vol, lbl=None, th_den=None):

        o = self.__offset
        s = self.__size
        w = o + s
        if (w[0] > vol.shape[0]) or (w[1] > vol.shape[1]) or (w[2] > vol.shape[2]):
            error_msg = 'The volume cannot hold the geometry.'
            raise pexceptions.PySegInputWarning(expr='print_in_numpy (Geometry)', msg=error_msg)

        if th_den is not None:
            mean = self.get_mean()
            std = self.get_std()
            self.__mask[self.__density[o[0]:w[0], o[1]:w[1], o[2]:w[2]] > (mean + th_den*std)] = False
        subvol = self.get_numpy_image(lbl)
        mask_arr = self.get_array_mask()
        vol[mask_arr[:, 0], mask_arr[:, 1], mask_arr[:, 2]] = subvol[self.__mask]

    def get_total_density(self):

        if self.__total_density is None:
            self.__total_density = np.sum(self.__image[self.__mask])

        return self.__total_density

    ###### Abstract methods
    @abstractmethod
    def __build_image(self):
        raise NotImplementedError(
            '__build_image() (Geometry). Abstract method, it requires an implementation.')

##########################################################################################
#   Class for holding thee dimensional geometry of a single vertex
#
#
class PointGeometry(Geometry):

    # pcoord: coordinates (x,y,z) of the point withing the manifold
    # manifold: image with the manifold (Numpy ndarray)
    # density: image with the density (Numpy ndarray)
    def __init__(self, pcoord, manifold, density):

        if len(pcoord) != 3:
            error_msg = 'Input coordinates must be a 3D vector.'
            raise pexceptions.PySegInputError(expr='__init___ (PointGeometry)', msg=error_msg)
        self.__seed_coord = pcoord
        super(PointGeometry, self).__init__(manifold, density)

    ######### Set/Get functions area


    ######### External area functionality


    ########### Internal Functionality area

    # Build the 3D sub_image with the mask and the density for the geometry which surrounds a point
    def _Geometry__build_image(self):

        # Compute size
        lbl = self._Geometry__manifold[int(np.floor(self.__seed_coord[0])),
                                       int(np.floor(self.__seed_coord[1])),
                                       int(np.floor(self.__seed_coord[2]))]
        idx = np.where(self._Geometry__manifold == lbl)
        xmin = np.min(idx[0])
        ymin = np.min(idx[1])
        zmin = np.min(idx[2])
        xmax = np.max(idx[0])
        ymax = np.max(idx[1])
        zmax = np.max(idx[2])
        self._Geometry__offset = np.asarray([xmin, ymin, zmin])
        self._Geometry__size = np.asarray([xmax-xmin+1, ymax-ymin+1, zmax-zmin+1])
        for i in range(self._Geometry__size.shape[0]):
            if (self._Geometry__size[i] < 0) or\
                    (self._Geometry__size[i] > self._Geometry__manifold.shape[i]):
                error_msg = 'Dimension lower than zero or bigger than input manifold.'
                raise pexceptions.PySegTransitionError(expr='__build_image (PointGeometry)', msg=error_msg)

        # Get image and mask
        xmax1 = xmax + 1
        ymax1 = ymax + 1
        zmax1 = zmax + 1
        self._Geometry__mask = self._Geometry__manifold[xmin:xmax1, ymin:ymax1, zmin:zmax1] == lbl
        self._Geometry__image = self._Geometry__density[xmin:xmax1, ymin:ymax1, zmin:zmax1] \
                                * self._Geometry__mask

##########################################################################################
#   Class for holding thee dimensional geometry of an Arc
#
#
class ArcGeometry(Geometry):

    # pcoords: array of coordinates (x,y,z) of the points withing the manifold
    # manifold: image with the manifold (Numpy ndarray)
    # density: image with the density (Numpy ndarray)
    def __init__(self, pcoords, manifold, density):

        if (len(pcoords.shape) != 2) or (pcoords.shape[1] != 3):
            error_msg = 'Input coordinates must be a numpy array 3D vectors.'
            raise pexceptions.PySegInputError(expr='__init___ (ArcGeometry)', msg=error_msg)
        self.__seed_coords = pcoords
        super(ArcGeometry, self).__init__(manifold, density)

    ######### Set/Get functions area


    ######### External area functionality


    ########### Internal Functionality area

    # Build the 3D sub_image with the mask and the density for the geometry which surrounds an array
    # of points connected a string
    def _Geometry__build_image(self):

        # Compute image size
        xmin = 0
        ymin = 0
        zmin = 0
        xmax = self._Geometry__manifold.shape[0]
        ymax = self._Geometry__manifold.shape[1]
        zmax = self._Geometry__manifold.shape[2]
        lbls = np.zeros(shape=self.__seed_coords.shape[0], dtype=np.int)
        idxs = (-1) * np.ones(shape=lbls.shape, dtype=object)
        for i in range(self.__seed_coords.shape[0]):
            lbls[i] = self._Geometry__manifold[int(round(self.__seed_coords[i][0])),
                                               int(round(self.__seed_coords[i][1])),
                                               int(round(self.__seed_coords[i][2]))]
            idx = np.where(self._Geometry__manifold == lbls[i])
            idxs[i] = idx
            hold = idx[0].min()
            if hold < xmin: xmin = hold
            hold = idx[1].min()
            if hold < ymin: ymin = hold
            hold = idx[2].min()
            if hold < zmin: zmin = hold
            hold = idx[0].min()
            if hold > xmax: xmax = hold
            hold = idx[1].max()
            if hold > ymax: ymax = hold
            hold = idx[2].max()
            if hold > zmax: zmax = hold
        self._Geometry__offset = np.asarray((xmin, ymin, zmin))
        self._Geometry__size = np.asarray((xmax, ymax, zmax))
        self._Geometry__size -= self._Geometry__offset

        # Create the image and mask
        self._Geometry__mask = np.zeros(shape=self._Geometry__size, dtype=np.bool)
        self._Geometry__image = np.zeros(shape=self._Geometry__size, dtype=self._Geometry__density.dtype)
        for i in range(self.__seed_coords.shape[0]):
            self._Geometry__mask[idxs[i]] = True
            self._Geometry__image[idxs[i]] = self._Geometry__density[idxs[i]]

