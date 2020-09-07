# List of changes to upgrade code from Python 2.7 to 3.8

## Installation

The outsource python packages has been updated as to be compatible with the new Python 3.8 interpreter:

* **Graph-tool**: installed with conda, version 2.29, newer versions fails for Python 3.8 (24/08/2020).
* **beautifulsoup4**, **lxml**, **numpy**, **opencv**, **pillow**, **scikit-image**, **scikit-learn**, **vtk**, **scikit-fmm**, **pytest**: latest version installed with conda.
* **astropy**: installed with ??, to substitute the deprecated PyFITS TODO.
* **imageio**: new in Python 3, to substitue scipy.misc.{imread,imsave} 

## Code modifications

Most of the job is done by the script **2to3** provided in Python 3. However, some additional modifications were required, they are listed below.

### Syntaxis

* In literals **is** and **is not** is substituted by **==** and **!=** respectively.

### Pickling

* Unpickling:

```{python}
    f_pkl = open(fname, 'rb') # Instead of just open(fname)
    try:
        gen_obj = pickle.load(f_pkl)
    finally:
        f_pkl.close()
```

### I/O for PNG images

Scipy I/O functions are no longer available for Python 3, now the functions:

```{python}
from imageio import imread, imwrite
# from scipy.misc import imsave, imread # For Python 2.7
...
# imsave is susbtituted by imread for all functions called
```

To allow 16 bits graylevel precision without loss for PNG images, the image data mapping to *uint16* type, usually from *float*, has to be 
done explicitly before the call **imwrite** function: 

```{python}
imwrite(part_name, lin_map(self.__particles[count],0,np.iinfo(np.uint16).max).astype(np.uint16))
```

### I/O for FITS image

Package **pyfits** is deprecated for Python 3, now it is include in the package **astropy**, that required the next chnages:

```{python}
from astropy.io import fits
# import pyfits # For Python 2.7
...
fits.writeto(fname, array, overwrite=True, output_verify='silentfix')
# pyfits.writeto(fname, array, overwrite=True, output_verify='silentfix') # For Python 2.7
...
hold = fits.getdata(DATA_DIR + '/mb_seg.fits')
# hold = pyfits.getdata(DATA_DIR + '/mb_seg.fits') # For Python 2.7
```

### Multiprocessing

In calls to **Array** function to create shared arrays as input parameter to *size_or_initializer* on accepts type **int** (e.g. **int64** raises errors), for that reason a explicit cast before the function call may be necessary:

```{python}
# particles_sh, masks_sh = mp.Array('f', part_h*part_r*npart), mp.Array('f', part_h*part_r*npart) # Python 2.7
particles_sh, masks_sh = mp.Array('f', int(part_h*part_r*npart)), mp.Array('f', int(part_h*part_r*npart))
```

### Pyto

In function **geometry.affine.transformArray** the old input paramter *origin* is now called **center**.

```{python}
# particle = r3d.transformArray(particle_u, origin=svol_cent, order=3, prefilter=True) # Python 2.7
particle = r3d.transformArray(particle_u, center=svol_cent, order=3, prefilter=True)
```

### Scikit-fmm

Because some unhandled and recently discovered numerical errors, **scikit-fmm** version=2019.1.30), the function **skfmm.travel_time()** is substitued by **skfmm.distance()**:

```{python}
dst_field = np.ma.MaskedArray(dst_field, np.invert(self.__voi))
self.__dst_field = skfmm.distance(dst_field, dx=1) 
# skfmm.travel_time(dst_field, self.__voi.astype(np.float32), dx=1) # Python 2.7
```