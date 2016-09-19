import holoviews as hv
import numpy as np
import pandas as pd
import oommffield
import re
import glob
import os


def field2inplane_vectorfield(field, slice_axis, slice_coord):
    """
    field2hv(field, slice_axis, slice_coord)

    This function constructs a Holoviews object
    which shows the in plane Magnetisation and out of plane Magnetisation

    Inputs
    ======
    field:
        Path to an OMF file or object of type oommffield.Field
    slice_axis:
        The axis along which the vector field will be sliced given as a string.
        Must be one of ['x', 'y', 'z']
    slice_coord:
        The coordinate along the slice_axis where the field is sliced
    """
    # Construct a field object if not a field object
    if isinstance(field, str):
        field = oommffield.read_oommf_file(field)
    field.normalise()
    if slice_axis == 'z':
        axis = (0, 1, 2)
    elif slice_axis == 'y':
        axis = (0, 2, 1)
    elif slice_axis == 'x':
        axis = (1, 2, 0)
    else:
        raise ValueError("Slice Axis must be one of 'x', 'y' ,'z'")
    dims = ['x', 'y', 'z']
    x, y, vec, coords = field.slice_field(slice_axis, slice_coord)
    X, Y = np.meshgrid(x, y)
    flat = vec.flatten()
    modm = (flat[axis[0]::3]**2 +
            flat[axis[1]::3]**2).reshape((len(x), len(y)))
    angm = np.arctan2(flat[axis[1]::3],
                      flat[axis[0]::3]).reshape((len(x), len(y)))
    kdims = kdims = [dims[axis[0]], dims[axis[1]]]
    return hv.VectorField([X, Y, angm, modm],
                          kdims=kdims, vdims=['xyfield'],
                          label='In-plane Magnetisation')


def field2inplane_angle(field, slice_axis, slice_coord):
    """
    field2hv(field, slice_axis, slice_coord)

    This function constructs a Holoviews object
    which shows the in plane Magnetisation and out of plane Magnetisation

    Inputs
    ======
    field:
        Path to an OMF file or object of type oommffield.Field
    slice_axis:
        The axis along which the vector field will be sliced given as a string.
        Must be one of ['x', 'y', 'z']
    slice_coord:
        The coordinate along the slice_axis where the field is sliced
    """
    # Construct a field object if not a field object
    if isinstance(field, str):
        field = oommffield.read_oommf_file(field)
    field.normalise()
    if slice_axis == 'z':
        axis = (0, 1, 2)
    elif slice_axis == 'y':
        axis = (0, 2, 1)
    elif slice_axis == 'x':
        axis = (1, 2, 0)
    else:
        raise ValueError("Slice Axis must be one of 'x', 'y' ,'z'")
    dims = ['x', 'y', 'z']
    bounds = [field.cmin[axis[0]],
              field.cmin[axis[1]],
              field.cmax[axis[0]],
              field.cmax[axis[1]]]
    x, y, vec, coords = field.slice_field(slice_axis, slice_coord)
    X, Y = np.meshgrid(x, y)
    flat = vec.flatten()
    angm = np.pi + np.arctan2(flat[axis[1]::3],
                              flat[axis[0]::3]).reshape((len(x), len(y)))
    kdims = [dims[axis[0]], dims[axis[1]]]
    return hv.Image(angm,
                    bounds=bounds,
                    kdims=kdims,
                    label='In-plane Magnetisation angle',
                    vdims=[hv.Dimension('xyfield'.format(slice_axis),
                                        range=(0, 2*np.pi))])


def field2outofplane(field, slice_axis, slice_coord):
    """
    field2hv(field, slice_axis, slice_coord)

    This function constructs a Holoviews object
    which shows the out of plane Magnetisation

    Inputs
    ======
    field:
        Path to an OMF file or object of type oommffield.Field
    slice_axis:
        The axis along which the vector field will be sliced given as a string.
        Must be one of ['x', 'y', 'z']
    slice_coord:
        The coordinate along the slice_axis where the field is sliced
    """
    # Construct a field object if not a field object
    if isinstance(field, str):
        field = oommffield.read_oommf_file(field)
    field.normalise()
    if slice_axis == 'z':
        axis = (0, 1, 2)
    elif slice_axis == 'y':
        axis = (0, 2, 1)
    elif slice_axis == 'x':
        axis = (1, 2, 0)
    else:
        raise ValueError("Slice Axis must be one of 'x', 'y' ,'z'")
    dims = ['x', 'y', 'z']
    bounds = [field.cmin[axis[0]],
              field.cmin[axis[1]],
              field.cmax[axis[0]],
              field.cmax[axis[1]]]
    x, y, vec, coords = field.slice_field(slice_axis, slice_coord)
    X, Y = np.meshgrid(x, y)
    flat = vec.flatten()
    mz = flat[axis[2]::3].reshape((len(x), len(y)))
    kdims = kdims = [dims[axis[0]], dims[axis[1]]]
    return hv.Image(mz, bounds=bounds,
                    label='Out of plane Magnetisation',
                    kdims=kdims,
                    vdims=[hv.Dimension('M{}'.format(slice_axis),
                                        range=(-1, 1))])


def create_inplane_holomap(files, slice_coordinates, axis='z'):
    physical_dimension = hv.Dimension('SliceDimension')
    file_dimension = hv.Dimension('File')
    slice_dimension = hv.Dimension('{} coordinate'.format('z'))
    filename_fun = lambda filename: int(filename.split('-')[3])
    slicecoords = list(slice_coordinates)
    inplane = [((filename_fun(file), slicecoord),
                field2inplane_vectorfield(file, axis, slicecoord))
               for file in files
               for slicecoord in slicecoords]
    return hv.HoloMap(inplane, kdims=[file_dimension, slice_dimension])


def create_outofplane_holomap(files, slice_coordinates, axis='z'):
    physical_dimension = hv.Dimension('SliceDimension')
    file_dimension = hv.Dimension('File')
    slice_dimension = hv.Dimension('{} coordinate'.format('z'))
    filename_fun = lambda filename: int(filename.split('-')[3])
    slicecoords = list(slice_coordinates)
    outofplane = [((filename_fun(file), slicecoord),
                   field2outofplane(file, axis, slicecoord))
                  for file in files
                  for slicecoord in slicecoords]

    return hv.HoloMap(outofplane, kdims=[file_dimension, slice_dimension])


def create_inplane_dynamic_map(files, slice_coordinates, axis='z'):
    filename_fun = lambda filename: int(filename.split('-')[3])
    file_dimension = hv.Dimension(
        'field', values=list(files), value_format=filename_fun)
    physical_dimension = hv.Dimension('slice_axis', values=[axis])
    slice_dimension = hv.Dimension(
        'slice_coord', values=list(slice_coordinates))
    return hv.DynamicMap(field2inplane_vectorfield,
           kdims=[file_dimension, physical_dimension, slice_dimension])


def create_outofplane_dynamic_map(files, slice_coordinates, axis='z'):
    filename_fun = lambda filename: int(filename.split('-')[3])
    file_dimension = hv.Dimension(
        'field', values=list(files), value_format=filename_fun)
    physical_dimension = hv.Dimension('slice_axis', values=[axis])
    slice_dimension = hv.Dimension(
        'slice_coord', values=list(slice_coordinates))
    return hv.DynamicMap(field2outofplane, kdims=[file_dimension, physical_dimension, slice_dimension])
