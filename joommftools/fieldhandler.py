import holoviews as hv
import numpy as np
import pandas as pd
import oommfodt
import discretisedfield
import re
import glob
import os

def filename_fun(filename):
    return int(filename.split('-')[3])


def field2inplane_vectorfield(field, slice_axis, slice_coord, opts=None):
    """
    field2hv(field, slice_axis, slice_coord)

    This function constructs a Holoviews object
    which shows the in plane Magnetisation and out of plane Magnetisation

    Inputs
    ======
    field:
        Path to an OMF file or object of type discretisedfield.Field
    slice_axis:
        The axis along which the vector field will be sliced given as a string.
        Must be one of ['x', 'y', 'z']
    slice_coord:
        The coordinate along the slice_axis where the field is sliced
    """
    # Construct a field object if not a field object
    if isinstance(field, str):
        field = discretisedfield.read_oommf_file(field, normalisedto=1)
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
                          label='In-plane Magnetisation', options=opts)


def field2inplane_angle(field, slice_axis, slice_coord, opts=None):
    """
    field2hv(field, slice_axis, slice_coord)

    This function constructs a Holoviews object
    which shows the in plane Magnetisation and out of plane Magnetisation

    Inputs
    ======
    field:
        Path to an OMF file or object of type discretisedfield.Field
    slice_axis:
        The axis along which the vector field will be sliced given as a string.
        Must be one of ['x', 'y', 'z']
    slice_coord:
        The coordinate along the slice_axis where the field is sliced
    """
    # Construct a field object if not a field object
    if isinstance(field, str):
        field = discretisedfield.read_oommf_file(field, normalisedto=1)
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
    print(dir(bounds))
    bounds = [field.mesh.p1[axis[0]],
              field.mesh.p1[axis[1]],
              field.mesh.p2[axis[0]],
              field.mesh.p2[axis[1]]]
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
                                        range=(0, 2*np.pi))], options=opts)


def field2outofplane(field, slice_axis, slice_coord, opts=None):
    """
    field2hv(field, slice_axis, slice_coord)

    This function constructs a Holoviews object
    which shows the out of plane Magnetisation

    Inputs
    ======
    field:
        Path to an OMF file or object of type discretisedfield.Field
    slice_axis:
        The axis along which the vector field will be sliced given as a string.
        Must be one of ['x', 'y', 'z']
    slice_coord:
        The coordinate along the slice_axis where the field is sliced
    """
    # Construct a field object if not a field object
    if isinstance(field, str):
        field = discretisedfield.read_oommf_file(field, normalisedto=1)
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
    bounds = [field.mesh.p1[axis[0]],
              field.mesh.p1[axis[1]],
              field.mesh.p2[axis[0]],
              field.mesh.p2[axis[1]]]
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
    """
    Creates an inplane magnetisation vector field plot holomap
    for the given list of files.
    Input
    -----
    files:
        List of strings containing file paths to OMF files.
        This must be ordered in the way that is needed in the
        HoloMap
    slice_coordinates:
        The coordinates along the slice_axis where the field is sliced
    slice_axis:
        The axis along which slices are taken.
    """
    physical_dimension = hv.Dimension('SliceDimension')
    file_dimension = hv.Dimension('File')
    slice_dimension = hv.Dimension('{} coordinate'.format('z'))
    slicecoords = list(slice_coordinates)
    inplane = [((filename_fun(file), slicecoord),
                field2inplane_vectorfield(file, axis, slicecoord))
               for file in files
               for slicecoord in slicecoords]
    return hv.HoloMap(inplane, kdims=[file_dimension, slice_dimension])


def create_inplane_holomap(files, slice_coordinates, axis='z'):
    """
    Creates an inplane magnetisation vector field plot holomap
    for the given list of files.
    Input
    -----
    files:
        List of strings containing file paths to OMF files.
        This must be ordered in the way that is needed in the
        HoloMap
    slice_coordinates:
        The coordinates along the slice_axis where the field is sliced
    slice_axis:
        The axis along which slices are taken.
    """
    physical_dimension = hv.Dimension('SliceDimension')
    file_dimension = hv.Dimension('File')
    slice_dimension = hv.Dimension('{} coordinate'.format('z'))
    slicecoords = list(slice_coordinates)
    inplane = [((filename_fun(file), slicecoord),
                field2inplane_angle(file, axis, slicecoord))
               for file in files
               for slicecoord in slicecoords]
    return hv.HoloMap(inplane, kdims=[file_dimension, slice_dimension])


def create_outofplane_holomap(files, slice_coordinates, axis='z'):
    """
    Creates an out of plane magnetisation magnitude plot holomap
    for the given list of files.
    Input
    -----
    files:
        List of strings containing file paths to OMF files.
        This must be ordered in the way that is needed in the
        HoloMap
    slice_coordinates:
        The coordinates along the slice_axis where the field is sliced
    slice_axis:
        The axis along which slices are taken.
    """
    physical_dimension = hv.Dimension('SliceDimension')
    file_dimension = hv.Dimension('File')
    slice_dimension = hv.Dimension('{} coordinate'.format('z'))
    slicecoords = list(slice_coordinates)
    outofplane = [((filename_fun(file), slicecoord),
                   field2outofplane(file, axis, slicecoord))
                  for file in files
                  for slicecoord in slicecoords]

    return hv.HoloMap(outofplane, kdims=[file_dimension, slice_dimension])


def create_inplane_dynamic_map(files, slice_coordinates, axis='z'):
    """
    Creates an inplane magnetisation vector field plot dynamic map
    for the given list of files.
    Input
    -----
    files:
        List of strings containing file paths to OMF files.
        This must be ordered in the way that is needed in the
        HoloMap
    slice_coordinates:
        The coordinates along the slice_axis where the field is sliced
    slice_axis:
        The axis along which slices are taken.
    """
    file_dimension = hv.Dimension(
        'File', value_format=filename_fun, values=list(files))
    physical_dimension = hv.Dimension('slice_axis', values=[axis])
    slice_dimension = hv.Dimension(
        'slice_coord', values=list(slice_coordinates))
    return hv.DynamicMap(field2inplane_vectorfield,
                         kdims=[file_dimension,
                                physical_dimension,
                                slice_dimension])


def create_outofplane_dynamic_map(files, slice_coordinates, axis='z'):
    """
    Creates an out of plane magnetisation magnitude plot dynamic map
    for the given list of files.
    Input
    -----
    files:
        List of strings containing file paths to OMF files.
        This must be ordered in the way that is needed in the
        HoloMap
    slice_coordinates:
        The coordinates along the slice_axis where the field is sliced
    slice_axis:
        The axis along which slices are taken.
    """
    file_dimension = hv.Dimension(
        'File', value_format=filename_fun, values=list(files))
    physical_dimension = hv.Dimension('slice_axis', values=[axis])
    slice_dimension = hv.Dimension(
        'slice_coord', values=list(slice_coordinates))
    return hv.DynamicMap(field2outofplane,
                         kdims=[file_dimension,
                                physical_dimension,
                                slice_dimension])


def create_inplane_angle_dynamic_map(files, slice_coordinates, axis='z'):
    """
    Creates an inplane magnetisation vector field plot dynamic map
    for the given list of files.
    Input
    -----
    files:
        List of strings containing file paths to OMF files.
        This must be ordered in the way that is needed in the
        HoloMap
    slice_coordinates:
        The coordinates along the slice_axis where the field is sliced
    slice_axis:
        The axis along which slices are taken.
    """
    file_dimension = hv.Dimension(
        'File', value_format=filename_fun, values=list(files))
    physical_dimension = hv.Dimension('slice_axis', values=[axis])
    slice_dimension = hv.Dimension(
        'slice_coord', values=list(slice_coordinates))
    return hv.DynamicMap(field2inplane_angle,
                         kdims=[file_dimension,
                                physical_dimension,
                                slice_dimension])


def field2topological_density(field, slice_axis, slice_coord):
    """
    field2topological_density(field, slice_axis, slice_coord)

    This function constructs a Holoviews object
    which shows the topological density.

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
    shape = np.shape(vec)[0], np.shape(vec)[1]
    mbig = np.zeros((shape[0] + 2, shape[1] + 2, 3))
    mbig[1:-1, 1:-1] = vec
    Q = np.zeros((shape[0] + 2, shape[1] + 2))
    print(shape)

    for i in range(1, shape[0]+1):
        for j in range(1, shape[1]+1):
            Q[i, j] = (mbig[i, j].dot(np.cross(mbig[i+1, j], mbig[i, j+1])) +
                       mbig[i, j].dot(np.cross(mbig[i-1, j], mbig[i, j-1])) -
                       mbig[i, j].dot(np.cross(mbig[i-1, j], mbig[i, j+1])) -
                       mbig[i, j].dot(np.cross(mbig[i+1, j], mbig[i, j-1]))
                       )

    return hv.Image(Q[1:-1, 1:-1],
                    label='Topological Density',
                    bounds=bounds,
                    kdims=[dims[axis[0]], dims[axis[1]]],
                    vdims=[hv.Dimension('Q_{}'.format(slice_axis))])


class ODT2hv:

    def __init__(self, odtpath, omfpaths):
        """
        ODT2hv(field, slice_axis, slice_coord)

        This function takes a list of OMF files and matches the filenames
        with outputs in a corresponding ODT file. Graphs of properties can
        be plotted using the method ODT2hv.get_curve

        Inputs
        ======
        odtpath:
            Path to an OOMMF ODT file.
        omfpaths:
            List of OMF files.
        """
        self.omfpaths = omfpaths
        strarray = [re.findall(r"[\w']+", file)[-3:-1] for file in omfpaths]
        relevantfiles = np.array([[int(i[0]), int(i[1])] for i in strarray])
        index = pd.DataFrame(relevantfiles, columns=('stage', 'iteration'))
        odtframe = oommfodt.OOMMFodt(odtpath).df
        reduced = pd.merge(index, odtframe)
        reduced = reduced.reset_index()
        reduced.rename(columns={'index': 'File'}, inplace=True)
        self.frame = reduced
        self.headers = list(reduced.columns)[1:]
        self.hv = hv.Table(self.frame)

    def get_curve(self, file, graph):
        """
        Inputs
        ------
        file:
            OMF filename
        graph:
            One of ODT2hv.headers, a header from the ODT file.
        """

        if isinstance(file, str):
            try:
                index = self.omfpaths.index(file)
            except:
                raise ValueError("File not in list of OMF files")
        else:
            index = file
        return self.hv.to.curve('File', graph, []) * \
            hv.VLine(index)

    def create_holomap(self):
        file_dimension = hv.Dimension('File')
        graphdim = hv.Dimension('Graph')
        inplane = [((filename_fun(file), graph),
                    self.get_curve(file, graph))
                   for file in self.omfpaths
                   for graph in self.headers]
        return hv.HoloMap(inplane, kdims=[file_dimension, graphdim])

    def create_dmap(self):
        file_dimension = hv.Dimension(
            'File', values=self.omfpaths, value_format=filename_fun)
        graph = hv.Dimension('Graph', values=self.headers)
        return hv.DynamicMap(self.get_curve, kdims=[file_dimension, graph])


