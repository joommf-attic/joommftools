import holoviews as hv
import numpy as np

def field2hv(field, slice_axis, slice_coord):
    # Read file if not a field object
    if isinstance(field, str):
        field = oommffield.read_oommf_file(field)
    if slice_axis == 'z':
        axis = (0, 1, 2)
    elif slice_axis == 'y':
        axis = (0, 2, 1)
    elif slice_axis == 'x':
        axis = (1, 2, 0)
    dims = ['x', 'y', 'z']
    bounds = [field.cmin[axis[0]], field.cmin[axis[1]], field.cmax[axis[0]], field.cmax[axis[1]]]
    x, y, vec, coords = field.slice_field(slice_axis, slice_coord)
    X, Y = np.meshgrid(x, y)
    flat = vec.flatten()
    modm = (flat[axis[0]::3]**2 + flat[axis[1]::3]**2).reshape((len(x), len(y)))
    angm = np.arctan2(flat[axis[1]::3], flat[axis[0]::3]).reshape((len(x), len(y)))
    outofplane = flat[axis[2]::3]

    cmap = flat[axis[2]::3].reshape((len(x), len(y)))
    label = 'Slice at {} = {}'.format(slice_axis, slice_coord)
    return hv.VectorField([X, Y, angm, modm], kdims=[dims[axis[0]], dims[axis[1]]], label=label) + \
           hv.Image(cmap, bounds=bounds, label='M_{} through slice'.format(dims[2]))
    #return X, Y, x, y, cmap