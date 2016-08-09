def field2hv(field, slice_axis='z', slice_coord = 0):
    if slice_axis == 'z':
        axis = (0, 1, 2)
    elif slice_axis == 'y':
        axis = (0, 2, 1)
    elif slice_axis == 'x':
        axis = (1, 2, 0)
    x, y, vec, coords = field.slice_field(slice_axis, slice_coord)
    X, Y = np.meshgrid(x, y)

    flat = vec.flatten()
    modm = (flat[axis[0]::3]**2 + flat[axis[1]::3]**2).reshape((len(x), len(y)))
    angm = np.arctan2(flat[axis[1]::3], flat[axis[0]::3]).reshape((len(x), len(y)))
    cmap = flat[axis[2]::3].reshape((len(x), len(y)))
    return hv.VectorField([X, Y, angm, modm])
