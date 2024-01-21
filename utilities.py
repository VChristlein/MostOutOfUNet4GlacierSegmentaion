"""
This module implements the patching functions.
"""
import numpy

#%% Extract patches & Reconstruct from patches

def extract_grayscale_patches( img, shape, offset=(0,0), stride=(1,1) ):
    """ Adopted from: http://jamesgregson.ca/extract-image-patches-in-python.html """
    
    """Extracts (typically) overlapping regular patches from a grayscale image

    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!

    Args:
        img (HxW ndarray): input image from which to extract patches

        shape (2-element arraylike): shape of that patches as (h,w)

        offset (2-element arraylike): offset of the initial point as (y,x)

        stride (2-element arraylike): vertical and horizontal strides

    Returns:
        patches (ndarray): output image patches as (N,shape[0],shape[1]) array

        origin (2-tuple): array of top and array of left coordinates
    """
    px, py = numpy.meshgrid( numpy.arange(shape[1]),numpy.arange(shape[0]))
    l, t = numpy.meshgrid(
        numpy.arange(offset[1],img.shape[1]-shape[1]+1,stride[1]),
        numpy.arange(offset[0],img.shape[0]-shape[0]+1,stride[0]))
    l = l.ravel()
    t = t.ravel()
    x = numpy.tile( px[None,:,:], (t.size,1,1)) + numpy.tile( l[:,None,None], (1,shape[0],shape[1]))
    y = numpy.tile( py[None,:,:], (t.size,1,1)) + numpy.tile( t[:,None,None], (1,shape[0],shape[1]))
    patches = img[y.ravel(),x.ravel()].reshape((t.size,shape[0],shape[1]))


    # right side tiles
    if img.shape[0]%stride[0] != 0:
        check = int((img.shape[0]-offset[0])/stride[0])
        for i in range(int((img.shape[0]-offset[0])/stride[0])):
            x_r, t_r = numpy.meshgrid(
                numpy.arange(img.shape[1] - shape[1],img.shape[1]),
                numpy.arange(i*stride[0]+offset[0],(i+1)*stride[0]+offset[0])
                )
            img_r = img[t_r.ravel(), x_r.ravel()].reshape((1, shape[0], shape[1]))
            patches = numpy.vstack((patches, img_r))


    # bottom side times
    if img.shape[1] % stride[1] != 0:
        check = int((img.shape[1] - offset[1]) / stride[1])
        for i in range(int((img.shape[1] - offset[1]) / stride[1])):
            x_b, t_b = numpy.meshgrid(
                numpy.arange(i * stride[1] + offset[1], (i + 1) * stride[1] + offset[1]),
                numpy.arange(img.shape[0] - shape[0], img.shape[0])
            )
            img_b = img[t_b.ravel(), x_b.ravel()].reshape((1, shape[0], shape[1]))
            patches = numpy.vstack((patches, img_b))

    # bottom corner tile
    if img.shape[0] % stride[0] != 0 or img.shape[1]%stride[1] != 0 :
        l_br, t_br = numpy.meshgrid(
            numpy.arange(img.shape[1] - shape[1], img.shape[1]),
            numpy.arange(img.shape[0] - shape[0], img.shape[0]))
        img_br = img[t_br.ravel(), l_br.ravel()].reshape((1, shape[0], shape[1]))
        patches = numpy.vstack((patches, img_br))


    return patches, (t,l)


def reconstruct_from_grayscale_patches( patches, origin, epsilon=1e-12 ):
    """ Adopted from: http://jamesgregson.ca/extract-image-patches-in-python.html """

    """Rebuild an image from a set of patches by averaging

    The reconstructed image will have different dimensions than the
    original image if the strides and offsets of the patches were changed
    from the defaults!

    Args:
        patches (ndarray): input patches as (N,patch_height,patch_width) array

        origin (2-tuple): top and left coordinates of each patch

        epsilon (scalar): regularization term for averaging when patches
            some image pixels are not covered by any patch

    Returns:
        image (ndarray): output image reconstructed from patches of
            size ( max(origin[0])+patches.shape[1], max(origin[1])+patches.shape[2])

        weight (ndarray): output weight matrix consisting of the count
            of patches covering each pixel
    """
    patch_width  = patches.shape[2]
    patch_height = patches.shape[1]
    img_width    = numpy.max( origin[1] ) + patch_width
    img_height   = numpy.max( origin[0] ) + patch_height

    out = numpy.zeros( (img_height,img_width) )
    wgt = numpy.zeros( (img_height,img_width) )
    for i in range(patch_height):
        for j in range(patch_width):
            out[origin[0]+i,origin[1]+j] += patches[:,i,j]
            wgt[origin[0]+i,origin[1]+j] += 1.0

    return out/numpy.maximum( wgt, epsilon ), wgt
