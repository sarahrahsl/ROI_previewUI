import os
import numpy as np
from skimage.io import imread, imsave
from skimage.exposure import equalize_adapthist, rescale_intensity


################ Helper functions for false-coloring #############################

HE_settings = {'nuclei': [0.17, 0.27, 0.105], 'cyto': [0.05, 1.0, 0.54]}


def getBackgroundLevels(image, threshold=50):
    image_DS = np.sort(image, axis=None)
    foreground_vals = image_DS[np.where(image_DS > threshold)]
    hi_val = foreground_vals[int(np.round(len(foreground_vals)*0.95))]
    background = hi_val / 5

    return hi_val, background


def FC_rescale(image, ClipLow, ClipHigh):
    
    Img_rescale = rescale_intensity(np.clip(image, ClipLow, ClipHigh)
                                    ,out_range=(0,10000)
                                    )

    return Img_rescale


def rapidFieldDivision(image, flat_field):
    """Used for rapidFalseColoring() when flat field has been calculated."""
    output = np.divide(image, flat_field, where=(flat_field != 0))
    return output


def rapidPreProcess(image, background, norm_factor):
    """Background subtraction optimized for CPU."""
    tmp = image - background
    tmp[tmp < 0] = 0
    tmp = (tmp ** 0.85) * (255 / norm_factor)
    return tmp


def rapidGetRGBframe(nuclei, cyto, nuc_settings, cyto_settings, k_nuclei, k_cyto):
    """CPU-based exponential false coloring operation."""
    tmp = nuclei * nuc_settings * k_nuclei + cyto * cyto_settings * k_cyto
    return 255 * np.exp(-1 * tmp)


def rapidFalseColor(nuclei, cyto, nuc_settings, cyto_settings,
                    nuc_normfactor=3000, cyto_normfactor=8000,
                    run_FlatField_nuc=False, 
                    run_FlatField_cyto=False,
                    nuc_bg_threshold=50, 
                    cyto_bg_threshold=50):

    nuclei = np.ascontiguousarray(nuclei, dtype=float)
    cyto = np.ascontiguousarray(cyto, dtype=float)

    # Set multiplicative constants
    k_nuclei = 1.0
    k_cyto = 1.0

    # Run background subtraction or normalization for nuc and cyto
    if not run_FlatField_nuc:
        k_nuclei = 0.08
        nuc_background = getBackgroundLevels(nuclei, threshold=nuc_bg_threshold)[1]
        nuclei = rapidPreProcess(nuclei, nuc_background, nuc_normfactor)

    if not run_FlatField_cyto:
        k_cyto = 0.012
        cyto_background = getBackgroundLevels(cyto, threshold=cyto_bg_threshold)[1]
        cyto = rapidPreProcess(cyto, cyto_background, cyto_normfactor)

    output_global = np.zeros((3, nuclei.shape[0], nuclei.shape[1]), dtype=np.uint8)
    for i in range(3):
        output_global[i] = rapidGetRGBframe(nuclei, cyto, nuc_settings[i], cyto_settings[i], k_nuclei, k_cyto)

    RGB_image = np.moveaxis(output_global, 0, -1).astype(np.uint8)
    return RGB_image




################ Making "FC" folder, and blocks sub-folder #############################


def collectTrainingROI(FC_dir,
                       ch1_dir,
                       ch2_dir,
                       blockname,
                       xcoords, 
                       ycoords,
                       zcoords):


    x1, x2  = xcoords[0], xcoords[1]
    y1, y2  = ycoords[0], ycoords[1]       
    zlevels = np.arange(zcoords[0], zcoords[1], 1)   

    # Create the block subfolder names and the jpeg file names for the false-colored training ROIs

    blockdirFC = os.path.join(FC_dir,
                              '%s_Xpos_%s_%s_Ypos_%s_%s_stack_%s_%s' %
                              (blockname,
                              '{:0>6d}'.format(x1),
                              '{:0>6d}'.format(x2),
                              '{:0>6d}'.format(y1),
                              '{:0>6d}'.format(y2),
                              '{:0>6d}'.format(zcoords[0]),
                              '{:0>6d}'.format(zcoords[1]))
                              )

    blockdirFC_T = blockdirFC + "_transpose"
    blockdirFC_M = blockdirFC + "_mirror"
    blockdirFC_F = blockdirFC + "_flip"

    if not os.path.exists(blockdirFC):
        print(blockdirFC)
        os.mkdir(blockdirFC)
        os.mkdir(blockdirFC_T)
        os.mkdir(blockdirFC_M)
        os.mkdir(blockdirFC_F)

    FCflists = [blockdirFC + os.sep + '%s_FC_pos%s%s_pos%s%s_' %
                                      (blockname,
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                        '{:0>6d}.jpeg'.format(z) for z in zlevels]

    FCTlists = [blockdirFC_T + os.sep + '%s_FC_pos%s%s_pos%s%s_transpose_' %
                                      (blockname,
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                        '{:0>6d}.jpeg'.format(z) for z in zlevels]

    FCMlists = [blockdirFC_M + os.sep + '%s_FC_pos%s%s_pos%s%s_mirror_' %
                                      (blockname,
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                        '{:0>6d}.jpeg'.format(z) for z in zlevels]

    FCFflists = [blockdirFC_F + os.sep + '%s_FC_pos%s%s_pos%s%s_mirror_' %
                                      (blockname,
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                        '{:0>6d}.jpeg'.format(z) for z in zlevels]

    chans = [(ch1_dir, 's01'),
             (ch2_dir, 's00')]
    
    flists  = []
    Tflists = []
    Mflists = []
    Fflists = []
    
    # Iterate through channels

    for ch in chans:
    
        blockdir = os.path.join(ch[0],
                                '%s_Xpos_%s_%s_Ypos_%s_%s_stack_%s_%s' %
                                (blockname,
                                '{:0>6d}'.format(x1),
                                '{:0>6d}'.format(x2),
                                '{:0>6d}'.format(y1),
                                '{:0>6d}'.format(y2),
                                '{:0>6d}'.format(zcoords[0]),
                                '{:0>6d}'.format(zcoords[1]))
                                )

        blockdir_T = blockdir + "_transpose"
        blockdir_M = blockdir + "_mirror"
        blockdir_F = blockdir + "_flip"

        flist = [blockdir + os.sep + '%s_%s_pos%s%s_pos%s%s_' %
                                (blockname,
                                ch[1],
                                x1,
                                x2,
                                y1,
                                y2) +
                '{:0>6d}.jpeg'.format(z) for z in zlevels]

        flists.append(flist)

        Tflist = [blockdir_T + os.sep + '%s_%s_pos%s%s_pos%s%s_transpose_' %
                                (blockname,
                                ch[1],
                                x1,
                                x2,
                                y1,
                                y2) +
                '{:0>6d}.jpeg'.format(z) for z in zlevels]

        Tflists.append(Tflist)

        Mflist = [blockdir_M + os.sep + '%s_%s_pos%s%s_pos%s%s_mirror_' %
                                (blockname,
                                ch[1],
                                x1,
                                x2,
                                y1,
                                y2) +
                '{:0>6d}.jpeg'.format(z) for z in zlevels]

        Mflists.append(Mflist)

        Fflist = [blockdir_F + os.sep + '%s_%s_pos%s%s_pos%s%s_flip_' %
                                (blockname,
                                ch[1],
                                x1,
                                x2,
                                y1,
                                y2) +
                '{:0>6d}.jpeg'.format(z) for z in zlevels]

        Fflists.append(Fflist)


    # Read in JPEG images and do false-color
    ROI_blocks = [flists,   Tflists,  Mflists,  Fflists]
    FC_blocks  = [FCflists, FCTlists, FCMlists, FCFflists]

    for ROI_block, FC_block in zip(ROI_blocks, FC_blocks):
        for i in range(len(ROI_block[0])):
            nuc  = imread(ROI_block[0][i])
            cyto = imread(ROI_block[1][i])
            nuc =  FC_rescale(nuc,  1, 10000)
            cyto = FC_rescale(cyto, 1, 10000)

            pseudoHE = rapidFalseColor(nuc, cyto, HE_settings['nuclei'], HE_settings['cyto'])
            imsave(FC_block[i], pseudoHE)






    