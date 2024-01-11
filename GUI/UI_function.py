import numpy as np
from skimage.exposure import equalize_adapthist, rescale_intensity









######################### False coloring helper functions ###############################

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


############################ Plot helper function ########################################

def Rescale(current_slice, 
            ClipLowLim, 
            ClipHighLim):

    current_slice = rescale_intensity(np.clip(current_slice,
                                              ClipLowLim,
                                              ClipHighLim),
                                              out_range='uint8')
    return current_slice