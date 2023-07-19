# ROI_previewUI

**This UI allows you to preview hdf5 data to decide where to generate ROI training data for [ITAS3D](https://github.com/WeisiX/ITAS3D).**

## Instruction for ROI Generation

To generate ROI training data, follow these steps:

1. Run the Jupyter notebook: *generate_ROI_PGP.ipynb*

2. Choose between the **Manual** or **Auto** options:

	- **Manual**: You need to input all the 10 parameters yourself.
	- **Auto**: Run the *ROI_v2.py* file and record all the necessary parameters into a .csv file. (look at [Instruction for Previewing ROI](#instruction-for-previewing-roi))

The Jupyter notebook calls *CollectingImgStackFused.py,* which contains all the necessary functions for ROI generation.

## Instruction for Previewing ROI

1. Run the following command to execute the Previewing UI:

	```
	python ROI_v2.py
	```

2. Select the .h5 file (fused) from the prompt window for which you want to create ROIs.
3. Scroll on the image to view different z-levels.
4. Change to different channels to view staining quality.
5. Adjust the aggressiveness of contrast clipping by pressing the up/down arrow on the clip values boxes.
6. Only the target channel allows using CLAHE as a contrast enhancement method.
7. Zoom in using the magnifying icon and click the "home" button to view the entire image again.
8. When you are ready, press "Save params" to save all the settings to a .csv file. The x, y-coordinates of the top-left corner of the ROI will be saved, with dimensions of 160p x 160p (3x downsample, 640p x 640p equivalent).
9. Continue moving to different areas and press "Save Param" until you have enough ROIs.
10. Close the GUI when you are done, or press "Select file" to view another .h5 file.

**Note**: Ensure that you have the necessary dependencies installed to run the code successfully.
