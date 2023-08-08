# ROI_previewUI

**This GUI allows you to preview hdf5 data to decide where to generate ROI training data for [ITAS3D](https://github.com/WeisiX/ITAS3D).**

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
7. Zoom in using the magnifying icon and click the **"crop"** button to crop to a specific ROI dimension.
8. Press **"Auto rescale"** to use the 2nd percentile and 98th percentile for contrast clipping calculated from current viewing ROI.
9. Click **"Save where?"** to specify a saving home path, and choose the appropriate antibody for target channel you are saving.
8. **Remember to Press "SAVE"** to export all parameters to a .csv file. There should be 12 parameters including x, y-coordinates of the top-left corner of the ROI.
9. Continue moving to different areas and press "SAVE" until you have enough ROIs.
10. Close the GUI when you are done, or press "Select file" to view another .h5 file.


## Instruction for ROI Generation

To generate ROI training data, follow these steps:

1. Run the Jupyter notebook: *generate_ROI_PGP.ipynb*

2. Choose between the **Manual** or **Auto** options:

	- **Manual**: Input all the 12 parameters yourself.
	- **Auto**: Change the name of the .csv file that contains all the parameters of ROIs. The parameters are saved by running *ROI_v2.py* (look at [Instruction for Previewing ROI](#instruction-for-previewing-roi))

The Jupyter notebook calls *CollectingImgStackFused.py,* which contains all the necessary functions for ROI generation.


**Note**: Ensure that you have the necessary dependencies installed to run the code successfully.
