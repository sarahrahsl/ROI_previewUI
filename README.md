# ROI_previewUI

**This GUI allows you to preview hdf5 data to decide where to generate ROI training data for [ITAS3D](https://github.com/WeisiX/ITAS3D).**


## Contents
- [System requirements](#system-requirements)
- [Installation](#installation)
- [GUI guide](#instruction-for-previewing-gui-to-export-csv-file-for-roi-generation)
- [Generating Training ROIs](#ipynb-to-generate-training-rois)
- [False-coloring Training ROIs](#ipynb-to-false-color-training-rois)

## System requirements
This GUI is written in Python. Users should install Anaconda (tested on Conda 4.12.0, Python 3.9.13, Window 10)

In addition, the following packages are required, many of which come with Python/Anaconda. This code has been tested with the version number indicated, though other versions may also work.
  - pandas=1.4.4
  - matplotlib=3.5.2
  - pyqt=5.15.7
  - h5py=2.10.0

## Installation
After installing the required packages above, run the following command in Anaconda to clone this repository:
```bash
git clone https://github.com/sarahrahsl/ROI_previewUI.git
```
To set up the environment, you can run the following: 

```
conda env create -f environment.yml
```

Alternatively, you can manually install the python library packages specified in [System requirements](#system-requirements)


## Instruction for Previewing GUI to export CSV file for ROI generation

The exported CSV file is very useful for subsequent training ROI generation and training ROI false-color, make sure you do this step first. 

1. Run the following command to execute the Previewing UI:

	```
	cd GUI/
	python ROI_v2.py
	```

2. Select the .h5 file (fused) from the prompt window for which you want to create ROIs.
3. Scroll on the image to view different z-levels.
4. Change to different channels (key a, s, d)
5. Adjust the aggressiveness of contrast clipping by pressing the up/down arrow on the clip values boxes.
7. Zoom in using the magnifying icon (or key z) and click the **"crop"** button (or key c) to crop to a specific ROI dimension.
8. Press **"Auto rescale"** (or key r) to use the 2nd percentile and 98th percentile for contrast clipping calculated from current viewing ROI.
9. Click **"Train root"** to specify the directory for your "train" folder, i.e., the antibody for target channel for training ROI generation.
8. Press **"SAVE"** to export all parameters to a **CSV file**. There should be 12 parameters including x, y-coordinates of the top-left corner of the ROI. 
9. Continue moving to different areas and press "SAVE" until you have enough ROIs.
10. Close the GUI when you are done, or press "HDF5 file" to view another .h5 file.


## IPYNB to generate training ROIs

To generate ROI training data, make sure you have the CSV file ready, follow these steps:

- Use ***scripts/generate_training_ROI.ipynb***

- Change the name of the CSV file that contains all the parameters of ROIs. 

The Jupyter notebook calls *CollectingImgStackFused.py,* which contains all the necessary functions for ROI generation.




## IPYNB to false-color training ROIs

- Use ***scripts/generate_FC_ROI.ipynb***
- Change the name of the CSV file that contains all the parameters of the ROIs. 



**Note**: Ensure that you have the necessary dependencies installed to run the code successfully.