# ROI_previewUI
## Previewing .h5 file to decide where to generate ROI from

**Run the Jupyter notebook :  *generate_ROI_PGP.ipynb***
- Manual or Auto options:
	- **Manual**:  you need to find the coordinates yourself. (using ROI_v2.py or bigstitcher)
	- **Auto**: you need to run the ROI_v2.py file and record all the necessary params into a .csv file.

The Jupyter notebook calls CollectingImgStackFused.py, which contains all neccessary function, from generating "train" folder, folders containing ROIs, and cropping and contrast adjusting ROIs. 

**To use the Previewing UI, *ROI_v2.py*** 
1. Run python ROI_v2.py 
2. Select the .h5 file (fused) you want to make ROI out of from the prompt window.
3. Scroll on the image to view different z-levels.
4. Change to different channel to view staining quality.
5. Change the aggressiveness of contrast clipping through pressing up/down arrow on the clip values boxes.
6. Only target channel allows using CLAHE as contrast enhancement method.
7. Zoom in using the magnifying icon, and click "home" button to go back and view the entire image.
8. When you are ready, press "Save params" to save all the settings to .csv file. The x, y-coords of the top left corner of the ROI will be saved, 160p x 160p (3x downsample, 640p x 640p equivalent)
9. Keep going to different places and press "Save Param" until it's enough.
10. Close the GUI when you are done, or press select file to view another .h5 file.


