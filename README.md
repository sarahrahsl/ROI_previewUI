# ROI_previewUI
Previewing .h5 file to decide where to generate ROI from

Run the Jupyter notebook :  generate_ROI_PGP.ipynb
- manual and auto options
	- manual option, you need to find the coordinates in bigstitchers
	- auto option, you need to run the ROI_v2.py file

CollectingImgStackFused.py : 
	- contains all neccessary functions

ROI_v2.py : 
	- select the .h5 file you want to make ROI out of from the prompt window.
	- scroll on the image to change z-level.
	- change to different channel to view the staining.
	- change the aggressiveness of contrast enhancement through pressing up/down arrow on 	  the clip values boxes.
	- only target channel allows using CLAHE as contrast enhancement method.
	- zoom in w/ the magnifying icon, and click "home" button to view the entire image.
	- when you are ready, press "Save" to save all the settings to .csv file, the coords 	  saved are the top left corner, 160p x 160p (3x downsample, 640p x 640p equivalent)
	- keep going to different place and press "Save Param" until it's enough
	- close window or press select file for another .h5 file


