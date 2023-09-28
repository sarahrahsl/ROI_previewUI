from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QDoubleSpinBox, QGridLayout, QWidget, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QMessageBox, QSizePolicy, QComboBox, QToolButton,
    QSpacerItem, QHBoxLayout, QVBoxLayout, QGroupBox, QLineEdit, QFormLayout, QFileDialog
)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QEvent
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from skimage.exposure import equalize_adapthist, rescale_intensity
import time
import csv
import datetime
import os

"""

Sarah, Jul 2023
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        #######################################################################################################
        ############################################## Layout #################################################

        self.setWindowTitle("ROI Preview")

        # Create layout for the UI
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Instruction Note
        note_label = QLabel("Quick instruction: \n\
        1. Change to different channels(Key A, key S). \n\
        2. Zoom(key Z) to desired ROI, input ROI dimension, press 'Crop'(Key C), drag(Key E) to fine tune ROI. \n\
        3. Find manually or press 'Auto Rescale'(key R) for optimal contrast clipping values. \n\
        4. Press 'Save!' to save coords to .csv file, 'home'(key F) to go back or 'Select File' for another data. \n\
        5. Only use falsecolor function after you crop it. \n \
You are viewing the 8x downsampled of the fused data. ")

        # Loading data label
        self.loading_label = QLabel()
        note_layout = QHBoxLayout()
        note_layout.addWidget(note_label, alignment=Qt.AlignTop | Qt.AlignLeft)
        note_layout.addWidget(self.loading_label, alignment=Qt.AlignBottom | Qt.AlignRight)

        # Create a matplotlib figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_release_event', self.on_zoom_completed)

        # Bottom pannel of the canvas
        Canvasbottom = QHBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)        
        Canvasbottom.addWidget(self.toolbar)

        ############ Add button to change between channels ###########
        button_layout = QHBoxLayout()
        self.cyto_button = QPushButton("Cyto")
        self.cyto_button.setCheckable(True)
        self.cyto_button.clicked.connect(self.update_img2cyto)
        button_layout.addWidget(self.cyto_button)
        self.nuc_button = QPushButton("Nuc")
        self.nuc_button.setCheckable(True)
        self.nuc_button.clicked.connect(self.update_img2nuc)
        button_layout.addWidget(self.nuc_button)
        self.FC_button = QPushButton("False Color")
        self.FC_button.setCheckable(True)
        self.FC_button.clicked.connect(self.update_img2FC)
        button_layout.addWidget(self.FC_button)


        button_container = QGroupBox("Channels")
        button_container.setLayout(button_layout)
        self.setCentralWidget(button_container)

        ############# Add x,y,z coordinate display #################
        self.x_coordinate_textbox = QLineEdit()
        self.x_coordinate_textbox.setReadOnly(True)
        self.x_coordinate_textbox.setStyleSheet("background-color: #f0f0f0;")
        self.y_coordinate_textbox = QLineEdit()
        self.y_coordinate_textbox.setReadOnly(True)
        self.y_coordinate_textbox.setStyleSheet("background-color: #f0f0f0;")
        self.arrayshape_textbox = QLineEdit()
        self.arrayshape_textbox.setReadOnly(True)
        self.arrayshape_textbox.setStyleSheet("background-color: #f0f0f0;")
        self.current_z_level_textbox = QLineEdit()
        self.current_z_level_textbox.setText("0") 
        self.current_z_level_textbox.returnPressed.connect(self.update_z_level_textbox)
        # self.current_z_level_textbox.setReadOnly(True)
        self.note_textbox = QLineEdit()
        form_layout = QFormLayout()
        form_layout.addRow("x-coordinate:", self.x_coordinate_textbox)
        form_layout.addRow("y-coordinate:", self.y_coordinate_textbox)
        form_layout.addRow("Current Z-Level:", self.current_z_level_textbox)
        form_layout.addRow("Vol Dim [z,y,x]:", self.arrayshape_textbox)
        form_layout.addRow("Note:", self.note_textbox)
        coordinates_container = QGroupBox("Coordinates [3x downsampled]")
        coordinates_container.setLayout(form_layout)

        ############ Add ROI dimension textbox, Crop button, and Auto Rescale button ################
        self.roi_dim_textbox = QLineEdit()
        self.roi_dim_textbox.setText("512")  # Set default value to 512
        self.roi_dim_textbox.textChanged.connect(self.ROI_dim_changed)
        roi_dim_label = QLabel("ROI dim (1x downsampled data)")
        self.crop_button = QPushButton("Crop")
        self.crop_button.setCheckable(True)
        self.crop_button.clicked.connect(self.Crop_ROI)
        auto_rescale_button = QPushButton("Auto Rescale")
        auto_rescale_button.clicked.connect(self.Auto_Rescale)
        roi_dim_layout = QHBoxLayout()
        roi_dim_layout.addWidget(roi_dim_label)
        roi_dim_layout.addWidget(self.roi_dim_textbox) 
        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(self.crop_button)
        button_layout2.addWidget(auto_rescale_button)

        button_group = QGroupBox("ROI Visualization")
        button_group_layout = QVBoxLayout()
        button_group_layout.addLayout(roi_dim_layout)
        button_group_layout.addLayout(button_layout2)
        button_group.setLayout(button_group_layout)

        ########### Add Contrast enhancement for cyto channel #################
        ClipLow_Cyto_default = 0
        ClipHigh_Cyto_default = 5000
        dropdown_layout1 = QVBoxLayout()
        dropdown_layout1.addWidget(QLabel("Contrast Enhancement Method:"))
        self.dropdown1 = QComboBox()
        self.dropdown1.addItem("Rescale")
        dropdown_layout1.addWidget(self.dropdown1)
        clip_high_layout1 = QHBoxLayout()
        clip_high_layout1.addWidget(QLabel("Clip High:"))
        self.ClipHighLim_cyto = QDoubleSpinBox()
        self.ClipHighLim_cyto.setRange(0, 30000)
        self.ClipHighLim_cyto.setSingleStep(50)
        self.ClipHighLim_cyto.setValue(ClipHigh_Cyto_default)
        clip_high_layout1.addWidget(self.ClipHighLim_cyto)
        clip_low_layout1 = QHBoxLayout()
        clip_low_layout1.addWidget(QLabel("Clip Low:"))
        self.ClipLowLim_cyto = QDoubleSpinBox()
        self.ClipLowLim_cyto.setRange(0, 2000)
        self.ClipLowLim_cyto.setSingleStep(50)
        self.ClipLowLim_cyto.setValue(ClipLow_Cyto_default)
        self.ClipHighLim_cyto.valueChanged.connect(self.cyto_clip_higher_change)
        self.ClipLowLim_cyto.valueChanged.connect(self.cyto_clip_lower_change)

        clip_low_layout1.addWidget(self.ClipLowLim_cyto)
        dropdown_layout1.addLayout(clip_high_layout1)
        dropdown_layout1.addLayout(clip_low_layout1)
        dropdown_container1 = QGroupBox("Cyto")
        dropdown_container1.setLayout(dropdown_layout1)

        ############ Add Contrast enhancement for Nuc channel ###############
        ClipLow_Nuc_default = 0
        ClipHigh_Nuc_default = 10000
        dropdown_layout2 = QVBoxLayout()
        dropdown_layout2.addWidget(QLabel("Contrast Enhancement Method:"))
        self.dropdown2 = QComboBox()
        self.dropdown2.addItem("Rescale")
        dropdown_layout2.addWidget(self.dropdown2)
        clip_high_layout2 = QHBoxLayout()
        clip_high_layout2.addWidget(QLabel("Clip High:"))
        self.ClipHighLim_nuc = QDoubleSpinBox()
        self.ClipHighLim_nuc.setRange(0, 30000)
        self.ClipHighLim_nuc.setSingleStep(50)
        self.ClipHighLim_nuc.setValue(ClipHigh_Nuc_default)
        clip_high_layout2.addWidget(self.ClipHighLim_nuc)
        clip_low_layout2 = QHBoxLayout()
        clip_low_layout2.addWidget(QLabel("Clip Low:"))
        self.ClipLowLim_nuc = QDoubleSpinBox()
        self.ClipLowLim_nuc.setRange(0, 2000)
        self.ClipLowLim_nuc.setSingleStep(50)
        self.ClipLowLim_nuc.setValue(ClipLow_Nuc_default) 
        self.ClipHighLim_nuc.valueChanged.connect(self.nuc_clip_higher_change)
        self.ClipLowLim_nuc.valueChanged.connect(self.nuc_clip_lower_change)

        clip_low_layout2.addWidget(self.ClipLowLim_nuc)
        dropdown_layout2.addLayout(clip_high_layout2)
        dropdown_layout2.addLayout(clip_low_layout2)
        dropdown_container2 = QGroupBox("Nuc")
        dropdown_container2.setLayout(dropdown_layout2)

        ############## Add normalfactor1 and normalfactor2 for false-coloring ################
        dropdown_layout3 = QVBoxLayout()
        dropdown_layout3.addWidget(QLabel("FC style:"))
        self.dropdown3 = QComboBox()
        self.dropdown3.addItem("H&E")
        self.dropdown3.addItem("IHC")
        dropdown_layout3.addWidget(self.dropdown3)
        Nuc_normafactor_layout = QHBoxLayout()
        Nuc_normafactor_layout.addWidget(QLabel("Nuc normfactor:"))
        self.Nuc_normafactor = QDoubleSpinBox()
        self.Nuc_normafactor.setRange(0, 15000)
        self.Nuc_normafactor.setSingleStep(500)
        self.Nuc_normafactor.setValue(5000)
        Nuc_normafactor_layout.addWidget(self.Nuc_normafactor)
        Cyto_normfactor_layout = QHBoxLayout()
        Cyto_normfactor_layout.addWidget(QLabel("Cyto normfactor:"))
        self.Cyto_normfactor = QDoubleSpinBox()
        self.Cyto_normfactor.setRange(0, 15000)
        self.Cyto_normfactor.setSingleStep(500)
        self.Cyto_normfactor.setValue(8000) 
        self.Cyto_normfactor.valueChanged.connect(self.normfactor_cyto_change)
        self.Nuc_normafactor.valueChanged.connect(self.normfactor_nuc_change)

        Cyto_normfactor_layout.addWidget(self.Cyto_normfactor)
        dropdown_layout3.addLayout(Nuc_normafactor_layout)
        dropdown_layout3.addLayout(Cyto_normfactor_layout)
        dropdown_container3 = QGroupBox("False-coloring")
        dropdown_container3.setLayout(dropdown_layout3)

        ############# Add Action buttons at bottom right corner ################
        file_button = QPushButton("HDF5 File")
        file_button.clicked.connect(self.select_file) 
        define_savehome = QPushButton("Save where?")
        define_savehome.clicked.connect(self.select_savehome) 

        # Override the behavior of the "Reset Original View" button
        home_button = self.toolbar.actions()[0] #String 0 = "Home" button
        home_button.triggered.connect(self.go_home)

        save_button = QPushButton(" SAVE ")
        save_button.setStyleSheet("QPushButton { background-color: green; color: white; padding: 5px; font-weight: bold; }")
        save_button.clicked.connect(self.save_coords)

        button_layout = QHBoxLayout()
        button_layout.addWidget(file_button)
        button_layout.addSpacing(30)
        button_layout.addWidget(define_savehome)
        button_layout.addWidget(save_button)
        button_layout.addStretch()

        # Configure left layout and right layout and make it central
        left_layout.addLayout(note_layout)
        left_layout.addWidget(self.canvas, stretch=1)
        left_layout.addLayout(Canvasbottom)
        left_layout.addWidget(button_container)
        right_layout.addWidget(coordinates_container)
        right_layout.addWidget(button_group)
        right_layout.addWidget(dropdown_container1)
        right_layout.addWidget(dropdown_container2)
        right_layout.addWidget(dropdown_container3)
        right_layout.addLayout(button_layout)
        main_layout.addLayout(left_layout, stretch=80)
        main_layout.addLayout(right_layout,stretch=20)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.cid = None

        #################################### End of Layout #######################################################
        ##########################################################################################################

        # Initialization and initial values
        self.save_home = os.getcwd()
        self.ROI_dim = 512
        self.normfactor_nuc = 8000
        self.normfactor_cyto = 5000
        self.select_file() # Including readHDF5() and plot_init_z()

        # Connect the mouse wheel event to the update_z_level method
        self.canvas.mpl_connect('scroll_event', self.update_z_level)

    ##################### Initialization functions  ###########################################
    
        """
        The following will only be run every time you select a new sample
        """

    def readHDF5(self):
    
        start = time.time()
        with h5.File(self.h5path, 'r') as f:
            self.cyto = f['t00000']['s01']['3/cells'][:, :, :].astype(np.uint16)
            self.nuc = f['t00000']['s00']['3/cells'][:, :, :].astype(np.uint16)
        f.close()

        if self.cyto.shape[0] < self.cyto.shape[1]:
            self.orient = 0
        else: 
            self.orient = 1
            self.cyto = np.moveaxis(self.cyto, 0, 1)
            self.nuc = np.moveaxis(self.nuc, 0, 1)

        print(time.time() - start, "s")

    def plot_init_z(self):

        # Initial params
        self.img = self.cyto
        self.current_chan = "cyto"
        self.vmax = 5000
        self.ClipLowLim = 0
        self.ClipHighLim = 5000

        # Display current z-level and vol shape
        self.shape = self.img.shape
        self.current_z_level = 0
        self.current_z_level_textbox.setText(str(self.current_z_level))
        self.arrayshape_textbox.setText(str(self.shape))

        # Plot initial slice
        ax = self.figure.add_subplot(111)
        ax.imshow(self.img[self.current_z_level], cmap="gray", vmax=self.vmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vol Preview')
        self.canvas.draw()
        ax = self.figure.gca()
        self.x_limits = ax.get_xlim()
        self.y_limits = ax.get_ylim()

        self.cyto_button.setChecked(True)
        self.nuc_button.setChecked(False)  
        self.dropdown1.setEnabled(True)
        self.dropdown2.setEnabled(False)  
        self.dropdown3.setEnabled(False) 
        self.ClipHighLim_cyto.setEnabled(True) 
        self.ClipLowLim_cyto.setEnabled(True)
        self.ClipHighLim_nuc.setEnabled(False) 
        self.ClipLowLim_nuc.setEnabled(False)  
        self.Nuc_normafactor.setEnabled(False)
        self.Cyto_normfactor.setEnabled(False)
        self.go_home()


    ########################## Zoom In/Out, Update xy coordinate ###############################

    def on_zoom_completed(self, event):
        if event.name == 'button_release_event':
            self.figure.canvas.mpl_disconnect(self.cid)
            self.cid = self.figure.canvas.mpl_connect('draw_event', self.on_draw_completed)

    def on_draw_completed(self, event):
        self.figure.canvas.mpl_disconnect(self.cid)
        ax = self.figure.gca()
        if ax:
            self.x_limits = ax.get_xlim()
            self.y_limits = ax.get_ylim()

            self.x_coordinate_textbox.setText(str(format(self.x_limits[0],".3f")))
            self.y_coordinate_textbox.setText(str(format(self.y_limits[1],".3f")))

            self.x_coordinate_textbox.update()
            self.y_coordinate_textbox.update()

            if self.x_limits[0] + int(self.ROI_dim)/4 > self.shape[2] - 1 or \
               self.y_limits[1] + int(self.ROI_dim)/4 > self.shape[1] - 1 :
                self.show_OutofBound()
            else: 
                self.hide_text()


    ########################## Update current Z-level ##########################################

    def update_z_level(self, event):

        if event.button == 'up' or event.button == 'down':

            current_z_level = self.current_z_level
            # Update the z-level based on the scroll direction
            if event.button == 'up':
                current_z_level += 1
                if current_z_level >= self.shape[0]: # deepest level limit
                    current_z_level = self.shape[0] - 1
            else:
                current_z_level -= 1
                if current_z_level < 0: # surface level limit
                    current_z_level = 0

            self.current_z_level = current_z_level
            self.current_z_level_textbox.setText(str(self.current_z_level))

            if current_z_level > self.shape[0] - 13: # out of bound error
                self.show_OutofBound()
            else: 
                self.hide_text()

            self.plot_slice()

    def update_z_level_textbox(self):

        current_z = int(self.current_z_level_textbox.text())

        if  current_z<0 or current_z> self.shape[0]-1:
            self.current_z_level_textbox.setText(str(self.current_z_level))
        else:
            self.current_z_level = current_z

            if self.current_z_level > self.shape[0] - 13: # out of bound error
                self.show_OutofBound()
            else: 
                self.hide_text()

            self.plot_slice()


    ########################## Update Channel ##########################################

    def update_img2cyto(self):
        self.img = self.cyto
        self.current_chan = "cyto"
        self.ClipLowLim = self.ClipLowLim_cyto.value()
        self.ClipHighLim = self.ClipHighLim_cyto.value()
        self.cyto_button.setChecked(True)
        self.nuc_button.setChecked(False)  
        self.FC_button.setChecked(False)
        self.dropdown1.setEnabled(True)
        self.dropdown2.setEnabled(False)   
        self.dropdown3.setEnabled(False)
        self.ClipHighLim_cyto.setEnabled(True)  
        self.ClipLowLim_cyto.setEnabled(True)
        self.ClipHighLim_nuc.setEnabled(False)  
        self.ClipLowLim_nuc.setEnabled(False) 
        self.Nuc_normafactor.setEnabled(False)
        self.Cyto_normfactor.setEnabled(False)        

        self.plot_slice()

    def update_img2nuc(self):
        self.img = self.nuc
        self.current_chan = "nuc"
        self.ClipLowLim = self.ClipLowLim_nuc.value()
        self.ClipHighLim = self.ClipHighLim_nuc.value()
        self.nuc_button.setChecked(True)
        self.cyto_button.setChecked(False)  
        self.FC_button.setChecked(False)
        self.dropdown1.setEnabled(False)
        self.dropdown2.setEnabled(True)
        self.dropdown3.setEnabled(False)
        self.ClipHighLim_nuc.setEnabled(True)  
        self.ClipLowLim_nuc.setEnabled(True)
        self.ClipHighLim_cyto.setEnabled(False)  
        self.ClipLowLim_cyto.setEnabled(False) 
        self.Nuc_normafactor.setEnabled(False)
        self.Cyto_normfactor.setEnabled(False)

        self.plot_slice()

    def update_img2FC(self):
        self.FC_button.setChecked(True)
        self.nuc_button.setChecked(False)
        self.cyto_button.setChecked(False)  
        self.dropdown1.setEnabled(False)
        self.dropdown2.setEnabled(False)
        self.dropdown3.setEnabled(True)
        self.ClipHighLim_nuc.setEnabled(False)  
        self.ClipLowLim_nuc.setEnabled(False)
        self.ClipHighLim_cyto.setEnabled(False)  
        self.ClipLowLim_cyto.setEnabled(False) 
        self.Nuc_normafactor.setEnabled(True)
        self.Cyto_normfactor.setEnabled(True)

        self.cyto_fc, self.nuc_fc = self.readHDF5_FC()
        self.Draw_FC()


    #################################### False-coloring ####################################

    def readHDF5_FC(self):
        ystart   = int(self.x_limits[0]*4)
        xstart   = int(self.y_limits[1]*4)
        zstart = int(self.current_z_level*4)
        ROI_dim = int(self.ROI_dim)
        xend = xstart + ROI_dim
        yend = ystart + ROI_dim
        zend = zstart + 1

        if self.orient == 1: 
            with h5.File(self.h5path, 'r') as f:
                cyto_fc = f['t00000']['s01']['1/cells'][xstart:xend, zstart:zend, ystart:yend].astype(np.uint16)
                nuc_fc = f['t00000']['s00']['1/cells'][xstart:xend, zstart:zend, ystart:yend].astype(np.uint16)
            f.close()
            cyto_fc = np.moveaxis(cyto_fc, 0, 1)
            nuc_fc = np.moveaxis(nuc_fc, 0, 1)
            print(cyto_fc.shape)
        else:
            with h5.File(self.h5path, 'r') as f:
                cyto_fc = f['t00000']['s01']['1/cells'][zstart:zend, xstart:xend, ystart:yend].astype(np.uint16)
                nuc_fc = f['t00000']['s00']['1/cells'][zstart:zend, xstart:xend, ystart:yend].astype(np.uint16)
            f.close()

        cyto_fc = self.FC_rescale(cyto_fc, self.ClipLowLim_cyto.value(), self.ClipHighLim_cyto.value())
        nuc_fc  = self.FC_rescale(nuc_fc, self.ClipLowLim_nuc.value(), self.ClipHighLim_nuc.value())
  
        return cyto_fc, nuc_fc

    def getBackgroundLevels(self, image, threshold=50):
        image_DS = np.sort(image, axis=None)
        foreground_vals = image_DS[np.where(image_DS > threshold)]
        hi_val = foreground_vals[int(np.round(len(foreground_vals)*0.95))]
        background = hi_val / 5

        return hi_val, background

    def FC_rescale(self, image, ClipLow, ClipHigh):
        
        Img_rescale = rescale_intensity(np.clip(image, ClipLow, ClipHigh)
                                        ,out_range=(0,10000)
                                        )

        return Img_rescale
    
    def rapidFieldDivision(self, image, flat_field):
        """Used for rapidFalseColoring() when flat field has been calculated."""
        output = np.divide(image, flat_field, where=(flat_field != 0))
        return output

    def rapidPreProcess(self, image, background, norm_factor):
        """Background subtraction optimized for CPU."""
        tmp = image - background
        tmp[tmp < 0] = 0
        tmp = (tmp ** 0.85) * (255 / norm_factor)
        return tmp

    def rapidGetRGBframe(self, nuclei, cyto, nuc_settings, cyto_settings, k_nuclei, k_cyto):
        """CPU-based exponential false coloring operation."""
        tmp = nuclei * nuc_settings * k_nuclei + cyto * cyto_settings * k_cyto
        return 255 * np.exp(-1 * tmp)

    def rapidFalseColor(self, nuclei, cyto, nuc_settings, cyto_settings,
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
            nuc_background = self.getBackgroundLevels(nuclei, threshold=nuc_bg_threshold)[1]
            nuclei = self.rapidPreProcess(nuclei, nuc_background, nuc_normfactor)

        if not run_FlatField_cyto:
            k_cyto = 0.012
            cyto_background = self.getBackgroundLevels(cyto, threshold=cyto_bg_threshold)[1]
            cyto = self.rapidPreProcess(cyto, cyto_background, cyto_normfactor)

        output_global = np.zeros((3, nuclei.shape[0], nuclei.shape[1]), dtype=np.uint8)
        for i in range(3):
            output_global[i] = self.rapidGetRGBframe(nuclei, cyto, nuc_settings[i], cyto_settings[i], k_nuclei, k_cyto)

        RGB_image = np.moveaxis(output_global, 0, -1).astype(np.uint8)
        return RGB_image

    ##### Actual FC function #########
    def RunFC_HE(self):
        """Nested in self.Draw_FC() """
        HE_settings = {'nuclei': [0.17, 0.27, 0.105], 'cyto': [0.05, 1.0, 0.54]}
        pseudoHE = self.rapidFalseColor(self.nuc_fc[0], self.cyto_fc[0], 
                                        HE_settings['nuclei'], HE_settings['cyto'],
                                        nuc_normfactor = self.normfactor_nuc, 
                                        cyto_normfactor = self.normfactor_cyto)
        return pseudoHE

    def Draw_FC(self):
        self.figure.clear()

        pseudoFC = self.RunFC_HE()

        vmax = np.percentile(pseudoFC,99)
        ax = self.figure.add_subplot(111)
        ax.imshow(pseudoFC, cmap='viridis', vmax=vmax)
        ax.axis('off')

        self.canvas.draw()

    def normfactor_nuc_change(self):
        self.normfactor_nuc = self.Nuc_normafactor.value()
        self.Draw_FC()

    def normfactor_cyto_change(self):
        self.normfactor_cyto = self.Cyto_normfactor.value()
        self.Draw_FC()


    ######################### Cropping ROI and auto rescale ################################

    def ROI_dim_changed(self):
        self.ROI_dim = self.roi_dim_textbox.text()

    def Crop_ROI(self):
        ROI_dim = float(self.ROI_dim)
        xstart = self.x_limits[0]
        ystart = self.y_limits[1]
        xend = xstart + ROI_dim/4
        yend = ystart + ROI_dim/4
        self.x_limits = [xstart, xend]
        self.y_limits = [yend, ystart]
        self.crop_button.setChecked(True)
        self.plot_slice()

    def Auto_Rescale(self):
        """
        This function calculate the p2 and p98 and update them.
        Plot will automatically update when values of the properties are changed
        """
        current_slice = self.img[self.current_z_level, :, :]
        xstart = int(self.x_limits[0])
        xend = int(self.x_limits[1])
        ystart = int(self.y_limits[1])
        yend = int(self.y_limits[0])
        if xstart >= 0 and xend <= self.shape[2] and ystart >= 0 and yend <= self.shape[1]:
            current_ROI = current_slice[ystart:yend,xstart:xend]
            p2, p98 = np.percentile(current_ROI, (2,99))
            p98 = p98*1.25
            if self.current_chan == "cyto":
                self.ClipHighLim_cyto.setValue(int(p98)) 
                self.ClipLowLim_cyto.setValue(int(p2))
            elif self.current_chan == "nuc":
                self.ClipHighLim_nuc.setValue(int(p98)) 
                self.ClipLowLim_nuc.setValue(int(p2))


    ############################## Update Clip limits ######################################

    def cyto_clip_lower_change(self):
        self.ClipLowLim = self.ClipLowLim_cyto.value()
        self.plot_slice()

    def cyto_clip_higher_change(self):
        self.ClipHighLim = self.ClipHighLim_cyto.value()
        self.plot_slice()

    def nuc_clip_lower_change(self):
        self.ClipLowLim = self.ClipLowLim_nuc.value()
        self.plot_slice()

    def nuc_clip_higher_change(self):
        self.ClipHighLim = self.ClipHighLim_nuc.value()
        self.plot_slice()


    ######################### Contrast Enhancement Method ##################################

    def CLAHE(self, block):

        block = np.clip(block, int(self.ClipLowLim), int(self.ClipHighLim))
        # set clahe kernel size to be 1/4 image size
        kernel_size = np.asarray([int(self.ROI_dim)//16, #1/4 kernel size
                                  int(self.ROI_dim)//16,
                                  int(self.ROI_dim)//16])
        # equalize histogram and convert to 8 bit
        block = equalize_adapthist(block,
                                   kernel_size=kernel_size,
                                   clip_limit=0.01)*255
        return block.astype(np.uint8)

    def Rescale(self, current_slice):

        current_slice = rescale_intensity(np.clip(current_slice,
                                                  self.ClipLowLim,
                                                  self.ClipHighLim),
                                         out_range='uint8')
        return current_slice

    def methodchange(self):

        self.plot_slice()


    ################################## Update Plot #######################################

    def plot_slice(self):
        """
        This function..
        - applies contrast enhancement to selected slice
        - changes self.img to scale [0,255]
        - plot current slice (selected z-level)

        """
        current_slice = self.img[self.current_z_level, :, :]
        selected_method = "Rescale"

        if selected_method == "Rescale":
            current_slice = self.Rescale(current_slice)
        elif selected_method == "CLAHE":
            xstart = int(self.x_limits[0])
            xend = xstart + int(self.ROI_dim/4)
            ystart = int(self.y_limits[1])
            yend = ystart + int(self.ROI_dim/4)
            self.x_limits = [xstart, xend]
            self.y_limits = [yend, ystart]
            block = self.img[:,ystart:yend,xstart:xend]
            block = self.CLAHE(block)
            current_slice = np.zeros_like(current_slice)
            current_slice[ystart:yend,xstart:xend] = block[self.current_z_level, :, :]

        # Plot current slice
        self.figure.clear()
        self.vmax = 255
        ax = self.figure.add_subplot(111)
        ax.imshow(current_slice, cmap='gray', vmax=self.vmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vol Preview')
        ax.set_xlim(self.x_limits)  
        ax.set_ylim(self.y_limits)  

        self.canvas.draw()

    ############################## Print loading/saving ###################################

    def show_loading_data(self):
        self.loading_label.setText("Loading data...")
        self.loading_label.setStyleSheet("color: blue;")
        QApplication.processEvents()

    def hide_text(self):
        self.loading_label.clear()
        QApplication.processEvents()

    def show_saving(self):
        self.loading_label.setText("Saving...")
        self.loading_label.setStyleSheet("color: green;")
        QApplication.processEvents()

    def show_OutofBound(self):
        self.loading_label.setText("Out of bound! Don't save!")
        self.loading_label.setStyleSheet("color: red;")
        QApplication.processEvents()


    ################################ Action buttons ######################################

    def select_file(self):
        """
        This function..
        - allows user select file to preview
        - record file path
        - call the function: readHDF5 and plotinit
        """
        file_dialog = QFileDialog()
        # file_dialog.setDirectory("W:/Trilabel_Data")
        file_path, _ = file_dialog.getOpenFileName(self, "Select fused HDF5 File")
        if file_path:
            self.h5path = file_path
            print(self.h5path)
            self.show_loading_data()
            self.readHDF5()
            self.plot_init_z()
            self.hide_text()
        else: 
            #self.h5path = "W:\\Trilabel_Data\\PGP9.5\\OTLS4_NODO_6-7-23_16-043J_PGP9.5\\data-f0.h5"
            pass

    def select_savehome(self):
        file_dialog = QFileDialog()
        savehome = file_dialog.getExistingDirectory(self, "Select save directory")
        self.save_home = savehome
        if savehome:
            if self.Antibody == "":
                self.Antibody = "PGP9.5"
            self.Ab_home = savehome + os.sep + self.Antibody
            print("Save directory : ", self.Ab_home)
            self.show_savedir()
        else: 
            pass

    def go_home(self):
        """
        This function go back to the whole slice (from a zoomed in region)
        """
        self.x_limits = [0, self.shape[2]]
        self.y_limits = [self.shape[1], 0]
        self.crop_button.setChecked(False)
        self.y_coordinate_textbox.setText("0")
        self.x_coordinate_textbox.setText("0")
        self.plot_slice()


    def save_coords(self):
        """
        This function..
        - retrieves all the values to be saved
        - write it into a .csv file when "Save" button is clicked
        """
        self.show_saving()

        xcoord   = int(self.x_limits[0]*4)
        ycoord   = int(self.y_limits[1]*4)
        currentZ = int(self.current_z_level*4)
        ROI_dim = int(self.ROI_dim)
        orient = int(self.orient)
        shape    = self.arrayshape_textbox.text()
        cyto_clipLow  = self.ClipLowLim_cyto.value()
        cyto_clipHigh = self.ClipHighLim_cyto.value()
        nuc_clipLow   = self.ClipLowLim_nuc.value()
        nuc_clipHigh  = self.ClipHighLim_nuc.value()

        note = self.note_textbox.text()
        h5path = self.h5path
        Abhome = self.Ab_home

        if xcoord < 0:
            xcoord = 0
        if ycoord < 0:
            ycoord = 0

        headers = ["h5path", "Abhome", "xcoord", "ycoord", "zcoord", "ROIdim", "Shape(8xds)",
                   "orient", "cyto_clipLow", "cyto_clipHigh", "nuc_clipLow", "nuc_clipHigh",
                    "Note"]
        values =  [h5path, Abhome, ycoord, xcoord, currentZ, ROI_dim, shape,
                   orient, cyto_clipLow, cyto_clipHigh, nuc_clipLow,
                   nuc_clipHigh, note]

        date =  str(datetime.date.today())
        filename = "ROI_coords_" + date + ".csv"
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(headers)
            writer.writerow(values)
        
        time.sleep(0.2)
        self.hide_text()
        print("saved")


    #################################
    ######## Shortcut Key############
    #################################

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.update_img2cyto()
        if event.key() == Qt.Key_S:
            self.update_img2nuc()
        if event.key() == Qt.Key_D:
            self.update_img2FC()
        if event.key() == Qt.Key_C:
            self.Crop_ROI()
        if event.key() == Qt.Key_R:
            self.Auto_Rescale()
        if event.key() == Qt.Key_F:
            self.go_home()
        if event.key() == Qt.Key_Z:
            self.toolbar.actions()[5].trigger()  # Zoom in
        if event.key() == Qt.Key_E:
            self.toolbar.actions()[4].trigger()  # Drag



app = QApplication([])
w = MainWindow()
w.show()
app.exec()
