from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QDoubleSpinBox, QGridLayout, QWidget, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QMessageBox, QSizePolicy, QComboBox,
    QSpacerItem, QHBoxLayout, QVBoxLayout, QGroupBox, QLineEdit, QFormLayout, QFileDialog
)

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from skimage.exposure import equalize_adapthist, rescale_intensity
import time
import csv
import datetime

"""
v1: all channels with CLAHE and Rescale
v2: only target CLAHE others Rescale

If you are accessing from a different server/machine:
******REMEMBER TO CHANGE THE SERVER DRIVE******
Change it in the function: select_file(self)
Example: "W:/Trilabel_Data" to "X:/Trilabel_Data"

Sarah, Jul 2023
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        ##########################################################################
        ################# Layout #################################################

        self.setWindowTitle("ROI Preview")

        # Create layout for the UI
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Readme Note
        note_label = QLabel("Caution! Beware of out of bound error: \n \
              You are viewing the 3x downsampled of fused.h5 file, so axis abd dimension should be x4 \n \
              Make sure enough space saving coords on edges, do not save when current z-level > z dim - 12 \n \
              Try not to zoom in to an ROI smaller than 160x160p \n \n \
                    Press 'Save params' after you got the desired coords and clipping vals for all 3 channels")

        # Create a matplotlib figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_release_event', self.on_zoom_completed)
        self.toolbar = NavigationToolbar(self.canvas, self)        

        # Add button to change between channels
        button_layout = QHBoxLayout()
        self.cyto_button = QPushButton("Cyto")
        self.cyto_button.setCheckable(True)
        self.cyto_button.clicked.connect(self.update_img2cyto)
        button_layout.addWidget(self.cyto_button)
        self.nuc_button = QPushButton("Nuc")
        self.nuc_button.setCheckable(True)
        self.nuc_button.clicked.connect(self.update_img2nuc)
        button_layout.addWidget(self.nuc_button)
        self.target_button = QPushButton("Target")
        self.target_button.setCheckable(True)
        self.target_button.clicked.connect(self.update_img2target)
        button_layout.addWidget(self.target_button)
        button_container = QGroupBox("Channels")
        button_container.setLayout(button_layout)
        self.setCentralWidget(button_container)

        # Add x,y,z coordinate display
        self.x_coordinate_textbox = QLineEdit()
        self.x_coordinate_textbox.setReadOnly(True)
        self.y_coordinate_textbox = QLineEdit()
        self.y_coordinate_textbox.setReadOnly(True)
        self.arrayshape_textbox = QLineEdit()
        self.arrayshape_textbox.setReadOnly(True)
        self.current_z_level_textbox = QLineEdit()
        self.current_z_level_textbox.setReadOnly(True)
        form_layout = QFormLayout()
        form_layout.addRow("x-coordinate:", self.x_coordinate_textbox)
        form_layout.addRow("y-coordinate:", self.y_coordinate_textbox)
        form_layout.addRow("Current Z-Level:", self.current_z_level_textbox)
        form_layout.addRow("Vol Dim [z,y,x]:", self.arrayshape_textbox)
        coordinates_container = QGroupBox("Coordinates")
        coordinates_container.setLayout(form_layout)

        # Add ROI dimension textbox, Crop button, and Auto Rescale button
        roi_dim_textbox = QLineEdit()
        roi_dim_label = QLabel("ROI dim:")
        crop_button = QPushButton("Crop")
        auto_rescale_button = QPushButton("Auto Rescale")

        roi_dim_layout = QHBoxLayout()
        roi_dim_layout.addWidget(roi_dim_label)
        roi_dim_layout.addWidget(roi_dim_textbox)

        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(crop_button)
        button_layout2.addWidget(auto_rescale_button)

        # Create a group box for the buttons and textbox
        button_group = QGroupBox()
        button_group_layout = QVBoxLayout()
        button_group_layout.addLayout(roi_dim_layout)
        button_group_layout.addLayout(button_layout2)
        button_group.setLayout(button_group_layout)

        # Add Contrast enhancement for cyto channel
        ClipLow_Cyto_default = 0
        ClipHigh_Cyto_default = 1200
        dropdown_layout1 = QVBoxLayout()
        dropdown_layout1.addWidget(QLabel("Contrast Enhancement Method:"))
        self.dropdown1 = QComboBox()
        self.dropdown1.addItem("Rescale")
        dropdown_layout1.addWidget(self.dropdown1)
        clip_high_layout1 = QHBoxLayout()
        clip_high_layout1.addWidget(QLabel("Clip High:"))
        self.ClipHighLim_cyto = QDoubleSpinBox()
        self.ClipHighLim_cyto.setRange(0, 4500)
        self.ClipHighLim_cyto.setSingleStep(50)
        self.ClipHighLim_cyto.setValue(ClipHigh_Cyto_default)
        clip_high_layout1.addWidget(self.ClipHighLim_cyto)
        clip_low_layout1 = QHBoxLayout()
        clip_low_layout1.addWidget(QLabel("Clip Low:"))
        self.ClipLowLim_cyto = QDoubleSpinBox()
        self.ClipLowLim_cyto.setRange(0, 4500)
        self.ClipLowLim_cyto.setSingleStep(50)
        self.ClipLowLim_cyto.setValue(ClipLow_Cyto_default)
        self.ClipHighLim_cyto.valueChanged.connect(self.cyto_clip_higher_change)
        self.ClipLowLim_cyto.valueChanged.connect(self.cyto_clip_lower_change)

        clip_low_layout1.addWidget(self.ClipLowLim_cyto)
        dropdown_layout1.addLayout(clip_high_layout1)
        dropdown_layout1.addLayout(clip_low_layout1)
        dropdown_container1 = QGroupBox("Cyto")
        dropdown_container1.setLayout(dropdown_layout1)

        # Add Contrast enhancement for Nuc channel
        ClipLow_Nuc_default = 0
        ClipHigh_Nuc_default = 2000
        dropdown_layout2 = QVBoxLayout()
        dropdown_layout2.addWidget(QLabel("Contrast Enhancement Method:"))
        self.dropdown2 = QComboBox()
        self.dropdown2.addItem("Rescale")
        dropdown_layout2.addWidget(self.dropdown2)
        clip_high_layout2 = QHBoxLayout()
        clip_high_layout2.addWidget(QLabel("Clip High:"))
        self.ClipHighLim_nuc = QDoubleSpinBox()
        self.ClipHighLim_nuc.setRange(0, 4500)
        self.ClipHighLim_nuc.setSingleStep(50)
        self.ClipHighLim_nuc.setValue(ClipHigh_Nuc_default)
        clip_high_layout2.addWidget(self.ClipHighLim_nuc)
        clip_low_layout2 = QHBoxLayout()
        clip_low_layout2.addWidget(QLabel("Clip Low:"))
        self.ClipLowLim_nuc = QDoubleSpinBox()
        self.ClipLowLim_nuc.setRange(0, 4500)
        self.ClipLowLim_nuc.setSingleStep(50)
        self.ClipLowLim_nuc.setValue(ClipLow_Nuc_default) 
        self.ClipHighLim_nuc.valueChanged.connect(self.nuc_clip_higher_change)
        self.ClipLowLim_nuc.valueChanged.connect(self.nuc_clip_lower_change)

        clip_low_layout2.addWidget(self.ClipLowLim_nuc)
        dropdown_layout2.addLayout(clip_high_layout2)
        dropdown_layout2.addLayout(clip_low_layout2)
        dropdown_container2 = QGroupBox("Nuc")
        dropdown_container2.setLayout(dropdown_layout2)

        # Add Contrast enhancement for Target channel
        ClipLow_PGP_default = 0
        ClipHigh_PGP_default = 700
        dropdown_layout3 = QVBoxLayout()
        dropdown_layout3.addWidget(QLabel("Contrast Enhancement Method:"))
        self.dropdown3 = QComboBox()
        self.dropdown3.addItem("Rescale")
        self.dropdown3.addItem("CLAHE")
        dropdown_layout3.addWidget(self.dropdown3)
        clip_high_layout3 = QHBoxLayout()
        clip_high_layout3.addWidget(QLabel("Clip High:"))
        self.ClipHighLim_pgp = QDoubleSpinBox()
        self.ClipHighLim_pgp.setRange(0, 4500)
        self.ClipHighLim_pgp.setSingleStep(50)
        self.ClipHighLim_pgp.setValue(ClipHigh_PGP_default)
        clip_high_layout3.addWidget(self.ClipHighLim_pgp)
        clip_low_layout3 = QHBoxLayout()
        clip_low_layout3.addWidget(QLabel("Clip Low:"))
        self.ClipLowLim_pgp = QDoubleSpinBox()
        self.ClipLowLim_pgp.setRange(0, 4500)
        self.ClipLowLim_pgp.setSingleStep(50)
        self.ClipLowLim_pgp.setValue(ClipLow_PGP_default)
        self.ClipHighLim_pgp.valueChanged.connect(self.pgp_clip_higher_change)
        self.ClipLowLim_pgp.valueChanged.connect(self.pgp_clip_lower_change)
        self.dropdown3.currentIndexChanged.connect(self.methodchange)

        clip_low_layout3.addWidget(self.ClipLowLim_pgp)
        dropdown_layout3.addLayout(clip_high_layout3)
        dropdown_layout3.addLayout(clip_low_layout3)
        dropdown_container3 = QGroupBox("Target")
        dropdown_container3.setLayout(dropdown_layout3)

        # Add Save button and home button
        file_button = QPushButton("Select File")
        file_button.clicked.connect(self.select_file) 
        home_button = QPushButton("Home")
        home_button.clicked.connect(self.go_home)
        save_button = QPushButton("Save params")
        save_button.setStyleSheet("QPushButton {padding: 5px; font-weight: bold;}")
        save_button.clicked.connect(self.save_coords)

        button_layout = QHBoxLayout()
        button_layout.addWidget(file_button)
        button_layout.addWidget(home_button)
        button_layout.addSpacing(20)
        button_layout.addWidget(save_button)
        button_layout.addStretch()

        # Configure left layout and right layout and make it central
        left_layout.addWidget(note_label)
        left_layout.addWidget(self.canvas, stretch=1)
        left_layout.addWidget(self.toolbar)
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

        ############# End of Layout #######################################################
        ###################################################################################

        # Initialization
        self.select_file() # Including readHDF5() and plot_init_z()

        # Connect the mouse wheel event to the update_z_level method
        self.canvas.mpl_connect('scroll_event', self.update_z_level)

    ##################### Initialization functions ###########################################

    def readHDF5(self):
    
        start = time.time()
        with h5.File(self.h5path, 'r') as f:
            self.cyto = f['t00000']['s00']['3/cells'][:, :, :].astype(np.uint16)
            self.cyto = np.moveaxis(self.cyto, 0, 1)
            self.nuc = f['t00000']['s01']['3/cells'][:, :, :].astype(np.uint16)
            self.nuc = np.moveaxis(self.nuc, 0, 1)
            self.pgp = f['t00000']['s02']['3/cells'][:, :, :].astype(np.uint16)
            self.pgp = np.moveaxis(self.pgp, 0, 1)
        f.close()
        print(time.time() - start, "s")

    def plot_init_z(self):

        # Initial params
        self.img = self.cyto
        self.current_chan = "cyto"
        self.vmax = 1200
        self.ClipLowLim = 0
        self.ClipHighLim = 1200

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

        self.nuc_button.setChecked(False)  
        self.target_button.setChecked(False)
        self.dropdown1.setEnabled(True)
        self.dropdown2.setEnabled(False)  
        self.dropdown3.setEnabled(False)  
        self.ClipHighLim_cyto.setEnabled(True) 
        self.ClipLowLim_cyto.setEnabled(True)
        self.ClipHighLim_nuc.setEnabled(False) 
        self.ClipLowLim_nuc.setEnabled(False)  
        self.ClipHighLim_pgp.setEnabled(False) 
        self.ClipLowLim_pgp.setEnabled(False)

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


    ########################## Update current Z-level ##########################################

    def update_z_level(self, event):

        if event.button == 'up' or event.button == 'down':

            current_z_level = self.current_z_level
            # Update the z-level based on the scroll direction
            if event.button == 'up':
                current_z_level += 1
                if current_z_level >= self.shape[0]:
                    current_z_level = self.shape[0] - 1
            else:
                current_z_level -= 1
                if current_z_level < 0:
                    current_z_level = 0

            self.current_z_level = current_z_level
            self.current_z_level_textbox.setText(str(self.current_z_level))

            self.plot_slice()

    ########################## Update Channel ##########################################

    def update_img2cyto(self):
        self.img = self.cyto
        self.current_chan = "cyto"
        self.ClipLowLim = self.ClipLowLim_cyto.value()
        self.ClipHighLim = self.ClipHighLim_cyto.value()
        self.nuc_button.setChecked(False)  
        self.target_button.setChecked(False)
        self.dropdown1.setEnabled(True)
        self.dropdown2.setEnabled(False)  
        self.dropdown3.setEnabled(False)  
        self.ClipHighLim_cyto.setEnabled(True)  
        self.ClipLowLim_cyto.setEnabled(True)
        self.ClipHighLim_nuc.setEnabled(False)  
        self.ClipLowLim_nuc.setEnabled(False) 
        self.ClipHighLim_pgp.setEnabled(False) 
        self.ClipLowLim_pgp.setEnabled(False)

        self.plot_slice()

    def update_img2nuc(self):
        self.img = self.nuc
        self.current_chan = "nuc"
        self.ClipLowLim = self.ClipLowLim_nuc.value()
        self.ClipHighLim = self.ClipHighLim_nuc.value()
        self.cyto_button.setChecked(False)  
        self.target_button.setChecked(False)
        self.dropdown1.setEnabled(False)
        self.dropdown2.setEnabled(True)
        self.dropdown3.setEnabled(False)  
        self.ClipHighLim_nuc.setEnabled(True)  
        self.ClipLowLim_nuc.setEnabled(True)
        self.ClipHighLim_cyto.setEnabled(False)  
        self.ClipLowLim_cyto.setEnabled(False) 
        self.ClipHighLim_pgp.setEnabled(False) 
        self.ClipLowLim_pgp.setEnabled(False)

        self.plot_slice()


    def update_img2target(self):
        self.img = self.pgp
        self.current_chan = "Target"
        self.ClipLowLim = self.ClipLowLim_pgp.value()
        self.ClipHighLim = self.ClipHighLim_pgp.value()
        self.cyto_button.setChecked(False)  
        self.nuc_button.setChecked(False)
        self.dropdown1.setEnabled(False)
        self.dropdown2.setEnabled(False)  
        self.dropdown3.setEnabled(True)
        self.ClipHighLim_pgp.setEnabled(True)  
        self.ClipLowLim_pgp.setEnabled(True)
        self.ClipHighLim_nuc.setEnabled(False) 
        self.ClipLowLim_nuc.setEnabled(False) 
        self.ClipHighLim_cyto.setEnabled(False)
        self.ClipLowLim_cyto.setEnabled(False)  

        self.plot_slice()


    ########################### Update Clip limits #######################################

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

    def pgp_clip_lower_change(self):
        self.ClipLowLim = self.ClipLowLim_pgp.value()
        self.plot_slice()

    def pgp_clip_higher_change(self):
        self.ClipHighLim = self.ClipHighLim_pgp.value()
        self.plot_slice()

    ######################### Contrast Enhancement Method ##################################

    def CLAHE(self, block):

        block = np.clip(block, int(self.ClipLowLim), int(self.ClipHighLim))
        # set clahe kernel size to be 1/4 image size
        kernel_size = np.asarray([160//4,
                                  160//4,
                                  160//4])
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
        if self.current_chan == "Target":
            selected_method = self.dropdown3.currentText()
        else:
            selected_method = "Rescale"

        if selected_method == "Rescale":
            current_slice = self.Rescale(current_slice)
        elif selected_method == "CLAHE":
            xstart = int(self.x_limits[0])
            xend = xstart + 160
            ystart = int(self.y_limits[1])
            yend = ystart + 160
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

    ################################ Action buttons ######################################

    def select_file(self):
        """
        This function..
        - allows user select file to preview
        - record file path
        - read HDF5 file and plot initial z-level
        """
        file_dialog = QFileDialog()
        file_dialog.setDirectory("W:/Trilabel_Data")
        file_path, _ = file_dialog.getOpenFileName(self, "Select File")
        if file_path:
            self.h5path = file_path
        else: 
            self.h5path = "W:\\Trilabel_Data\\PGP9.5\\OTLS4_NODO_6-7-23_16-043J_PGP9.5\\data-f0.h5"
        
        print(self.h5path)
        self.readHDF5()
        self.plot_init_z()



    def go_home(self):
        """
        This function go back to the whole slice (from a zoomed in region)
        """
        self.x_limits = [0, self.shape[2]]
        self.y_limits = [self.shape[1], 0]
        self.dropdown3.setCurrentIndex(0) # change the method to "rescale"
        self.plot_slice()

    def save_coords(self):
        """
        This function..
        - retrieves all the values to be saved
        - write it into a .csv file when "Save" button is clicked
        """
        xcoord   = int(self.x_limits[0]*4)
        ycoord   = int(self.y_limits[1]*4)
        currentZ = int(self.current_z_level*4)
        shape    = self.arrayshape_textbox.text()
        cyto_clipLow  = self.ClipLowLim_cyto.value()
        cyto_clipHigh = self.ClipHighLim_cyto.value()
        nuc_clipLow   = self.ClipLowLim_nuc.value()
        nuc_clipHigh  = self.ClipHighLim_nuc.value()
        pgp_clipLow   = self.ClipLowLim_pgp.value()
        pgp_clipHigh  = self.ClipHighLim_pgp.value()
        pgp_ctehmt_method = self.dropdown3.currentText()
        h5path = self.h5path

        headers = ["h5path", "xcoord", "ycoord", "CurrentZlevel", "Shape(3xds)",
                   "cyto_clipLow", "cyto_clipHigh", "nuc_clipLow", "nuc_clipHigh",
                   "pgp_ctehmt_method", "pgp_clipLow", "pgp_clipHigh"]
        values =  [h5path, ycoord, xcoord, currentZ, shape,
                   cyto_clipLow, cyto_clipHigh, nuc_clipLow,
                   nuc_clipHigh, pgp_ctehmt_method, pgp_clipLow, pgp_clipHigh]

        date =  str(datetime.date.today())
        filename = "ROI_coords_" + date + ".csv"
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(headers)
            writer.writerow(values)


app = QApplication([])
w = MainWindow()
w.show()
app.exec()