#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import argparse
import time
import os
from datetime import timedelta

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False

parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subpixel', action='store_true', help="Enable subpixel mode.")
parser.add_argument('-ext', '--extended_disparity', action='store_true', help="Enable extended disparity mode.")
parser.add_argument('-alpha', type=float, default=None, help="Alpha scaling parameter to increase float. [0,1] valid interval.")
args = parser.parse_args()
alpha = args.alpha
subpixel = args.subpixel
extended_disparity = args.extended_disparity

fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_480_P

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.Camera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
sync = pipeline.create(dai.node.Sync)

xoutGrp = pipeline.create(dai.node.XLinkOut)

xoutGrp.setStreamName("xout")

#Properties
rgbCamSocket = dai.CameraBoardSocket.CAM_A

camRgb.setBoardSocket(rgbCamSocket)
camRgb.setSize(1920, 1080)
camRgb.setFps(fps)

# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(rgbCamSocket)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise

sync.setSyncThreshold(timedelta(milliseconds=50))

left.setResolution(monoResolution)
left.setCamera("left")
left.setFps(fps)
right.setResolution(monoResolution)
right.setCamera("right")
right.setFps(fps)


# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setExtendedDisparity(extended_disparity)
stereo.setSubpixel(subpixel)
#stereo.setDepthAlign(DepthAdepthai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER)
stereo.setDepthAlign(rgbCamSocket)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 28
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

# Linking
left.out.link(stereo.left)
right.out.link(stereo.right)
camRgb.video.link(sync.inputs["rgb"])
stereo.disparity.link(sync.inputs["disp"])
sync.out.link(xoutGrp.input)

camRgb.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
if alpha is not None:
    camRgb.setCalibrationAlpha(alpha)
    stereo.setAlphaScaling(alpha)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameDisp = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbdWindowName = "rgbd"
    cv2.namedWindow(rgbdWindowName)
    disparityMultiplier = 255.0 / stereo.initialConfig.getMaxDisparity()
    queue = device.getOutputQueue("xout", 10, False)

    display_width = 400
    aspect_ratio = 1920.0 / 1080.0
    display_height = int(display_width / aspect_ratio)
    
    while True:
    
        latestImage = {}
        latestImage["rgb"] = None
        latestImage["disp"] = None

        msgGrp = queue.get()
        for name, msg in msgGrp:
            if name == "disp":
                frame = msg.getFrame()
                frame = (frame * disparityMultiplier).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                latestImage[name] = np.ascontiguousarray(frame)

            elif name == "rgb":
                frame = msg.getCvFrame()
                latestImage[name] = frame
                
            if latestImage["rgb"] is not None and latestImage["disp"] is not None:
                resized_rgb = cv2.resize(latestImage["rgb"], (display_width, display_height))        
                resized_disp = cv2.resize(latestImage["disp"], (display_width, display_height))
                resized_concat_frame = cv2.hconcat([resized_rgb, resized_disp])
                cv2.imshow(rgbdWindowName, resized_concat_frame)
                
                key = cv2.waitKey(1)

                if key == ord('p'):
                    print("rgb size: ", latestImage["rgb"].shape)
                    #resized_disp = cv2.resize(latestImage["disp"], (latestImage["rgb"].shape[1], latestImage["rgb"].shape[0]))
                    print("disp size: ", latestImage["disp"].shape)
                    concat_frame = cv2.hconcat([latestImage["rgb"], latestImage["disp"]])

                    filename = f"/home/{os.getlogin()}/Desktop/rgb_depth/frame_{int(time.time())}.png"
                    cv2.imwrite(filename, concat_frame)
                    print(f"Saved frame to {filename}")

                    # Flash the screen white briefly
                    flash_frame = np.ones_like(resized_concat_frame) * 255
                    cv2.imshow(rgbdWindowName, flash_frame)
                    cv2.waitKey(100)  # Display the white frame for 100 ms
                elif key == ord('q'):
                    break
                latestImage["rgb"] = None
                latestImage["disp"] = None
        if key == ord('q'):
            break    
            
