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

# make cvColorMap the reverse of cv2.COLORMAP_HOT
cvColorMap = cv2.COLORMAP_PLASMA
cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cvColorMap)
cvColorMap = cvColorMap[::-1]
cvColorMap[0] = [0, 0, 0]  # Set 0 to black

cvDispColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
cvDispColorMap[0] = [0, 0, 0]  # Set 0 to black



parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subpixel', action='store_true', help="Use for better accuracy for longer distance.")
parser.add_argument('-ext', '--extended_disparity', action='store_true', help="Use for closer-in minimum depth")
parser.add_argument('-alpha', type=float, default=None, help="Alpha scaling parameter to increase float. [0,1] valid interval.")
parser.add_argument('-post', '--post', action='store_true', help="Enable post processing")  # Enable post processing


args = parser.parse_args()
alpha = args.alpha
subpixel = args.subpixel
extended_disparity = args.extended_disparity
post = args.post

use_disparity = not post or subpixel # bug work around for disparity not correct size when post processing is enabled 
if not use_disparity:
    print("Due to bug disparity does not work with post processing unless subpixel is enabled")
    exit(1)

print("subpixel = ", subpixel)
print("extended_disparity = ", extended_disparity)
print("use_disparity = ", use_disparity)

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
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setExtendedDisparity(extended_disparity)
stereo.setSubpixel(subpixel)
stereo.setDepthAlign(rgbCamSocket)

if post:
    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 28
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 10
    config.postProcessing.spatialFilter.numIterations = 1
    #config.postProcessing.thresholdFilter.minRange = 200
    #config.postProcessing.thresholdFilter.maxRange = 5000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

# Linking
left.out.link(stereo.left)
right.out.link(stereo.right)
camRgb.video.link(sync.inputs["rgb"])
#stereo.depth.link(sync.inputs["depth"])
stereo.disparity.link(sync.inputs["disp"])
sync.out.link(xoutGrp.input)

camRgb.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
if alpha is not None:
    camRgb.setCalibrationAlpha(alpha)
    stereo.setAlphaScaling(alpha)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbdWindowName = "rgbd"
    cv2.namedWindow(rgbdWindowName)
    #print("max disparity: ", stereo.initialConfig.getMaxDisparity())
    disparityMultiplier = 255.0 / stereo.initialConfig.getMaxDisparity()
    queue = device.getOutputQueue("xout", 10, False)

    display_width = 400
    aspect_ratio = 1920.0 / 1080.0
    display_height = int(display_width / aspect_ratio)
    
    while True:
    
        latestImage = {}
        latestImage["rgb"] = None
        #latestImage["depth"] = None
        latestImage["disp"] = None

        msgGrp = queue.get()
        for name, msg in msgGrp:
            if name == "disp":
                #print("disp width =", msg.getWidth(), "height = ", msg.getHeight(), "disp type is ", msg.getType())
                frame = msg.getFrame()
                #frame = (frame * disparityMultiplier).astype(np.uint8)
                latestImage[name] = frame
                
            # elif name == "depth":
            #     frame = msg.getFrame()
            #     latestImage[name] = frame
        
            elif name == "rgb":
                frame = msg.getCvFrame()
                latestImage[name] = frame
                
            if latestImage["rgb"] is not None and latestImage["disp"] is not None:                
                resized_rgb = cv2.resize(latestImage["rgb"], (display_width, display_height))
                disp = (latestImage["disp"] * disparityMultiplier).astype(np.uint8)
                resized_disp = cv2.resize(disp, (display_width, display_height))
                resized_disp = cv2.applyColorMap(resized_disp, cvDispColorMap)
                resized_concat_frame = cv2.hconcat([resized_rgb, resized_disp])
                cv2.imshow(rgbdWindowName, resized_concat_frame)
                
                key = cv2.waitKey(1)

                if key == ord('p'):
                    dispFrame = (latestImage["disp"] << 3)
                    rgbFrame = (latestImage["rgb"].astype(np.uint16) << 8)
                    if len(dispFrame.shape) < 3:
                        dispFrame = cv2.cvtColor(dispFrame, cv2.COLOR_GRAY2BGR)
                    
                    #print the min above 0 and max values in the depth
                    # print("Min disparity value: ", np.min(dispFrame[dispFrame > 0]))
                    # print("Max disparity value: ", np.max(dispFrame))
                    
                    #print shape of dispFrame and rgbFrame
                    # print("dispFrame shape: ", dispFrame.shape)
                    # print("rgbFrame shape: ", rgbFrame.shape)
                    concat_frame = cv2.hconcat([rgbFrame, dispFrame])

                    filename = f"/home/{os.getlogin()}/Desktop/rgb_depth/frame_{int(time.time())}.png"
                    cv2.imwrite(filename, concat_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])  # Save as PNG
                    print(f"Saved frame to {filename}")
                    # Flash the screen white briefly
                    flash_frame = np.ones_like(resized_concat_frame) * 255
                    cv2.imshow(rgbdWindowName, flash_frame)
                    cv2.waitKey(100)  # Display the white frame for 100 ms
                elif key == ord('q'):
                    break
                latestImage["rgb"] = None
                #latestImage["depth"] = None
                latestImage["disp"] = None
        if key == ord('q'):
            break    
            
