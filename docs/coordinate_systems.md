# Coordinate Systems

The document provides information about the various coordinate systems that are used in the codebase, and also information about the various camera coordinate system formats that are used in the codebase.

## A. Camera Coordinate System Formats
In this codebase, a "coordinate system format" is used to describe how the axes of a coordinate system are oriented with reference to "some thing". If we are dealing with a camera coordinate system, we can define a coordinate system format based on how its axes are oriented with respect to the image plane.

In this codebase, we would like define two camera coordinate system formats. They are:

2. The Classic Computer Vision (Classic CV) Coordinate System Format
2. The OpenGL Coordinate System Format.

A diagramatic represtation of both formats is shown in Fig. 1. below:

<img src="media/camera_coordinate_system_formats.png" alt="camera_coordinate_system_formats" style="zoom: 33%;" />

<div style="text-align:center"><i>Fig. 1: On the left, the Classic CV coordinate system format is shown. On the right, the OpenGL coordinate system format is shown.</i></div>

The two important points that are to be noted are:

1. Both coordinate system formats are right-handed.
2. In the Classic CV coodinate system format, the positive direction of the Z-Axis **enters** the image plane. In the OpenGL coordinate coordinate system format, the positive direction of the Z-Axis **leaves** the image plane.

This camera coordinate systems in **most parts** of this codebase is in the Classic CV coordinate format. This is **important** to note because the poses in the official implementation are in the OpenGL format.

In **some parts** of the codebase however, the OpenGL format maybe used. Notably, the poses in the blender dataset are in the OpenGl format. However, once the poses are retrieved from the loaded metadata files, they are converted to the Classic CV format (you may refer to the function `_opengl_to_classic_cv` in the class `BlenderDataset` for information on how the convertion is made). 

## B. Coordinate Systems in the Code
Many different types of coordinate systems are mentioned in the codebase. In many places in the codebase, transformations between coordinate systems are used. This section provides information about the various coordinate systems that are mentioned in the codebase.

> TODO: Elaborate the reasoning behind having these different world coordinate systems.

### 1. World 1 Coordinate System (W1)
This is the world coordinate system of the dataset.

### 2. World 2 Coordinate System (W2)
For a given dataset representing a 360-degree inward-facing sene, the codebase sets up a new world coordinate system called W2. The position and orientation of this new world coordinate system W2 may be different from W1 (however in some cases, it may be the same as well in some cases). This new world coordinate system W2 has the same scale as of W1.

### 3. World 3 Coordinate System (W3)
Given the newly setup W2 coordinate system, the codebase also setups another coordinate system called W3. The W3 coordinate system is just a scaled version of the W2 coordinate system.

### 4. Camera Coordinate System (C)
Every camera in the dataset has its own camera coordiante system.
