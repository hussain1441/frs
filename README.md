# Face Recognition System for Attendance

## Overview

This Face Recognition System (FRS) is designed for automated attendance tracking using advanced face recognition technology. The system uses the InsightFace model to detect and recognize faces in real-time, automatically marking attendance when employees enter or exit designated areas.

## Table of Contents

-   [Prerequisites](#prerequisites)
-   [Tech Stack](#tech-stack)
-   [Hardware Requirements](#hardware-requirements)
-   [Installation & Setup](#installation--setup)
    -   [1. RTSP Server Setup](#1-rtsp-server-setup)
    -   [2. GPU Configuration (Optional but Recommended)](#2-gpu-configuration-optional-but-recommended)
    -   [3. MongoDB Setup](#3-mongodb-setup)
    -   [4. Python Environment Setup](#4-python-environment-setup)
-   [Project Structure](#project-structure)
-   [How It Works](#how-it-works)
-   [Usage](#usage)
-   [Model Selection](#model-selection)
-   [Technical Hub Integration](#technical-hub-integration)
-   [Jetson Compatibility](#jetson-compatibility)
-   [Troubleshooting](#troubleshooting)

## Prerequisites

-   **Python 3.8+** with pip
-   **FFmpeg** for video streaming
-   **MongoDB Compass** for database management
-   **CUDA 12.5 and cuDNN 9.1** (for GPU acceleration)
-   **NVIDIA GPU** (RTX 3050 or better recommended for optimal performance)

## Tech Stack

-   **InsightFace** - Face detection and recognition model
-   **OpenCV** - Video processing and computer vision
-   **FastAPI** - Web server for browser-based streaming
-   **MongoDB** - Database for storing face embeddings and attendance records
-   **PyTorch** - Deep learning framework
-   **ONNX Runtime** - Model inference (CPU/GPU)

## Hardware Requirements

-   **Recommended**: System with NVIDIA RTX 3050 or better
-   **Camera Options**:
    -   USB webcam (fastest, no latency)
    -   Built-in system camera
    -   Intel RealSense camera
    -   RTSP network camera (has latency)

> **Note**: Direct USB or built-in cameras are strongly recommended over RTSP streams due to significantly lower latency.

## Installation & Setup

### 1. RTSP Server Setup

RTSP (Real-Time Streaming Protocol) allows streaming video over a network, though using a directly connected camera is faster.

#### Download MediaMTX

1. Visit [MediaMTX Releases](https://github.com/bluenviron/mediamtx/releases)
2. Download the version for your OS (Windows/Linux/macOS)
3. For macOS, the executable is located in `environment-mac/mediamtx`
4. Run the `mediamtx` executable to start the RTSP server

#### Install FFmpeg

Download from [ffmpeg.org](https://ffmpeg.org/)

#### Stream Video to RTSP Server

**macOS:**

```bash
# List available video devices
ffmpeg -f avfoundation -list_devices true -i ""

# Stream from device (replace "0" with your device ID)
ffmpeg -f avfoundation -framerate 30 -video_size 640x480 -i "0" \
  -vcodec libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://localhost:8554/mystream
```

**Windows:**

```bash
# List available video devices
ffmpeg -list_devices true -f dshow -i dummy

# Stream from device (replace "Device Name" with your camera name)
ffmpeg -f dshow -video_size 640x480 -framerate 30 -i video="Device Name" \
  -vcodec libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://localhost:8554/mystream
```

> **Note**: Windows and macOS use different video capture methods (`dshow` vs `avfoundation`)

#### Verify Stream with VLC

1. Open VLC Media Player
2. Go to **File → Open Network**
3. Enter your stream URL:
    - Local: `rtsp://localhost:8554/mystream`
    - With credentials: `rtsp://admin:admin@10.16.59.44:8554/live`
4. You should see the live stream

Reference screenshot: `images/vlc-screen.png`

### 2. GPU Configuration (Optional but Recommended)

GPU acceleration significantly improves performance. This project uses CUDA 12.5 and cuDNN 9.1, but other versions should work as well.

#### Install CUDA Toolkit 12.5

1. Download from [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. Select your system configuration
3. Run the installer (click Next through the installation)
4. Verify installation:

```bash
   nvcc --version
```

5. Check system environment variables - the CUDA `bin` path should be present

#### Install cuDNN 9.1

1. Download from [NVIDIA cuDNN Archive](https://developer.nvidia.com/cudnn-archive)
2. If Windows 10 isn't available, download the Windows 11 version
3. Run the installer
4. Copy cuDNN files to CUDA installation:
    - Navigate to `C:\Program Files\NVIDIA\CUDNN\v9.1\`
    - Copy files from `lib`, `include`, and `bin` folders
    - Paste into `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\` respective folders
5. Add cuDNN to system PATH:
    - Add `C:\Program Files\NVIDIA\CUDNN\v9.1\bin` to system environment variables

#### Verify Installation

```bash
nvcc --version
```

You should see CUDA version 12.5 displayed.

### 3. MongoDB Setup

#### Create Database Structure

1. Install and open **MongoDB Compass**
2. Create an organization (e.g., "Technical Hub")
3. Create a project (e.g., "FRS")
4. Create a cluster (e.g., "frs-cluster")
5. Create two databases:

    **Database 1: Face Embeddings Storage**

    - Database name: `frs-db`
    - Collection name: `embeddings`

    **Database 2: Attendance Records**

    - Database name: `frs-attendance`
    - Collection name: `attendance`

#### Configure Environment Variables

Create a `.env` file in the project root:

```env
MONGODB_URL="mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
DB_NAME="frs-db"
ATTENDANCE_COLLECTION_NAME="attendance"
EMBEDDING_COLLECTION_NAME="embeddings"
```

Replace with your actual MongoDB credentials and database names.

### 4. Python Environment Setup

#### Create Virtual Environment

**macOS/Linux:**

```bash
# Option 1 — Python venv
python -m venv .env
source .env/bin/activate

# Option 2 — uv:
uv venv .env --python 3.12
source .env/bin/activate
```

**Windows:**

```bash
python -m venv .env
.env\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

-   `opencv-python` - Computer vision
-   `insightface` - Face recognition model
-   `pymongo` - MongoDB driver
-   `numpy` - Numerical computing
-   `fastapi` - Web framework
-   `uvicorn` - ASGI server
-   `python-dotenv` - Environment variables
-   `torch` - PyTorch (required to prevent GPU bugs)
-   `onnxruntime-gpu` (Windows with CUDA) or `onnxruntime` (macOS/CPU)
-   `pyrealsense2` (Windows) or `pyrealsense2-macosx` (macOS) - For Intel RealSense cameras

> **Important**: Install `torch` even if you're using CPU-only mode to prevent potential GPU-related bugs.

#### Intel RealSense SDK (Windows Only)

If using Intel RealSense cameras on Windows, download and install the SDK:
[Intel RealSense SDK Releases](https://github.com/IntelRealSense/librealsense/releases)

## Project Structure

```
📦 frs
┣ 📂 attendance-system          # Main attendance tracking system
┃ ┣ 📂 cameras
┃ ┃ ┣ 📜 attendance-multi.py    # Multi-camera attendance tracking
┃ ┃ ┗ 📜 attendance-single.py   # Single-camera attendance tracking
┃ ┣ 📂 core
┃ ┃ ┗ 📜 attendance_db.py       # Database operations for attendance
┃ ┗ 📂 tools
┃   ┣ 📜 attendance_query.py    # Query attendance records
┃   ┗ 📜 coordinate_finder.py   # Find coordinates for entry/exit lines
┣ 📂 environment-mac             # macOS-specific files
┃ ┣ 📜 mediamtx                 # RTSP server executable
┃ ┗ 📜 mediamtx.yml             # RTSP server configuration
┣ 📂 images                      # Documentation images
┃ ┗ 📜 vlc-screen.png
┣ 📂 live-versions               # Real-time processing scripts
┃ ┣ 📜 browser-stream.py        # Web-based live stream viewer
┃ ┗ 📜 live-camera-mac.py       # Live camera processing (macOS)
┣ 📂 preview-versions            # Preview and testing scripts
┃ ┣ 📜 picture-preview-input.py # Test with single images
┃ ┣ 📜 video-preview-input.py   # Preview video with recognition
┃ ┗ 📜 video-process-input.py   # Process and output recognized video
┣ 📂 realsense-camera            # Intel RealSense camera scripts
┃ ┣ 📜 realsense-ffmpeg.py      # Stream RealSense to RTSP
┃ ┗ 📜 realsense-preview-test.py # Test RealSense camera
┣ 📂 setup                       # Initial setup files
┃ ┣ 📂 faces                    # Employee face images (named by employee code)
┃ ┃ ┣ 📜 1285.png
┃ ┃ ┣ 📜 1637.png
┃ ┃ ┗ 📜 ...
┃ ┣ 📜 add-face-embeddings.py   # Script to add faces to database
┃ ┗ 📜 all-employees.json       # Employee information
┣ 📜 .env                        # Environment variables (MongoDB, etc.)
┣ 📜 .gitignore
┗ 📜 README.md
```

### Directory Explanations

-   **attendance-system/**: Production attendance tracking code

    -   **cameras/**: Single and multi-camera attendance scripts
    -   **core/**: Core database operations
    -   **tools/**: Utility scripts for querying and configuration

-   **live-versions/**: Real-time recognition scripts (fastest performance)

-   **preview-versions/**: Testing scripts that process video/images with visual output (slower, for debugging)

-   **realsense-camera/**: Scripts specific to Intel RealSense depth cameras

-   **setup/**: Initial configuration and face database population

## How It Works

### Face Recognition Process

The system uses InsightFace for both face detection and face recognition:

1. **Face Detection**: Locates faces within video frames by identifying facial features and boundaries
2. **Face Recognition**: Extracts unique mathematical representations (embeddings/vectors) of each face, which act as a "facial fingerprint" for identification

### Workflow

1. **Initialization**:

    - Face embeddings for all employees are pre-computed and stored in MongoDB (`frs-db` database)
    - Each employee's face is converted into a 512-dimensional vector

2. **Video Capture**:

    - The system continuously captures frames from the camera source
    - Processes each frame in real-time (or near real-time depending on hardware)

3. **Face Detection**:

4. **Face Detection**:

    - Each frame is analyzed to detect the presence and location of faces
    - Bounding boxes are drawn around detected faces

5. **Face Recognition**:
   When a face is detected:

    - The system extracts facial features as a 512-dimensional vector
    - Compares this vector against all stored employee embeddings using cosine similarity
    - If similarity score exceeds the threshold (typically 0.4-0.6), the person is identified
    - Lower threshold = more false positives, Higher threshold = more false negatives

6. **Attendance Marking**:
    - Employees must cross a designated **entry line** to be marked "present"
    - Crossing an **exit line** marks them as "logged out"
    - All punch-in and punch-out times are recorded with employee codes in the `attendance` collection
    - The system prevents duplicate entries within a short time window

### Attendance Logic

-   **Entry Line**: Virtual line in the camera frame - crossing triggers attendance marking
-   **Exit Line**: Virtual line for marking logout (optional)
-   **Multiple Cameras**: Different cameras can handle different entry/exit points or monitor different areas
-   **Complete Records**: System logs every punch-in and punch-out with precise timestamps and employee codes
-   **Coordinate Finder**: Use `tools/coordinate_finder.py` to visually select line coordinates for your camera setup

## Usage

### 1. Add Employee Face Data

First, add employee face images and information:

1. Place employee photos in `setup/faces/` directory

    - Name files with employee code (e.g., `1285.png`, `6324.png`)
    - Use clear, front-facing photos with good lighting
    - One face per image

2. Update `setup/all-employees.json` with employee information:

```json
{
    "id": "1123",
    "name": "John Doe",
    "path": "./setup/faces/johndoe.png"
}
```

3. Run the embedding script:

```bash
   python setup/add-face-embeddings.py
```

This extracts face embeddings and stores them in MongoDB. You'll need to re-run this script whenever you add new employees.

### 2. Find Entry/Exit Line Coordinates

Use the coordinate finder tool to set up attendance lines:

```bash
python attendance-system/tools/coordinate_finder.py
```

This opens a window showing your camera feed. Click to set points for entry and exit lines. Note the coordinates for use in attendance scripts.

### 3. Test with Preview Versions

Preview versions process video/images but are slower (not real-time). Use these for testing:

**Test with images:**

```bash
python preview-versions/picture-preview-input.py
```

**Preview video with recognition:**

```bash
python preview-versions/video-preview-input.py
```

**Process and save recognized video:**

```bash
python preview-versions/video-process-input.py
```

### 4. Run Live Recognition

For real-time performance use the live versions:

**Browser-based viewer:**

```bash
python live-versions/browser-stream.py
```

Opens a local website (typically `http://localhost:8000`) showing the camera feed with recognized faces.

**Platform-specific live camera:**

macOS:

```bash
python live-versions/live-camera-mac.py
```

Windows: Use the equivalent Windows version with minimal code differences.

### 5. Run Attendance System

**Single camera setup:**

```bash
python -m attendance-system.cameras.attendance-single
```

**Multi-camera setup:**

```bash
python -m attendance-system.cameras.attendance-multi
```

Multi-camera setup allows:

-   Different cameras monitoring different entry/exit points
-   Redundancy for better coverage
-   Separate attendance zones

### 6. Query Attendance Records

```bash
python attendance-system/tools/attendance_query.py
```

This tool allows you to query and export attendance data from MongoDB.

### 7. Intel RealSense Camera (Optional)

If using Intel RealSense cameras:

**Test camera:**

```bash
python realsense-camera/realsense-preview-test.py
```

**Stream to RTSP:**

```bash
python realsense-camera/realsense-ffmpeg.py
```

> **Note**: Windows users must install the Intel RealSense SDK first.

## Model Selection

The system uses InsightFace models for face recognition. Two primary models are available:

### buffalo_l (Default)

-   **Pros**: Faster processing, lower computational requirements
-   **Cons**: Slightly lower accuracy
-   **Recommended for**: Real-time systems with limited hardware

### glintr100

-   **Pros**: Superior recognition accuracy, better with difficult angles/lighting
-   **Cons**: Higher computational cost, slower processing
-   **Recommended for**: High-accuracy requirements, powerful hardware available

To switch models, change the model name in the recognition scripts:

```python
# From:
app.prepare(ctx_id=0, det_size=(640, 640), model_name='buffalo_l')

# To:
app.prepare(ctx_id=0, det_size=(640, 640), model_name='glintr100')
```

## Technical Hub Integration

### Understanding NVR Systems

**NVR (Network Video Recorder)** is a device that records video from IP cameras over a network. In the Technical Hub deployment:

1. All security cameras are physically connected to the NVR
2. The NVR captures, processes, and stores video streams
3. Video can be viewed from a central server
4. **Challenge**: NVR and development systems are on separate networks

### Integration Options

#### Option 1: RTSP-Based Recognition (Easier but with Latency)

**How it works:**

1. Access camera feeds via RTSP links from the NVR
2. Process frames and perform face recognition
3. Send recognition data (JSON with coordinates and identities) back to the system
4. Hikvision NVR can overlay this data on the video stream before sending to monitors

**Pros:**

-   Easier to implement
-   Works across networks (with proper configuration)
-   No special SDK required

**Cons:**

-   RTSP protocol introduces latency (200-500ms typical)
-   Network-dependent performance
-   May struggle with real-time requirements

**Latency reduction strategies:**

-   Use lower resolution streams
-   Optimize network configuration
-   Implement frame skipping for processing
-   Use hardware acceleration

#### Option 2: Hikvision SDK Integration (Minimal Latency)

**How it works:**

1. Use Hikvision SDK to directly access camera streams
2. Process video with minimal overhead
3. Send recognition data to NVR
4. NVR overlays data and streams to displays

**Pros:**

-   Minimal latency (~50-100ms)
-   Direct camera access
-   More control over stream parameters

**Cons:**

-   **Windows only** - SDK not available for macOS/Linux
-   Must be written in **C++**
-   Requires **Inter-Process Communication (IPC)** with Python scripts
-   More complex implementation
-   **Must be on same LAN** as NVR

**Implementation requirements:**

-   C++ wrapper for Hikvision SDK
-   IPC mechanism (shared memory, named pipes, or sockets)
-   Python script for face recognition
-   Systems must be on the same local network

### Recommended Approach for Technical Hub

1. **Testing Phase**: Start with a dedicated test camera separate from the main NVR system
2. **Proof of Concept**: Implement Option 1 (RTSP) first to validate recognition accuracy
3. **Production Deployment**: If latency is acceptable, continue with RTSP; otherwise implement Option 2
4. **Network Configuration**: Work with IT to establish network connectivity between systems

## Jetson Compatibility

### Jetson Nano Limitations

The InsightFace model and Intel RealSense camera are **not compatible** with Jetson Nano for the following reasons:

#### 1. Insufficient Processing Power

-   Even with GPU capabilities, Jetson Nano is significantly less powerful than RTX 2050/3050 GPUs
-   Processing latency would be too high for real-time attendance tracking
-   Frame rates would drop below acceptable levels (likely <5 FPS with recognition)

#### 2. ARM Architecture Issues

-   Jetson Nano uses ARM-based CPU, not x86
-   Uses TensorRT CUDA instead of standard CUDA
-   Many Python packages (including InsightFace) are not compatible with TensorRT
-   While CUDA code can be compiled to TensorRT, the process is complex and time-consuming

#### 3. Legacy Software Support

-   Jetson Nano runs on an old version of JetPack
-   Limited community support and documentation
-   Many modern libraries have dropped support for older JetPack versions

### Alternative Solutions

If you must use Jetson or ARM-based systems:

-   Consider Jetson Xavier NX or AGX Xavier (more powerful)
-   Look into TensorRT-optimized models specifically designed for Jetson
-   Explore alternative face recognition libraries with ARM support
-   Accept significantly reduced performance and real-time capabilities

## Troubleshooting

### Common Issues

**GPU not being utilized:**

-   Ensure CUDA and cuDNN are properly installed
-   Verify `torch` is installed even if not explicitly used
-   Check that `onnxruntime-gpu` is installed (Windows)
-   Verify GPU drivers are up to date

**MongoDB connection errors:**

-   Check `.env` file configuration
-   Verify MongoDB cluster is running
-   Ensure IP whitelist includes your current IP (for MongoDB Atlas)
-   Test connection with MongoDB Compass

**Poor recognition accuracy:**

-   Improve lighting conditions
-   Use higher resolution camera
-   Ensure face images in database are clear and well-lit
-   Adjust similarity threshold
-   Consider switching to `glint360k` model

**High latency/low FPS:**

-   Reduce frame resolution
-   Use GPU acceleration
-   Switch to directly connected camera instead of RTSP
-   Close other GPU-intensive applications
-   Consider using frame skipping (process every Nth frame)

**Camera not detected:**

-   Verify camera permissions in OS settings
-   Check camera device ID with FFmpeg list command
-   Try different USB ports
-   Update camera drivers

**RealSense camera issues (Windows):**

-   Install Intel RealSense SDK
-   Check USB 3.0 connection (required for depth features)
-   Update RealSense firmware
