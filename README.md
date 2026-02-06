\# RBC Segmentation using U-Net++ for Digital Microscopy



!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

!\[Platform](https://img.shields.io/badge/Platform-Windows%20(MSYS2)%20%7C%20Embedded-blue)

!\[Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green)



This project implements a Deep Learning pipeline for segmenting Red Blood Cells (RBC) from blood smear images to assist in the diagnosis of Iron Deficiency Anemia (IDA). It features a hybrid architecture:

1\.  \*\*Python (TensorFlow/Keras):\*\* Used for training the U-Net++ model and exporting it to ONNX.

2\.  \*\*C++ (OpenCV DNN):\*\* A lightweight inference engine designed for deployment on embedded devices (e.g., Raspberry Pi) or Windows via MSYS2.



\## 📂 Repository Structure



```text

.

├── build/                  # Directory for compiled executables (ignored by git)

├── rbc\_segmentation.cpp    # C++ Source Code for Real-time Inference

├── unet\_plus\_plus\_rbc.py   # Python Script for Model Training \& ONNX Export

├── unet\_plusplus\_rbc.onnx  # Pre-trained Model (Standard NHWC Layout)

├── CMakeLists.txt          # CMake Configuration for C++ Build

├── sample\_blood\_smear.jpg  # Sample Image for Testing

└── README.md               # Project Documentation

```



\## 🛠️ Prerequisites

1\. \*\*Python Environment (Training and Export)\*\*



Ensure you have Python 3.x installed.

\*\*Critical\*\*: You must use ```numpy<2.0``` to avoid compatibility issues with ```tf2onnx```.



```

pip install tensorflow tf2onnx "numpy<2.0" opencv-python

```



\*\*2. C++ Environment (Windows MSYS2)\*\*



This project relies on MSYS2 UCRT64 for a unified toolchain (GCC + OpenCV + Protobuf).



Open your MSYS2 UCRT64 terminal and install the dependencies:



```

pacman -S mingw-w64-ucrt-x86\_64-toolchain

pacman -S mingw-w64-ucrt-x86\_64-cmake

pacman -S mingw-w64-ucrt-x86\_64-opencv

pacman -S mingw-w64-ucrt-x86\_64-protobuf

```



\## 🚀 Usage Guide

\### Step 1: Train and Export Model (Python)



Run the Python script to build the U-Net++ architecture and export the trained model to ONNX format.



```

python unet\_plus\_plus\_rbc.py

```

\*Output\*: This will generate unet\_plusplus\_rbc.onnx in your project root.



\### Step 2: Compile C++ Inference Engine



We use CMake to build the C++ application. Ensure you are using the MinGW generator.



1\. Create a build directory:

```

mkdir build

cd build

```



2\. Configure CMake (Point to your MSYS2 library path if needed):

```

cmake -G "MinGW Makefiles" -DOpenCV\_DIR="D:/msys64/ucrt64/lib/cmake/opencv4" ..

```



3\. Compile:

```

mingw32-make

```



\### Step 3: Run Inference



Before running, ensure unet\_plusplus\_rbc.onnx and sample\_blood\_smear.jpg are present in the build/ directory (or copy them there).



\## ⚠️ Technical Notes



\* Data Layout: The ONNX model is exported in TensorFlow's standard NHWC format (Batch, Height, Width, Channels).

\* C++ Preprocessing: The C++ script manually constructs the input tensor to match the NHWC format. Standard cv::dnn::blobFromImage is avoided because it forces NCHW layout, which causes channel mismatch errors with Keras-exported models.



\## 📜 License



MIT License

