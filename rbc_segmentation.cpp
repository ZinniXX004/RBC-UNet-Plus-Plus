/*
 * RBC Segmentation Inference
 * Using Manual NHWC Input to match TensorFlow Keras Model
 * Work command: py unet_plus_plus.py -> mkdir build -> cd build -> cmake -G "MinGW Makefiles" -DOpenCV_DIR="[opencv4 folder path]" .. -> mingw32 make -> rbc_seg.exe
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const int IMG_WIDTH = 256;
const int IMG_HEIGHT = 256;
const float CONFIDENCE_THRESHOLD = 0.5;

int main(int argc, char** argv) {
    
    // 1. Load ONNX Model, make sure that this file is in the same folder as .exe
    string modelPath = "E:/Mikroskop HETI ADB Deep Learning/U-Net++ for RBC/unet_plusplus_rbc.onnx";
    Net net;
    
    try {
        net = readNetFromONNX(modelPath);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        cout << "[INFO] Model loaded successfully!" << endl;
    } catch (const cv::Exception& e) {
        cerr << "[ERROR] Could not load model: " << e.what() << endl;
        return -1;
    }

    // 2. Load Image, make sure that this file is in the same folder as .exe 
    string imagePath = "E:/Mikroskop HETI ADB Deep Learning/U-Net++ for RBC/sample_blood_smear.jpg"; 
    Mat image = imread(imagePath);
    
    if (image.empty()) {
        cerr << "[ERROR] Could not read image: " << imagePath << endl;
        return -1;
    }

    // 3. Manual Preprocessing (NHWC Format)
    
    Mat resized, floatImage;
    
    // a. Resize
    resize(image, resized, Size(IMG_WIDTH, IMG_HEIGHT));
    
    // b. Convert BGR to RGB (Keras models usually expect RGB)
    cvtColor(resized, resized, COLOR_BGR2RGB);

    // c. Convert to Float32 and Normalize (0-1)
    resized.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // d. Wrap into 4D Tensor with NHWC Layout (1, 256, 256, 3) 
    int dimensions[4] = {1, IMG_HEIGHT, IMG_WIDTH, 3};
    Mat blob = Mat(4, dimensions, CV_32F, floatImage.data);

    // 4. Inference
    net.setInput(blob);
    
    cout << "[INFO] Running inference..." << endl;
    double t = (double)getTickCount();
    
    Mat output;
    try {
        // Output from U-Net++ usually comes as NHWC (1, 256, 256, 1) or NCHW depending on export
        output = net.forward(); 
    } catch (const cv::Exception& e) {
        cerr << "[ERROR] Inference failed: " << e.what() << endl;
        return -1;
    }
    
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "[INFO] Inference time: " << t << " seconds (" << 1.0/t << " FPS)" << endl;

    // 5. Post-Processing 
    Mat resultMask;
    
    // Check output dimensionality
    if (output.dims == 4) {
        // If NHWC format: (1, 256, 256, 1)
        if (output.size[3] == 1) {
             // Make 2D wrapper for that data
             Mat pred(output.size[1], output.size[2], CV_32F, output.ptr<float>());
             pred.copyTo(resultMask);
        }
        // If NHCW format: (1, 1, 256, 256)
        else if (output.size[1] == 1) {
             int sizes[] = {output.size[2], output.size[3]};
             Mat pred(2, sizes, CV_32F, output.ptr<float>());
             pred.copyTo(resultMask);
        }
    }

    if (resultMask.empty()) {
        cerr << "[ERROR] Unexpected output shape." << endl;
        return -1;
    }

    // Thresholding
    Mat binaryMask;
    threshold(resultMask, binaryMask, CONFIDENCE_THRESHOLD, 255, THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8U);
    
    // Resize mask back to original image size for visualization
    resize(binaryMask, binaryMask, image.size());

    // 6. Visualization
    imshow("Original", image);
    imshow("Segmentation Result", binaryMask);
    waitKey(0);

    return 0;
}
