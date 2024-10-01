

typedef unsigned char uchar;

#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/cuda_frame.h"
#include <memory>

float *readPNGToFloatArray(const char *filename, int &width, int &height, int &channels)
{
    // Read the image using OpenCV
    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);

    if (image.empty())
    {
        std::cerr << "Error loading image: " << filename << std::endl;
        return nullptr;
    }

    width = image.cols;
    height = image.rows;
    channels = image.channels();

    // Allocate memory for the float array
    float *float_data = new float[width * height * channels];

    // Convert pixel data to float
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                float_data[(y * width + x) * channels + c] = (float)image.at<cv::Vec3b>(y, x)[c];
            }
        }
    }

    return float_data;
}
void writeRGBToPNG(const unsigned char *data, int width, int height, const std::string &filename)
{
    // Create a Mat object to hold the image data
    cv::Mat image(height, width, CV_8UC3);

    // Copy the RGB data to the Mat object
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                image.at<cv::Vec3b>(y, x)[c] = data[(y * width + x) * 3 + c];
            }
        }
    }

    // Write the image to a PNG file
    cv::imwrite(filename, image);
}

bool convert_image(const char *file, const char *outp_file)
{

    int width, height, channels;
    float *img = readPNGToFloatArray(file, width, height, channels);
    if (!img)
    {
        fprintf(stderr, "unable to read %s\n", file);
        return false;
    }
    u_char *res = new u_char[sizeof(float) * channels * height * width];

    CudaFrame f(img, width, height, 0, 0, 0, 0, 0, 0);
    f.convertToColorFrame(res);
    writeRGBToPNG(res, width, height, outp_file);

    delete []img;
    delete []res;
    return true;
}


// bool test_set_goal(const char *file)
// {
//     int width, height, channels;
//     float *img = readPNGToFloatArray(file, width, height, channels);
//     if (!img)
//     {
//         fprintf(stderr, "unable to read %s\n", file);
//         return false;
//     }
//     auto f = std::make_unique<CudaFrame>(img, width, height, 0, 0, 119, 148, 137, 108);

//     int goal_x = 0;
//     int goal_z = 0;

//     float *img2 = new float[height * width * 3];

//     f->setGoal(goal_x, goal_z);
//     f->copyBack(img2);

//     for (int i = 0; i < height; i++)
//         for (int j = 0; j < width; j++)
//         {
//             float dz = (i - goal_z);
//             float dx = (j - goal_x);
//             auto expected = sqrt( dx * dx + dz * dz);
//             auto obtained = img2[3 * (i * width + j) + 1];

//             if (expected != obtained)
//             {
//                 printf("computeEuclidianCostToGoal values mismatch at (x,z) %d, %d:  expected: %f obtainded %f\n", j, i, expected, obtained);
//                 return false;
//             }
//         }

//     delete []img2;
//     delete []img;

//     return true;
// }


int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "please use %s <image.png>", argv[0]);
        return 1;
    }

    if (!convert_image(argv[1], "converted.png")) 
        return 1;
    
    // if (!test_set_goal("img_0.png"))
    // {
    //     return -1;
    // }
    return 0;
}