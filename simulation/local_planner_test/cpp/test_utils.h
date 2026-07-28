// test_utils.h
//
// C++ port of the python-only test helpers (TestConfig / TestUtils / TestTimer)
// used by test_executor.py. These classes are NOT part of libdriveless /
// FastRRT themselves (the python versions live purely under pydriveless's
// test helpers), so this header re-implements them on top of the real
// driveless/fastrrt C++ API so the rest of the executor can be a 1:1 port.
#pragma once

#include <driveless/angle.h>
#include <driveless/waypoint.h>
#include <driveless/map_pose.h>
#include <driveless/world_pose.h>
#include <driveless/search_params.h>
#include <driveless/search_frame.h>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <chrono>
#include <map>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstring>

using json = nlohmann::json;

enum FrameFileType
{
    FILE_TYPE_PFM = 1,
    FILE_TYPE_RGB = 2
};

// -----------------------------------------------------------------------
// TestConfig: mirrors the python TestConfig plain data object
// -----------------------------------------------------------------------
struct TestConfig
{
    Waypoint start{0, 0, angle::deg(0)};
    Waypoint goal{0, 0, angle::deg(0)};
    FrameFileType file_type = FILE_TYPE_RGB;
    cv::Mat raw_frame; // CV_32FC3 (pfm) or CV_8UC3 (rgb)

    std::vector<float> segmentation_costs;
    std::vector<std::tuple<int, int, int>> segmentation_colors;

    std::pair<int, int> lower_bound{0, 0};
    std::pair<int, int> upper_bound{0, 0};
    std::pair<float, float> og_real_size{0.f, 0.f};

    float max_curvature = 0.f;
    float max_steering_angle_deg = 0.f;
    float vehicle_length_m = 0.f;

    std::pair<float, float> meters_to_pixel_ratio{1.f, 1.f};
    std::pair<float, float> pixel_to_meters_ratio{1.f, 1.f};

    WorldPose world_origin{angle::rad(0), angle::rad(0), 0, angle::rad(0)};
    std::pair<int, int> min_dist{0, 0};
};

// -----------------------------------------------------------------------
// TestTimer: mirrors python TestTimer (simple named stopwatch helper)
// -----------------------------------------------------------------------
class TestTimer
{
    static inline std::map<std::string, std::chrono::high_resolution_clock::time_point> _start_time;

public:
    static void exec_start(const std::string &key = "default")
    {
        _start_time[key] = std::chrono::high_resolution_clock::now();
    }

    // Returns elapsed seconds, or -1 if exec_start() was never called for this key
    static double exec_stop(const std::string &key = "default")
    {
        auto it = _start_time.find(key);
        if (it == _start_time.end())
            return -1.0;

        auto start = it->second;
        _start_time.erase(it);
        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double>(end - start).count();
        std::cout << "[exec_" << key << "] " << (1000.0 * execution_time) << " ms" << std::endl;
        return execution_time;
    }
};

// -----------------------------------------------------------------------
// TestUtils: mirrors python TestUtils (scenario loading + debug export)
// -----------------------------------------------------------------------
class TestUtils
{
private:
    // ---- PFM reader, equivalent of python's __read_pfm/__convert_pfm ----
    static cv::Mat __read_pfm(const std::string &file_path)
    {
        std::ifstream f(file_path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Could not open PFM file: " + file_path);

        std::string header;
        std::getline(f, header);
        bool color;
        if (header == "PF")
            color = true;
        else if (header == "Pf")
            color = false;
        else
            throw std::runtime_error("Not a PFM file.");

        std::string dims_line;
        while (std::getline(f, dims_line))
        {
            if (!dims_line.empty() && dims_line[0] == '#')
                continue;
            break;
        }
        int width, height;
        {
            std::istringstream iss(dims_line);
            iss >> width >> height;
        }

        std::string scale_line;
        std::getline(f, scale_line);
        double scale = std::stod(scale_line);
        bool little_endian = scale < 0;
        scale = std::fabs(scale);

        int channels = color ? 3 : 1;
        std::vector<float> data(static_cast<size_t>(width) * height * channels);
        f.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));

        if (!little_endian)
        {
            // swap byte order if the file is big-endian and host is little-endian
            for (auto &v : data)
            {
                char *b = reinterpret_cast<char *>(&v);
                std::swap(b[0], b[3]);
                std::swap(b[1], b[2]);
            }
        }

        cv::Mat mat(height, width, color ? CV_32FC3 : CV_32FC1);
        std::memcpy(mat.data, data.data(), data.size() * sizeof(float));
        cv::flip(mat, mat, 0); // PFM stores pixels bottom to top
        return mat;
    }

    static cv::Mat __convert_pfm(const cv::Mat &raw)
    {
        cv::Mat new_frame(raw.rows, raw.cols, CV_32FC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < raw.rows; i++)
        {
            for (int j = 0; j < raw.cols; j++)
            {
                float v = raw.at<float>(i, j);
                if (std::isfinite(v))
                {
                    new_frame.at<cv::Vec3f>(i, j) = cv::Vec3f(1.0f, 255.f * v / 0.75f, 0.f);
                }
                else
                {
                    new_frame.at<cv::Vec3f>(i, j) = cv::Vec3f(0.f, 0.f, 0.f);
                }
            }
        }
        return new_frame;
    }

    static void __draw_arrow(cv::Mat &frame, int row, int col, double heading_deg,
                              cv::Scalar color = cv::Scalar(0, 0, 255), int thickness = 2, int length = 20)
    {
        double rad = heading_deg * CV_PI / 180.0;
        int dx = static_cast<int>(std::cos(rad) * length);
        int dy = static_cast<int>(-std::sin(rad) * length);
        cv::Point start(col, row);
        cv::Point end(col + dx, row + dy);
        cv::arrowedLine(frame, start, end, color, thickness, cv::LINE_8, 0, 0.2);
    }

public:
    // ---- read_config: loads scenarios/<name>_cfg.json (+ .pfm or .png) ----
    static TestConfig read_config(const std::string &scenario_name)
    {
        std::string json_file = "scenarios/" + scenario_name + "_cfg.json";
        std::ifstream jf(json_file);
        if (!jf.is_open())
            throw std::runtime_error("Could not open config file: " + json_file);

        json raw_config;
        jf >> raw_config;

        TestConfig config;
        config.start = Waypoint(raw_config["start"][0].get<int>(),
                                 raw_config["start"][1].get<int>(),
                                 angle::deg(raw_config["start"][2].get<double>()));
        config.goal = Waypoint(raw_config["goal"][0].get<int>(),
                                raw_config["goal"][1].get<int>(),
                                angle::deg(raw_config["goal"][2].get<double>()));

        std::string pfm_file = "scenarios/" + scenario_name + ".pfm";
        std::string png_file = "scenarios/" + scenario_name + ".png";
        std::ifstream test_pfm(pfm_file);
        if (test_pfm.good())
        {
            config.raw_frame = __convert_pfm(__read_pfm(pfm_file));
            config.file_type = FILE_TYPE_PFM;
        }
        else
        {
            std::ifstream test_png(png_file);
            if (test_png.good())
                config.raw_frame = cv::imread(png_file, cv::IMREAD_COLOR);
            config.file_type = FILE_TYPE_RGB;
        }

        for (auto &c : raw_config["segmentation_costs"])
            config.segmentation_costs.push_back(c.get<float>());

        for (auto &c : raw_config["segmentation_colors"])
            config.segmentation_colors.push_back(
                std::make_tuple(c[0].get<int>(), c[1].get<int>(), c[2].get<int>()));

        config.lower_bound = {raw_config["lower_bound"][0].get<int>(), raw_config["lower_bound"][1].get<int>()};
        config.upper_bound = {raw_config["upper_bound"][0].get<int>(), raw_config["upper_bound"][1].get<int>()};
        config.og_real_size = {raw_config["og_real_size"][0].get<float>(), raw_config["og_real_size"][1].get<float>()};
        config.max_curvature = raw_config["max_curvature"].get<float>();
        config.max_steering_angle_deg = raw_config["max_steering_angle_deg"].get<float>();
        config.vehicle_length_m = raw_config["vehicle_length_m"].get<float>();
        config.meters_to_pixel_ratio = {raw_config["meters_to_pixel_ratio"][0].get<float>(),
                                         raw_config["meters_to_pixel_ratio"][1].get<float>()};
        config.pixel_to_meters_ratio = {raw_config["pixel_to_meters_ratio"][0].get<float>(),
                                         raw_config["pixel_to_meters_ratio"][1].get<float>()};
        config.min_dist = {raw_config["min_distance"][0].get<int>(), raw_config["min_distance"][1].get<int>()};

        auto &wo = raw_config["world_origin"];
        config.world_origin = WorldPose(
            angle::deg(wo[0].get<double>()),
            angle::deg(wo[1].get<double>()),
            wo[2].get<double>(),
            angle::deg(wo[3].get<double>()));

        return config;
    }

    // ---- export_color_frame: builds a debug frame with start/goal arrows ----
    // Pass an empty string for `file` to only get the cv::Mat back (no write to disk).
    static cv::Mat export_color_frame(TestConfig &conf, const std::string &file)
    {
        SearchFrame f(conf.raw_frame.cols, conf.raw_frame.rows, {-1, -1}, {-1, -1});

        std::vector<float> raw(static_cast<size_t>(conf.raw_frame.cols) * conf.raw_frame.rows * 3);
        // raw_frame may be CV_32FC3 (pfm) or CV_8UC3 (rgb) -> normalize to float*
        cv::Mat float_frame;
        conf.raw_frame.convertTo(float_frame, CV_32FC3);
        std::memcpy(raw.data(), float_frame.data, raw.size() * sizeof(float));

        f.copyFrom(raw.data());
        f.setClassCosts(conf.segmentation_costs);
        f.setClassColors(conf.segmentation_colors);

        std::vector<uchar> dest(static_cast<size_t>(f.width()) * f.height() * 3);
        f.exportToColorFrame(dest.data());
        cv::Mat color_frame(f.height(), f.width(), CV_8UC3, dest.data());
        color_frame = color_frame.clone(); // own the memory before `dest` goes out of scope

        __draw_arrow(color_frame, conf.start.z(), conf.start.x(), 90 - conf.start.heading().deg());
        __draw_arrow(color_frame, conf.goal.z(), conf.goal.x(), 90 - conf.goal.heading().deg(), cv::Scalar(128, 30, 128));

        if (!file.empty())
            cv::imwrite(file, color_frame);

        return color_frame;
    }

    static void export_planner_result(TestConfig &conf, const std::string &file, std::vector<Waypoint> &path)
    {
        cv::Mat color_frame = export_color_frame(conf, "");
        for (auto &p : path)
            color_frame.at<cv::Vec3b>(p.z(), p.x()) = cv::Vec3b(255, 0, 0);
        cv::imwrite(file, color_frame);
    }

    static void export_frame_planner_result(TestConfig &conf, SearchFrame &frame, const std::string &file,
                                             std::vector<Waypoint> &path, cv::Scalar path_color = cv::Scalar(255, 0, 0))
    {
        std::vector<uchar> dest(static_cast<size_t>(frame.width()) * frame.height() * 3);
        frame.exportToColorFrame(dest.data());
        cv::Mat color_frame(frame.height(), frame.width(), CV_8UC3, dest.data());
        color_frame = color_frame.clone();

        __draw_arrow(color_frame, conf.start.z(), conf.start.x(), 90 - conf.start.heading().deg());
        __draw_arrow(color_frame, conf.goal.z(), conf.goal.x(), 90 - conf.goal.heading().deg(), cv::Scalar(128, 30, 128));

        for (int h = 0; h < color_frame.rows; h++)
        {
            for (int w = 0; w < color_frame.cols; w++)
            {
                int t = frame.getTraversability(w, h);
                if (t & 0x100)
                    color_frame.at<cv::Vec3b>(h, w) = cv::Vec3b(128, 128, 128);
            }
        }

        for (auto &p : path)
            color_frame.at<cv::Vec3b>(p.z(), p.x()) =
                cv::Vec3b(static_cast<uchar>(path_color[0]), static_cast<uchar>(path_color[1]), static_cast<uchar>(path_color[2]));

        cv::imwrite(file, color_frame);
    }

    // ---- builders equivalent to python's build_cuda_frame / build_ego_params / build_search_params ----
    static SearchFrame *build_cuda_frame(TestConfig &conf)
    {
        SearchFrame *f = new SearchFrame(conf.raw_frame.cols, conf.raw_frame.rows, conf.lower_bound, conf.upper_bound);

        std::vector<float> raw(static_cast<size_t>(conf.raw_frame.cols) * conf.raw_frame.rows * 3);
        cv::Mat float_frame;
        conf.raw_frame.convertTo(float_frame, CV_32FC3);
        std::memcpy(raw.data(), float_frame.data, raw.size() * sizeof(float));

        f->copyFrom(raw.data());
        f->setClassCosts(conf.segmentation_costs);
        f->setClassColors(conf.segmentation_colors);
        return f;
    }

    static EgoParams build_ego_params(TestConfig &conf)
    {
        return EgoParams::init(conf.raw_frame.cols, conf.raw_frame.rows)
            .withSearchPhysicalSize(conf.og_real_size.first, conf.og_real_size.second)
            .withEgoUpperBound(conf.upper_bound)
            .withEgoLowerBound(conf.lower_bound)
            .withMaxCurvature(conf.max_curvature)
            .withMaxSteeringAngle(angle::deg(conf.max_steering_angle_deg))
            .withVehicleLength(conf.vehicle_length_m)
            .withSegmentationClassColors(conf.segmentation_colors)
            .withSegmentationClassCosts(conf.segmentation_costs)
            .withWorldOrigin(conf.world_origin)
            .build();
    }

    static SearchParams build_search_params(TestConfig &conf, bool gpu, int timeout = 60000)
    {
        SearchFrame *frame = build_cuda_frame(conf); // CPU frame path intentionally omitted (gpu=true in test_executor.py)
        (void)gpu;

        return SearchParams::init(conf.start, conf.goal)
            .withWorldOrigin(conf.world_origin)
            .withDistanceToGoalTolerance(15.0f)
            .withVelocity(1.0f)
            .withMapOrigin(MapPose(0, 0, 0, angle::rad(0)))
            .withEgoPose(MapPose(0, 0, 0, conf.start.heading()))
            .withHeadingErrorTolerance(angle::deg(5))
            .withTimeout(timeout)
            .withMaxPathSize(200.0f)
            .withMinDistance(conf.min_dist)
            .withFrame(frame)
            .build();
    }

    static void save_path(std::vector<Waypoint> &path, const std::string &file)
    {
        std::ofstream f(file);
        for (auto &p : path)
            f << "x=" << p.x() << ", z=" << p.z() << ", heading_deg=" << p.heading().deg() << "\n";
    }
};
