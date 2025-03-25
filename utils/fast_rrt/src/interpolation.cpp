#include "../include/fastrrt.h"
#include "../include/cuda_params.h"
#include "../include/math_utils.h"
#include "../include/waypoint.h"
#include <bits/algorithmfwd.h>

// extern std::vector<Waypoint> drawKinematicPath(double *physicalParams, int width,  int height, int2 center, int2 start, float start_heading, Waypoint end, float velocity_m_s);


// std::vector<Waypoint> interpolateKinematics(std::vector<Waypoint>& path, double *physicalParams, int width, int height, int2 center, float velocity_m_s) {
//     int2 start = {path[0].x(), path[0].z()};
//     float curr_heading = path[0].heading().rad();

//     std::vector<Waypoint> res;

//     for (int i = 1; i < path.size(); i++) {
//         printf ("start: %d, %d, %f\n", start.x, start.y, curr_heading);
//         std::vector<Waypoint> subpath = drawKinematicPath(physicalParams, width, height, center, start, curr_heading, path[i], velocity_m_s);
//         printf("subpath size: %ld\n", subpath.size());
//         res.insert(res.end(), subpath.begin(), subpath.end());
        
//         Waypoint last = subpath.at(subpath.size() - 1);
//         start.x = last.x();
//         start.y = last.z();
//         curr_heading = last.heading().rad();

//     }
//     return res;
// }


Waypoint evaluateBezier(Waypoint& P0, Waypoint& P1, Waypoint& P2, Waypoint& P3, double t) {
    double x = std::pow(1 - t, 3) * P0.x() + 3 * std::pow(1 - t, 2) * t * P1.x() + 
               3 * (1 - t) * std::pow(t, 2) * P2.x() + std::pow(t, 3) * P3.x();

    double z = std::pow(1 - t, 3) * P0.z() + 3 * std::pow(1 - t, 2) * t * P1.z() + 
               3 * (1 - t) * std::pow(t, 2) * P2.z() + std::pow(t, 3) * P3.z();

    return Waypoint(x, z, angle::rad(0));
}

std::vector<Waypoint> interpolate(std::vector<Waypoint>& controlPoints, int width, int height) {
    int resolution = 64;
    std::vector<Waypoint> interpolatedPoints;
    
    if (controlPoints.size() < 4) {
        printf("At least 4 control points are required for cubic Bezier interpolation.\n");
        return {};
    }

    for (size_t i = 0; i + 3 < controlPoints.size(); i += 3) {
        for (int j = 0; j <= resolution; ++j) {
            double t = static_cast<double>(j) / resolution;
            Waypoint p = evaluateBezier(controlPoints[i], controlPoints[i + 1], 
                controlPoints[i + 2], controlPoints[i + 3], t);

            if (p.x() < 0 || p.x() > width) continue;
            if (p.z() < 0 || p.z() > height) continue;
            interpolatedPoints.push_back(p);
        }
    }

    return interpolatedPoints;
}