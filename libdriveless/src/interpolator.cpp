#include "../include/interpolator.h"
#include <stdexcept>
#include <algorithm>

typedef struct d2
{
    double x, y;
} d2;

std::vector<Waypoint> Interpolator::hermite(int width, int height, Waypoint p1, Waypoint p2)
{
    std::vector<Waypoint> curve;

    // int numPoints = 2 * abs(max(int(p2.z() - p1.z()), int(p2.x() - p1.x()), 100));

    double d = Waypoint::distanceBetween(p1, p2);
    int numPoints = TO_INT(d);
    // int numPoints = 100;
    curve.reserve(numPoints);

    float a1 = p1.heading().rad() - PI / 2;
    float a2 = p2.heading().rad() - PI / 2;

    // Tangent vectors
    d2 tan1 = {d * cosf(a1), d * sinf(a1)};
    d2 tan2 = {d * cosf(a2), d * sinf(a2)};

    int last_x = -1;
    int last_z = -1;

    for (int i = 0; i < numPoints; ++i)
    {
        double t = static_cast<double>(i) / (numPoints - 1);

        double t2 = t * t;
        double t3 = t2 * t;

        // Hermite basis functions
        double h00 = 2 * t3 - 3 * t2 + 1;
        double h10 = t3 - 2 * t2 + t;
        double h01 = -2 * t3 + 3 * t2;
        double h11 = t3 - t2;

        double x = h00 * p1.x() + h10 * tan1.x + h01 * p2.x() + h11 * tan2.x;
        double z = h00 * p1.z() + h10 * tan1.y + h01 * p2.z() + h11 * tan2.y;

        if (x < 0 || x >= width)
            continue;
        if (z < 0 || z >= height)
            continue;

        int cx = static_cast<int>(round(x));
        int cz = static_cast<int>(round(z));

        if (cx == last_x && cz == last_z)
            continue;
        if (cx < 0 || cx >= width)
            continue;
        if (cz < 0 || cz >= height)
            continue;

        double t00 = 6 * t2 - 6 * t;
        double t10 = 3 * t2 - 4 * t + 1;
        double t01 = -6 * t2 + 6 * t;
        double t11 = 3 * t2 - 2 * t;

        double ddx = t00 * p1.x() + t10 * tan1.x + t01 * p2.x() + t11 * tan2.x;
        double ddz = t00 * p1.z() + t10 * tan1.y + t01 * p2.z() + t11 * tan2.y;

        float heading = static_cast<float>(atan2f(ddz, ddx) + HALF_PI);

        // Interpolated point
        curve.push_back({cx, cz, angle::rad(heading)});
        last_x = cx;
        last_z = cz;
    }

    return curve;
}

static void catmull_roll_interpolate(std::vector<Waypoint> &points, Waypoint p0, Waypoint p1, Waypoint p2, Waypoint p3, int resolution)
{
    double inv_res = (double)1.0 / static_cast<double>(resolution);
    double t = 0;
    while (t < 1.0)
    {
        double t2 = t * t;
        double t3 = t2 * t;

        double x = 0.5 * ((2.0 * p1.x()) + (-p0.x() + p2.x()) * t +
                         (2.0 * p0.x() - 5.0 * p1.x() + 4.0 * p2.x() - p3.x()) * t2 +
                         (-p0.x() + 3.0 * p1.x() - 3.0 * p2.x() + p3.x()) * t3);
        double y = 0.5 * ((2.0 * p1.z()) + (-p0.z() + p2.z()) * t +
                         (2.0 * p0.z() - 5.0 * p1.z() + 4.0 * p2.z() - p3.z()) * t2 +
                         (-p0.z() + 3.0 * p1.z() - 3.0 * p2.z() + p3.z()) * t3);
        double dx = 0.5 * ((-p0.x() + p2.x()) +
                          2.0 * (2.0 * p0.x() - 5.0 * p1.x() + 4.0 * p2.x() - p3.x()) * t +
                          3.0 * (-p0.x() + 3 * p1.x() - 3.0 * p2.x() + p3.x()) * t2);
        double dy = 0.5 * ((-p0.z() + p2.z()) +
                          2.0 * (2.0 * p0.z() - 5.0 * p1.z() + 4.0 * p2.z() - p3.z()) * t +
                          3.0 * (-p0.z() + 3.0 * p1.z() - 3.0 * p2.z() + p3.z()) * t2);

        points.push_back(Waypoint(TO_INT(x), TO_INT(y), angle::rad(0.0 + HALF_PI - atan2(-dy, dx))));
        t += inv_res;
    }
}

// def catmull_roll_spline_interpolation(path: list[Waypoint], resolution: int = 10) -> list[Waypoint]:
//         padded = [path[0]] + path + [path[-1]]
//         for i in range(len(padded) - 3):
//             p0, p1, p2, p3 = padded[i:i+4]
//             new_points = catmull_roll_interpolate(p0, p1, p2, p3, resolution)
//             path.extend(new_points)
//         return path

std::vector<Waypoint> Interpolator::cubicSpline(std::vector<Waypoint> &dataPoints, int resolution)
{
    std::vector<Waypoint> res;
    const int size = dataPoints.size();
    if (size < 4)
        return std::vector<Waypoint>(dataPoints);

    //printf("[%d] res size: %d\n", 1, res.size());
    catmull_roll_interpolate(res,
                             dataPoints[0], dataPoints[0], dataPoints[1], dataPoints[2], resolution);

    //printf("[%d] res size: %d\n", 2, res.size());

    if (size >= 4)
    {
        for (int i = 0; i < (size - 3); i++)
        {
            const Waypoint p0 = dataPoints[i];
            const Waypoint p1 = dataPoints[i + 1];
            const Waypoint p2 = dataPoints[i + 2];
            const Waypoint p3 = dataPoints[i + 3];
            catmull_roll_interpolate(res, p0, p1, p2, p3, resolution);
            //printf("[loop %d] res size: %d\n", i, res.size());
        }
    }

    catmull_roll_interpolate(res,
                             dataPoints[size - 3], dataPoints[size - 2], dataPoints[size - 1], dataPoints[size - 1], resolution);

    //printf("[%d] res size: %d\n", 3, res.size());

    return res;

}

float *to_float_array(std::vector<Waypoint> res)
{
    const int size = res.size();
    //    printf("converting %d points to float array\n", size);
    float *points = new float[3 * size + 1];
    int i = 0;
    points[0] = static_cast<float>(size);
    for (auto p : res)
    {
        const int pos = (3 * i) + 1;
        points[pos] = static_cast<float>(p.x());
        points[pos + 1] = static_cast<float>(p.z());
        points[pos + 2] = p.heading().rad();
        i++;
    }
    //    printf("ptr address %x\n", points);
    //    printf("last 3 values in ptr: address %f, %f, %f\n", points[3 * size - 2], points[3 * size - 1], points[3 * size]);
    return points;
}
std::vector<Waypoint> from_float_array(float *arr)
{
    std::vector<Waypoint> res;
    if (arr == nullptr)
        return res;

    int count = static_cast<int>(arr[0]);
    res.reserve(count);

    for (int i = 0; i < count; i++)
    {
        const int pos = 3 * i + 1;
        res.push_back(Waypoint(static_cast<int>(arr[pos]), static_cast<int>(arr[pos + 1]), angle::rad(arr[pos + 2])));
    }
    return res;
}

extern "C"
{

    float *interpolate_hermite(int width, int height, int x1, int z1, float h1_rad, int x2, int z2, float h2_rad)
    {
        std::vector<Waypoint> res = Interpolator::hermite(width, height, Waypoint(x1, z1, angle::rad(h1_rad)), Waypoint(x2, z2, angle::rad(h2_rad)));
        return to_float_array(res);
    }

    float *interpolate_cubic_spline(float *arr, int resolution)
    {
        std::vector<Waypoint> points = from_float_array(arr);
        auto interpol_points = Interpolator::cubicSpline(points, resolution);
        return to_float_array(interpol_points);
    }

    void free_interpolation_arr(float *arr)
    {
        delete[] arr;
    }
}