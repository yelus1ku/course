#ifndef CAMERA_H
#define CAMERA_H
#include "rtweekend.h"
class camera {
public:
    camera(
        point3 lookfrom,
        point3 lookat,
        vec3 vup,
        double vfov, // vertical field-of-view in degrees
        double aspect_ratio,
        double aperture,
        double focus_dist,
        double _time0 = 0,
        double _time1 = 0
    ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 -
            focus_dist * w;
        lens_radius = aperture / 2;
        time0 = _time0;
        time1 = _time1;
    }
    ray get_ray(double s, double t) const {
        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin -
            offset,
            random_double(time0, time1)
        );

    }
private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    double lens_radius;
    double time0, time1; // shutter open/close times
};
/*
class camera {
public:
    camera(
        point3 lookfrom,      // 相机位置
        point3 lookat,        // 目标点
        vec3 vup,             // 垂直向量
        double vfov,          // 垂直视野角度
        double aspect_ratio   // 宽高比
    ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        auto w = unit_vector(lookfrom - lookat);      // 视线方向
        auto u = unit_vector(cross(vup, w));         // 水平基向量
        auto v = cross(w, u);                        // 垂直基向量

        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
    }

    ray get_ray(double s, double t) const {
        return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
    }

private:
    point3 origin;              // 相机位置
    point3 lower_left_corner;   // 视野左下角
    vec3 horizontal;            // 视野的水平范围
    vec3 vertical;              // 视野的垂直范围
};
*/
#endif