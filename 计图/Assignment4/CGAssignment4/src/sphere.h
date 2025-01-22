#ifndef SPHERE_H
#define SPHERE_H
#include "hittable.h"
#include "vec3.h"

// 球体类，继承自 hittable
class sphere : public hittable {
public:
    sphere() {}
    sphere(point3 cen, double r, shared_ptr<material> m)
        : center(cen), radius(r), mat_ptr(m) {};

    // 判断射线是否与球体发生碰撞
    virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
    virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

public:
    point3 center;  // 球心
    double radius;  // 半径
    shared_ptr<material> mat_ptr;
private:
    static void get_sphere_uv(const point3& p, double& u, double& v) {
    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + pi;
    u = phi / (2 * pi);
    v = theta / pi;
    }
};

// 判断射线是否与球体碰撞
bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();  // 射线方向的模平方
    auto half_b = dot(oc, r.direction());     // 射线起点到球心的投影长度
    auto c = oc.length_squared() - radius * radius;  // 距离平方减半径平方
    auto discriminant = half_b * half_b - a * c;     // 判别式

    if (discriminant < 0) return false;  // 判别式小于 0，表示没有交点
    auto sqrtd = sqrt(discriminant);

    // 找到满足条件的最近根
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;  // 记录 t
    rec.p = r.at(rec.t);  // 计算碰撞点
    vec3 outward_normal = (rec.p - center) / radius;  // 计算碰撞点法向量
    rec.set_face_normal(r, outward_normal);  // 确定法向量方向
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;
    return true;
}
bool sphere::bounding_box(double time0, double time1, aabb& output_box) const {
    output_box = aabb(
        center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));
    return true;
}
#endif
