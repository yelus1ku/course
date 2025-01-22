#ifndef SPHERE_H
#define SPHERE_H
#include "hittable.h"
#include "vec3.h"

// �����࣬�̳��� hittable
class sphere : public hittable {
public:
    sphere() {}
    sphere(point3 cen, double r, shared_ptr<material> m)
        : center(cen), radius(r), mat_ptr(m) {};

    // �ж������Ƿ������巢����ײ
    virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
    virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

public:
    point3 center;  // ����
    double radius;  // �뾶
    shared_ptr<material> mat_ptr;
private:
    static void get_sphere_uv(const point3& p, double& u, double& v) {
    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + pi;
    u = phi / (2 * pi);
    v = theta / pi;
    }
};

// �ж������Ƿ���������ײ
bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();  // ���߷����ģƽ��
    auto half_b = dot(oc, r.direction());     // ������㵽���ĵ�ͶӰ����
    auto c = oc.length_squared() - radius * radius;  // ����ƽ�����뾶ƽ��
    auto discriminant = half_b * half_b - a * c;     // �б�ʽ

    if (discriminant < 0) return false;  // �б�ʽС�� 0����ʾû�н���
    auto sqrtd = sqrt(discriminant);

    // �ҵ����������������
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;  // ��¼ t
    rec.p = r.at(rec.t);  // ������ײ��
    vec3 outward_normal = (rec.p - center) / radius;  // ������ײ�㷨����
    rec.set_face_normal(r, outward_normal);  // ȷ������������
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
