#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
#include "hittable.h"
#include "aabb.h"
#include <memory>
#include <vector>
using std::shared_ptr;
using std::make_shared;

// 可碰撞物体列表类，继承自 hittable
class hittable_list : public hittable {
public:
    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    // 清空物体列表
    void clear() { objects.clear(); }

    // 添加物体到列表
    void add(shared_ptr<hittable> object) { objects.push_back(object); }

    // 检测射线是否与列表中的任意物体发生碰撞
    virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
    virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

public:
    std::vector<shared_ptr<hittable>> objects;  // 存储物体的共享指针列表
};

// 判断射线是否与列表中的任意物体发生碰撞
bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;          // 是否发生碰撞
    auto closest_so_far = t_max;        // 记录当前最近的碰撞点

    for (const auto& object : objects) {
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t; // 更新最近碰撞点
            rec = temp_rec;              // 更新碰撞记录
        }
    }
    return hit_anything;
}
bool hittable_list::bounding_box(double time0, double time1, aabb& output_box)
const {
    if (objects.empty()) return false;
    aabb temp_box;
    bool first_box = true;
    for (const auto& object : objects) {
        if (!object->bounding_box(time0, time1, temp_box)) return false;
        output_box = first_box ? temp_box : surrounding_box(output_box,
            temp_box);
        first_box = false;
    }
    return true;
}
#endif
