﻿/*The MIT License (MIT)

Copyright (c) 2021-Present, Wencong Yang (yangwc3@mail2.sysu.edu.cn).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.*/

#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <random>
#include "texture.h"
#include "moving_sphere.h"
#include "WindowsApp.h"
#include "vec3.h"
#include "ray.h"
#include "bvh.h"
#include "rtweekend.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "aarect.h"
#include "box.h"
#include "constant_medium.h"
double hit_sphere(const point3& center, double radius, const ray& r) {
	vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) {
		return -1.0;
	}
	else {
		return (-half_b - sqrt(discriminant)) / a;
	}
}
//基础任务二
/*
color ray_color(const ray& r, const hittable& world) {
	hit_record rec;
	if (world.hit(r, 0, infinity, rec)) {
		return 0.5 * (rec.normal + color(1, 1, 1));
	}
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}
*/
/*
//基础任务
color ray_color(const ray& r, const hittable& world, int depth) {
	hit_record rec;
	// If we've exceeded the ray bounce limit, no more light is gathered.
	if (depth <= 0)
		return color(0, 0, 0);
	if (world.hit(r, 0.001, infinity, rec)) {
		ray scattered;
		color attenuation;
		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return attenuation * ray_color(scattered, world, depth - 1);
		return color(0, 0, 0);
	}
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}
*/
color ray_color(const ray& r, const color& background, const hittable& world,
	int depth) {
	hit_record rec;
	// If we've exceeded the ray bounce limit, no more light is gathered.
	if (depth <= 0)
		return color(0, 0, 0);
	// If the ray hits nothing, return the background color.
	if (!world.hit(r, 0.001, infinity, rec))
		return background;
	ray scattered;
	color attenuation;
	color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
	if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
		return emitted;
	return emitted + attenuation * ray_color(scattered, background, world,
		depth - 1);
}
static std::vector<std::vector<color>> gCanvas;		//Canvas

// The width and height of the screen
auto aspect_ratio = 1.0;
int gWidth = 800;
const int gHeight = static_cast<int>(gWidth / aspect_ratio);

void rendering();

int main(int argc, char* args[])
{
	// Create window app handle
	WindowsApp::ptr winApp = WindowsApp::getInstance(gWidth, gHeight, "CGAssignment4: Ray Tracing");
	if (winApp == nullptr)
	{
		std::cerr << "Error: failed to create a window handler" << std::endl;
		return -1;
	}

	// Memory allocation for canvas
	gCanvas.resize(gHeight, std::vector<color>(gWidth));

	// Launch the rendering thread
	// Note: we run the rendering task in another thread to avoid GUI blocking
	std::thread renderingThread(rendering);

	// Window app loop
	while (!winApp->shouldWindowClose())
	{
		// Process event
		winApp->processEvent();

		// Display to the screen
		winApp->updateScreenSurface(gCanvas);

	}

	renderingThread.join();

	return 0;
}
/*
void write_color(int x, int y, color pixel_color)
{
	// Out-of-range detection
	if (x < 0 || x >= gWidth)
	{
		std::cerr << "Warnning: try to write the pixel out of range: (x,y) -> (" << x << "," << y << ")" << std::endl;
		return;
	}

	if (y < 0 || y >= gHeight)
	{
		std::cerr << "Warnning: try to write the pixel out of range: (x,y) -> (" << x << "," << y << ")" << std::endl;
		return;
	}

	// Note: x -> the column number, y -> the row number
	gCanvas[y][x] = pixel_color;

}
*/
void write_color(int x, int y, std::ostream& out, color pixel_color, int samples_per_pixel) {
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();
	// Divide the color by the number of samples.
	auto scale = 1.0 / samples_per_pixel;
	r *= scale;
	g *= scale;
	b *= scale;
	r = sqrt(r);
	g = sqrt(g);
	b = sqrt(b);
	r = clamp(r, 0.0, 0.999);
	g = clamp(g, 0.0, 0.999);
	b = clamp(b, 0.0, 0.999);
	color tmp(r, g, b);
	gCanvas[y][x] = tmp;
}

hittable_list final_scene() {
	hittable_list boxes1;
	auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));
	const int boxes_per_side = 20;
	for (int i = 0; i < boxes_per_side; i++) {
		for (int j = 0; j < boxes_per_side; j++) {
			auto w = 100.0;
			auto x0 = -1000.0 + i * w;
			auto z0 = -1000.0 + j * w;
			auto y0 = 0.0;
			auto x1 = x0 + w;
			auto y1 = random_double(1, 101);
			auto z1 = z0 + w;
			boxes1.add(make_shared<box>(point3(x0, y0, z0), point3(x1, y1, z1),
				ground));
		}
	}
	hittable_list objects;
	objects.add(make_shared<bvh_node>(boxes1, 0, 1));
	auto light = make_shared<diffuse_light>(color(7, 7, 7));
	objects.add(make_shared<xz_rect>(123, 423, 147, 412, 554, light));
	auto center1 = point3(400, 400, 200);
	auto center2 = center1 + vec3(30, 0, 0);
	auto moving_sphere_material = make_shared<lambertian>(color(0.7, 0.3,
		0.1));
	objects.add(make_shared<moving_sphere>(center1, center2, 0, 1, 50,
		moving_sphere_material));
	objects.add(make_shared<sphere>(point3(260, 150, 45), 50,
		make_shared<dielectric>(1.5)));
	objects.add(make_shared<sphere>(
		point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)
	));
	auto boundary = make_shared<sphere>(point3(360, 150, 145), 70,
		make_shared<dielectric>(1.5));
	objects.add(boundary);
	objects.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4,
		0.9)));
	boundary = make_shared<sphere>(point3(0, 0, 0), 5000,
		make_shared<dielectric>(1.5));
	objects.add(make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));
	auto emat = make_shared<lambertian>(make_shared<image_texture>
		("earthmap.jpg"));
	objects.add(make_shared<sphere>(point3(400, 200, 400), 100, emat));
	auto pertext = make_shared<noise_texture>(0.1);
	objects.add(make_shared<sphere>(point3(220, 280, 300), 80,
		make_shared<lambertian>(pertext)));
	hittable_list boxes2;
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	int ns = 1000;
	for (int j = 0; j < ns; j++) {
		boxes2.add(make_shared<sphere>(point3::random(0, 165), 10, white));
	}
	objects.add(make_shared<translate>(
		make_shared<rotate_y>(
			make_shared<bvh_node>(boxes2, 0.0, 1.0), 15),
		vec3(-100, 270, 395)
	)
	);
	return objects;
}

hittable_list cornell_smoke() {
	hittable_list objects;
	auto red = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(7, 7, 7));
	objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
	objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
	objects.add(make_shared<xz_rect>(113, 443, 127, 432, 554, light));
	objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
	objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
	objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));
	shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0),
		point3(165, 330, 165), white);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265, 0, 295));
	shared_ptr<hittable> box2 = make_shared<box>(point3(0, 0, 0),
		point3(165, 165, 165), white);
	box2 = make_shared<rotate_y>(box2, -18);
	box2 = make_shared<translate>(box2, vec3(130, 0, 65));
	objects.add(make_shared<constant_medium>(box1, 0.01, color(0, 0, 0)));
	objects.add(make_shared<constant_medium>(box2, 0.01, color(1, 1, 1)));
	return objects;
}
hittable_list cornell_box() {
	hittable_list objects;
	auto red = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));
	objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
	objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
	objects.add(make_shared<xz_rect>(213, 343, 227, 332, 554, light));
	objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
	objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
	objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));
	objects.add(make_shared<box>(point3(130, 0, 65), point3(295, 165, 230),
		white));
	objects.add(make_shared<box>(point3(265, 0, 295), point3(430, 330, 460),
		white));
	shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330,
		165), white);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265, 0, 295));
	objects.add(box1);
	shared_ptr<hittable> box2 = make_shared<box>(point3(0, 0, 0),
		point3(165, 165, 165), white);
	box2 = make_shared<rotate_y>(box2, -18);
	box2 = make_shared<translate>(box2, vec3(130, 0, 65));
	objects.add(box2);
	return objects;
}
hittable_list simple_light() {
	hittable_list objects;
	auto pertext = make_shared<noise_texture>(4);
	objects.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
		make_shared<lambertian>(pertext)));
	objects.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>
		(pertext)));
	auto difflight = make_shared<diffuse_light>(color(4, 4, 4));
	objects.add(make_shared<xy_rect>(3, 5, 1, 3, -2, difflight));
	// 第二个光源：光源球体
	auto sphere_light = make_shared<diffuse_light>(color(7, 7, 7)); // 光强度为 (7, 7, 7)
	objects.add(make_shared<sphere>(
		point3(0, 5.5 ,0), 1.5, // 球心位置 (0, 5.5, 0)，半径为 1.5
		sphere_light
	));
	return objects;
}
hittable_list two_spheres() {
	hittable_list objects;
	auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
		color(0.9, 0.9, 0.9));
	objects.add(make_shared<sphere>(point3(0, -10, 0), 10,
		make_shared<lambertian>(checker)));
	objects.add(make_shared<sphere>(point3(0, 10, 0), 10,
		make_shared<lambertian>(checker)));
	return objects;
}
hittable_list earth() {
	auto earth_texture = make_shared<image_texture>("D:/桌面/compic/Assignment4_Ray_Tracing/earthmap.jpg");
	auto earth_surface = make_shared<lambertian>(earth_texture);
	auto globe = make_shared<sphere>(point3(0, 0, 0), 2, earth_surface);
	return hittable_list(globe);
}
hittable_list two_perlin_spheres() {
	hittable_list objects;
	auto pertext = make_shared<noise_texture>(4);
	objects.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
		make_shared<lambertian>(pertext)));
	objects.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>
		(pertext)));
	return objects;
}
hittable_list random_scene() {
	hittable_list world;
	auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
		color(0.9, 0.9, 0.9));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
		make_shared<lambertian>(checker)));
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9 * random_double(), 0.2, b +
				0.9 * random_double());
			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				shared_ptr<material> sphere_material;
				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					sphere_material = make_shared<lambertian>(albedo);
					world.add(make_shared<sphere>(center, 0.2,
						sphere_material));
					auto center2 = center + vec3(0, random_double(0, .5), 0);
					world.add(make_shared<moving_sphere>(
						center, center2, 0.0, 1.0, 0.2, sphere_material));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = make_shared<metal>(albedo, fuzz);
					world.add(make_shared<sphere>(center, 0.2,
						sphere_material));
				}
				else {
					// glass
					sphere_material = make_shared<dielectric>(1.5);
					world.add(make_shared<sphere>(center, 0.2,
						sphere_material));
				}
			}
		}
	}
	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));
	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));
	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));
	// 返回 BVH 树的根节点
	return make_shared<bvh_node>(world, 0.0, 1.0);
}
void rendering()
{
	double startFrame = clock();

	printf("CGAssignment4 (built %s at %s) \n", __DATE__, __TIME__);
	std::cout << "Ray-tracing based rendering launched..." << std::endl;

	// Image
	/*
	const int image_width = gWidth;
	const int image_height = gHeight;
	const int samples_per_pixel = 100;
	const int max_depth = 50;
	*/
	const int image_width = gWidth;
	const int image_height = static_cast<int>(image_width / aspect_ratio);
	 int samples_per_pixel = 100;
	const int max_depth = 50;
	// Render
	// 基础1
	// World
	
	/*
	//chapter 9之前
	world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
	world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));
	*/
	/*
	//chapter 9,10
	hittable_list world;
	auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
	auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
	auto material_left = make_shared<dielectric>(1.5);
	auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);
	world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0,
		material_ground));
	world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5,
		material_center));
	world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5,
		material_left));
	world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), -0.4,
		material_left));
	world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5,
		material_right));
	// Camera
	camera cam;
	*/
	/*
	//chapter11.1
	auto R = cos(pi / 4);
	hittable_list world;
	auto material_left = make_shared<lambertian>(color(0, 0, 1));
	auto material_right = make_shared<lambertian>(color(1, 0, 0));
	world.add(make_shared<sphere>(point3(-R, 0, -1), R, material_left));
	world.add(make_shared<sphere>(point3(R, 0, -1), R, material_right));
	// Camera
	camera cam(90.0, aspect_ratio);
	*/
	/*
	//chapter 12
	hittable_list world;
	
	auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
	auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
	auto material_left = make_shared<dielectric>(1.5);
	auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);
	world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0,
		material_ground));
	world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5,
		material_center));
	world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5,
		material_left));
	world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), -0.45,
		material_left));
	world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5,
		material_right));
	point3 lookfrom(3, 3, 2);       // 相机位置
	point3 lookat(0, 0, -1);        // 目标点
	vec3 vup(0, 1, 0);              // 垂直向量
	auto dist_to_focus = (lookfrom - lookat).length();
	auto aperture = 2.0;            // 光圈大小

	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
	*/
	/*
	//chapter13
	auto world = random_scene();

	// Camera
	point3 lookfrom(13, 2, 3);
	point3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	auto dist_to_focus = 10.0;
	auto aperture = 0.1;
	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture,
		dist_to_focus, 0.0, 1.0);
		*/
	hittable_list world;
	point3 lookfrom;
	point3 lookat;
	auto vfov = 40.0;
	auto aperture = 0.0;
	color background(0, 0, 0);
	switch (0) {
	case 1:
		world = random_scene();
		background = color(0.70, 0.80, 1.00);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		aperture = 0.1;
		break;
	case 2:
		world = two_spheres();
		background = color(0.70, 0.80, 1.00);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		break;
	case 3:
		world = two_perlin_spheres();
		background = color(0.70, 0.80, 1.00);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		break;
	case 4:
		world = earth();
		background = color(0.70, 0.80, 1.00);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		break;
	case 5:
		world = simple_light();
		samples_per_pixel = 400;
		background = color(0, 0, 0);
		lookfrom = point3(26, 3, 6);
		lookat = point3(0, 2, 0);
		vfov = 20.0;
		break;
	case 6:
		world = cornell_box();
		aspect_ratio = 1.0;
		gWidth = 600;
		samples_per_pixel = 200;
		background = color(0, 0, 0);
		lookfrom = point3(278, 278, -800);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		break;
	case 7:
		world = cornell_smoke();
		aspect_ratio = 1.0;
		gWidth = 600;
		samples_per_pixel = 200;
		lookfrom = point3(278, 278, -800);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		break;
	default:
	case 8:
		world = final_scene();
		aspect_ratio = 1.0;
		gWidth = 800;
		samples_per_pixel = 10000;
		background = color(0, 0, 0);
		lookfrom = point3(478, 278, -600);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		break;
	
	}
	// Camera
	vec3 vup(0, 1, 0);
	auto dist_to_focus = 10.0;
	camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
		0.0, 1.0);
	/*
	auto viewport_height = 2.0;
	auto viewport_width = aspect_ratio * viewport_height;
	auto focal_length = 1.0;

	auto origin = point3(0, 0, 0);
	auto horizontal = vec3(viewport_width, 0, 0);
	auto vertical = vec3(0, viewport_height, 0);
	auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);
	*/
	// The main ray-tracing based rendering loop
	// TODO: finish your own ray-tracing renderer according to the given tutorials
	std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
	for (int j = image_height - 1; j >= 0; --j) {
		for (int i = 0; i < image_width; ++i) {
			color pixel_color(0, 0, 0);
			for (int s = 0; s < samples_per_pixel; ++s) {
				auto u = (i + random_double()) / (image_width - 1);
				auto v = (j + random_double()) / (image_height - 1);
				ray r = cam.get_ray(u, v);
				pixel_color += ray_color(r, background, world, max_depth);
			}
			write_color(i, j, std::cout, pixel_color, samples_per_pixel);
		}
	}


	double endFrame = clock();
	double timeConsuming = static_cast<double>(endFrame - startFrame) / CLOCKS_PER_SEC;
	std::cout << "Ray-tracing based rendering over..." << std::endl;
	std::cout << "The rendering task took " << timeConsuming << " seconds" << std::endl;
}