#ifndef COLOR_H
#define COLOR_H
#include "vec3.h"
#include <iostream>
#include "rtweekend.h"
extern std::vector<std::vector<color>> gCanvas;

/*
void write_color(std::ostream& out, color pixel_color) {
	// Write the translated [0,255] value of each color component.
	out << static_cast<int>(255.999 * pixel_color.x()) << ' '
		<< static_cast<int>(255.999 * pixel_color.y()) << ' '
		<< static_cast<int>(255.999 * pixel_color.z()) << '\n';
}
*/
void write_color(int x, int y, color pixel_color, int samples_per_pixel) {
    // Divide the color by the number of samples
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();
    auto scale = 1.0 / samples_per_pixel;
    r *= scale;
    g *= scale;
    b *= scale;

    // Clamp the color to the range [0.0, 0.999]
    r = clamp(r, 0.0, 0.999);
    g = clamp(g, 0.0, 0.999);
    b = clamp(b, 0.0, 0.999);

    // Convert to [0, 255] range and write to the canvas
    gCanvas[y][x] = color(
        static_cast<int>(256 * r),
        static_cast<int>(256 * g),
        static_cast<int>(256 * b)
    );
}

#endif
