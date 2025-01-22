#include "TRShaderPipeline.h"

#include <algorithm>

namespace TinyRenderer
{
	//----------------------------------------------VertexData----------------------------------------------

	TRShaderPipeline::VertexData TRShaderPipeline::VertexData::lerp(
		const TRShaderPipeline::VertexData &v0,
		const TRShaderPipeline::VertexData &v1,
		float frac)
	{
		//Linear interpolation
		VertexData result;
		result.pos = (1.0f - frac) * v0.pos + frac * v1.pos;
		result.col = (1.0f - frac) * v0.col + frac * v1.col;
		result.nor = (1.0f - frac) * v0.nor + frac * v1.nor;
		result.tex = (1.0f - frac) * v0.tex + frac * v1.tex;
		result.cpos = (1.0f - frac) * v0.cpos + frac * v1.cpos;
		result.spos.x = (1.0f - frac) * v0.spos.x + frac * v1.spos.x;
		result.spos.y = (1.0f - frac) * v0.spos.y + frac * v1.spos.y;

		return result;
	}

	TRShaderPipeline::VertexData TRShaderPipeline::VertexData::barycentricLerp(
		const VertexData &v0, 
		const VertexData &v1, 
		const VertexData &v2,
		glm::vec3 w)
	{
		VertexData result;
		result.pos = w.x * v0.pos + w.y * v1.pos + w.z * v2.pos;
		result.col = w.x * v0.col + w.y * v1.col + w.z * v2.col;
		result.nor = w.x * v0.nor + w.y * v1.nor + w.z * v2.nor;
		result.tex = w.x * v0.tex + w.y * v1.tex + w.z * v2.tex;
		result.cpos = w.x * v0.cpos + w.y * v1.cpos + w.z * v2.cpos;
		result.spos.x = w.x * v0.spos.x + w.y * v1.spos.x + w.z * v2.spos.x;
		result.spos.y = w.x * v0.spos.y + w.y * v1.spos.y + w.z * v2.spos.y;

		return result;
	}

	void TRShaderPipeline::VertexData::prePerspCorrection(VertexData &v)
	{
		//Perspective correction: the world space properties should be multipy by 1/w before rasterization
		//https://zhuanlan.zhihu.com/p/144331875
		//We use pos.w to store 1/w
		v.pos.w = 1.0f / v.cpos.w;
		v.pos = glm::vec4(v.pos.x * v.pos.w, v.pos.y * v.pos.w, v.pos.z * v.pos.w, v.pos.w);
		v.tex = v.tex * v.pos.w;
		v.nor = v.nor * v.pos.w;
		v.col = v.col * v.pos.w;
	}

	void TRShaderPipeline::VertexData::aftPrespCorrection(VertexData &v)
	{
		//Perspective correction: the world space properties should be multipy by w after rasterization
		//https://zhuanlan.zhihu.com/p/144331875
		//We use pos.w to store 1/w
		float w = 1.0f / v.pos.w;
		v.pos = v.pos * w;
		v.tex = v.tex * w;
		v.nor = v.nor * w;
		v.col = v.col * w;
	}

	//----------------------------------------------TRShaderPipeline----------------------------------------------

	void TRShaderPipeline::rasterize_wire(
		const VertexData &v0,
		const VertexData &v1,
		const VertexData &v2,
		const unsigned int &screen_width,
		const unsigned int &screene_height,
		std::vector<VertexData> &rasterized_points)
	{
		//Draw each line step by step
		rasterize_wire_aux(v0, v1, screen_width, screene_height, rasterized_points);
		rasterize_wire_aux(v1, v2, screen_width, screene_height, rasterized_points);
		rasterize_wire_aux(v0, v2, screen_width, screene_height, rasterized_points);
	}

	float edgeFunction(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
		return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
	}


	void TRShaderPipeline::rasterize_fill_edge_function(
		const VertexData &v0,
		const VertexData &v1,
		const VertexData &v2,
		const unsigned int &screen_width,
		const unsigned int &screen_height,
		std::vector<VertexData> &rasterized_points)
	{
		//Edge-function rasterization algorithm

		// 2: Implement edge-function triangle rassterization algorithm
		// Note: You should use VertexData::barycentricLerp(v0, v1, v2, w) for interpolation, 
		//       interpolated points should be pushed back to rasterized_points.
		//       Interpolated points shold be discarded if they are outside the window. 

		//       v0.spos, v1.spos and v2.spos are the screen space vertices.

		//For instance:
		//rasterized_points.push_back(v0);
		//rasterized_points.push_back(v1);
		//rasterized_points.push_back(v2);
		// 获取三角形的屏幕空间坐标
		int x0 = static_cast<int>(v0.spos.x);
		int y0 = static_cast<int>(v0.spos.y);
		int x1 = static_cast<int>(v1.spos.x);
		int y1 = static_cast<int>(v1.spos.y);
		int x2 = static_cast<int>(v2.spos.x);
		int y2 = static_cast<int>(v2.spos.y);

		// 计算三角形的包围盒
		int minX = std::max(0, std::min({ x0, x1, x2 }));
		int maxX = std::min(int(screen_width) - 1, std::max(x0, std::max(x1, x2)));
		int minY = std::max(0, std::min({ y0, y1, y2 }));
		int maxY = std::min(int(screen_height) - 1, std::max(y0, std::max(y1, y2)));

		// 遍历包围盒内的像素点
		for (int y = minY; y <= maxY; ++y) {
			for (int x = minX; x <= maxX; ++x) {
				glm::vec2 p(x + 0.5f, y + 0.5f); // 当前像素的中心坐标

				float w0 = edgeFunction(v1.spos, v2.spos, p);
				float w1 = edgeFunction(v2.spos, v0.spos, p);
				float w2 = edgeFunction(v0.spos, v1.spos, p);

				// 检查是否在三角形内部
				if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
					// 计算重心坐标
					glm::vec3 weights = glm::vec3(w0, w1, w2);
					weights /= (w0 + w1 + w2); // 归一化

					// 使用重心插值生成当前像素的 VertexData，并将其加入到 rasterized_points 中
					rasterized_points.push_back(VertexData::barycentricLerp(v0, v1, v2, weights));
				}
			}
		}

	}

	void TRShaderPipeline::rasterize_wire_aux(
		const VertexData &from,
		const VertexData &to,
		const unsigned int &screen_width,
		const unsigned int &screen_height,
		std::vector<VertexData> &rasterized_points)
	{

		//1: Implement Bresenham line rasterization
		// Note: You shold use VertexData::lerp(from, to, weight) for interpolation,
		//       interpolated points should be pushed back to rasterized_points.
		//       Interpolated points shold be discarded if they are outside the window. 
		
		//       from.spos and to.spos are the screen space vertices.

		//For instance:

		//rasterized_points.push_back(from);
		//rasterized_points.push_back(to);
		// 获取起点和终点的屏幕空间坐标
		int x0 = static_cast<int>(from.spos.x);
		int y0 = static_cast<int>(from.spos.y);
		int x1 = static_cast<int>(to.spos.x);
		int y1 = static_cast<int>(to.spos.y);

		// 计算 dx 和 dy
		int dx = x1 - x0;
		int dy = y1 - y0;

		// 确定步长方向
		int stepX = (dx > 0) ? 1 : -1; // 水平方向的步长
		int stepY = (dy > 0) ? 1 : -1; // 垂直方向的步长
		dx = std::abs(dx); // 取绝对值
		dy = std::abs(dy);

		// 初始化误差
		int err = dx - dy; // 初始误差

		// 光栅化直线，从起点到终点
		while (true) {
			// 检查当前像素是否在窗口内
			if (x0 >= 0 && x0 < screen_width && y0 >= 0 && y0 < screen_height) {
				// 计算插值因子
				float weight = (dx + dy > 0) ? static_cast<float>(std::abs(x0 - x1) + std::abs(y0 - y1)) / (dx + dy) : 0.0f;
				// 使用线性插值生成当前像素的 VertexData，并将其加入到 rasterized_points 中
				rasterized_points.push_back(VertexData::lerp(from, to, weight));
			}

			// 检查是否到达终点
			if (x0 == x1 && y0 == y1) break;

			// 计算新的误差
			int err2 = err * 2;

			// 更新 x 和 y 坐标
			if (err2 > -dy) {
				err -= dy; // 更新误差
				x0 += stepX; // 水平移动
			}
			if (err2 < dx) {
				err += dx; // 更新误差
				y0 += stepY; // 垂直移动
			}
		}

	}

	void TRDefaultShaderPipeline::vertexShader(VertexData &vertex)
	{
		//Local space -> World space -> Camera space -> Project space
		vertex.pos = m_model_matrix * glm::vec4(vertex.pos.x, vertex.pos.y, vertex.pos.z, 1.0f);
		vertex.cpos = m_view_project_matrix * vertex.pos;
	}

	void TRDefaultShaderPipeline::fragmentShader(const VertexData &data, glm::vec4 &fragColor)
	{
		//Just return the color.
		fragColor = glm::vec4(data.tex, 0.0, 1.0f);
	}
}