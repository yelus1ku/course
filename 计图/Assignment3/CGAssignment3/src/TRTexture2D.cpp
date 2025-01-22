#include "TRTexture2D.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
namespace TinyRenderer
{
	//----------------------------------------------TRTexture2D 23.10.25----------------------------------------------

	TRTexture2D::TRTexture2D() :
		m_width(0), m_height(0), m_channel(0), m_pixels(nullptr),
		m_warp_mode(TRTextureWarpMode::TR_REPEAT),
		m_filtering_mode(TRTextureFilterMode::TR_NEAREST) {}

	TRTexture2D::~TRTexture2D() { freeLoadedImage(); }

	void TRTexture2D::setWarpingMode(TRTextureWarpMode mode) { m_warp_mode = mode; }

	void TRTexture2D::setFilteringMode(TRTextureFilterMode mode) { m_filtering_mode = mode; }

	bool TRTexture2D::loadTextureFromFile(
		const std::string &filepath,
		TRTextureWarpMode warpMode,
		TRTextureFilterMode filterMode)
	{
		freeLoadedImage();

		m_warp_mode = warpMode;
		m_filtering_mode = filterMode;

		//Load image from given file using stb_image.h
		//Refs: https://github.com/nothings/stb
		{
			stbi_set_flip_vertically_on_load(true);
			m_pixels = stbi_load(filepath.c_str(), &m_width, &m_height, &m_channel, 0);
		}

		if (m_pixels == nullptr)
		{
			std::cerr << "Failed to load image from " << filepath << std::endl;
			exit(1);
		}

		return true;
	}

	void TRTexture2D::readPixel(int u, int v, unsigned char &r, unsigned char &g, unsigned char &b, unsigned char &a) const
	{
		//Handling out of range situation
		{
			if (u < 0 || u >= m_width)
			{
				switch (m_warp_mode)
				{
				case TRTextureWarpMode::TR_REPEAT:
					u = u > 0 ? (u % m_width) : (m_width - 1 + u % m_width);
					break;
				case TRTextureWarpMode::TR_CLAMP_TO_EDGE:
					u = (u < 0) ? 0 : m_width - 1;
					break;
				default:
					u = (u < 0) ? 0 : m_width - 1;
					break;
				}
			}

			if (v < 0 || v >= m_height)
			{
				switch (m_warp_mode)
				{
				case TRTextureWarpMode::TR_REPEAT:
					v = v > 0 ? (v % m_height) : (m_height - 1 + v % m_height);
					break;
				case TRTextureWarpMode::TR_CLAMP_TO_EDGE:
					v = (v < 0) ? 0 : m_height - 1;
					break;
				default:
					v = (v < 0) ? 0 : m_height - 1;
					break;
				}
			}
		}

		int index = (v * m_height + u) * m_channel;
		r = m_pixels[index + 0];
		g = m_pixels[index + 1];
		b = m_pixels[index + 2];
		a = (m_channel >= 4) ? m_pixels[index + 3] : a;

		return;
	}

	void TRTexture2D::freeLoadedImage()
	{
		if (m_pixels != nullptr)
		{
			stbi_image_free(m_pixels);
		}

		m_pixels = nullptr;
		m_width = m_height = m_channel = 0;
	}

	glm::vec4 TRTexture2D::sample(const glm::vec2 &uv) const
	{
		//Perform sampling procedure
		//Note: return texel that ranges from 0.0f to 1.0f instead of [0,255]
		glm::vec4 texel(1.0f);
		switch (m_filtering_mode)
		{
		case TRTextureFilterMode::TR_NEAREST:
			texel = TRTexture2DSampler::textureSampling_nearest(*this, uv);
			break;
		case TRTextureFilterMode::TR_LINEAR:
			texel = TRTexture2DSampler::textureSampling_bilinear(*this, uv);
			break;
		default:
			break;
		}

		return texel;
	}

	//----------------------------------------------TRTexture2DSampler----------------------------------------------

	glm::vec4 TRTexture2DSampler::textureSampling_nearest(const TRTexture2D &texture, glm::vec2 uv)
	{
		unsigned char r = 255, g = 255, b = 255, a = 255;

		//Task1: Implement nearest sampling algorithm for texture sampling
		// Note: You should use texture.readPixel() to read the pixel, and for instance, 
		//       use texture.readPixel(25,35,r,g,b,a) to read the pixel in (25, 35).
		//       But before that, you need to map uv from [0,1]*[0,1] to [0,width-1]*[0,height-1].
		{
			// 将 uv 坐标映射到纹理的像素坐标范围 [0, width-1] 和 [0, height-1]
			int x = static_cast<int>(std::round(uv.x * (texture.getWidth() - 1)));  // 使用四舍五入
			int y = static_cast<int>(std::round(uv.y * (texture.getHeight() - 1))); // 使用四舍五入

			// 使用 texture.readPixel 获取指定像素的 RGBA 值
			texture.readPixel(x, y, r, g, b, a);

		}
		// 归一化 RGBA 值到 [0, 1] 范围
		constexpr float denom = 1.0f / 255.0f;
		return glm::vec4(r, g, b, a) * denom;
	}

	glm::vec4 TRTexture2DSampler::textureSampling_bilinear(const TRTexture2D &texture, glm::vec2 uv)
	{
		//Note: Delete this line when you try to implement Task 4. 
		//return textureSampling_nearest(texture, uv);
		// 将 uv 坐标映射到纹理的像素坐标范围 [0, width-1] 和 [0, height-1]
		float x = uv.x * (texture.getWidth() - 1);
		float y = uv.y * (texture.getHeight() - 1);

		// 获取整数部分和小数部分
		int x0 = static_cast<int>(std::floor(x)); // 左下角 x 坐标
		int y0 = static_cast<int>(std::floor(y)); // 左下角 y 坐标
		int x1 = std::min(x0 + 1, texture.getWidth() - 1); // 右下角 x 坐标
		int y1 = std::min(y0 + 1, texture.getHeight() - 1); // 右上角 y 坐标

		// 获取四个邻近像素的 RGBA 值
		unsigned char r0, g0, b0, a0; // 左下角像素
		unsigned char r1, g1, b1, a1; // 右下角像素
		unsigned char r2, g2, b2, a2; // 左上角像素
		unsigned char r3, g3, b3, a3; // 右上角像素

		texture.readPixel(x0, y0, r0, g0, b0, a0);
		texture.readPixel(x1, y0, r1, g1, b1, a1);
		texture.readPixel(x0, y1, r2, g2, b2, a2);
		texture.readPixel(x1, y1, r3, g3, b3, a3);

		// 计算水平插值
		float tx = x - x0; // 小数部分（横向插值因子）
		glm::vec4 color0(r0, g0, b0, a0); // 左下角像素颜色
		glm::vec4 color1(r1, g1, b1, a1); // 右下角像素颜色
		glm::vec4 color2(r2, g2, b2, a2); // 左上角像素颜色
		glm::vec4 color3(r3, g3, b3, a3); // 右上角像素颜色

		glm::vec4 hInterp0 = glm::mix(color0, color1, tx); // 左下与右下之间的插值
		glm::vec4 hInterp1 = glm::mix(color2, color3, tx); // 左上与右上之间的插值

		// 计算垂直插值
		float ty = y - y0; // 小数部分（纵向插值因子）
		glm::vec4 finalColor = glm::mix(hInterp0, hInterp1, ty); // 最终的插值结果

		// 归一化 RGBA 值到 [0, 1] 范围
		constexpr float denom = 1.0f / 255.0f;
		return finalColor * denom;
		//Improvement: Implement bilinear sampling algorithm for texture sampling
		// Note: You should use texture.readPixel() to read the pixel, and for instance, 
		//       use texture.readPixel(25,35,r,g,b,a) to read the pixel in (25, 35).
	}
}