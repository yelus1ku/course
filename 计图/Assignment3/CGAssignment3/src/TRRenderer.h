#ifndef TRRENDERER_H
#define TRRENDERER_H

#include "glm/glm.hpp"
#include "SDL2/SDL.h"
#include "TRFrameBuffer.h"
#include "TRDrawableMesh.h"
#include "TRShadingState.h"
#include "TRShadingPipeline.h"

#include <mutex>

namespace TinyRenderer
{
	class TRRenderer final
	{
	public:
		typedef std::shared_ptr<TRRenderer> ptr;

		TRRenderer(int width, int height);
		~TRRenderer() = default;

		//Drawable objects load/unload
		void addDrawableMesh(TRDrawableMesh::ptr mesh);
		void addDrawableMesh(const std::vector<TRDrawableMesh::ptr> &meshes);
		void unloadDrawableMesh();

		void clearColor(glm::vec4 color);

		//Setting
		void setViewMatrix(const glm::mat4 &view);
		void setModelMatrix(const glm::mat4 &model);
		void setProjectMatrix(const glm::mat4 &project, float near, float far);
		void setShaderPipeline(TRShadingPipeline::ptr shader);
		void setViewerPos(const glm::vec3 &viewer);

		int addPointLight(glm::vec3 pos, glm::vec3 atten, glm::vec3 color);
		//新添加--------------
		int addSpotLight(const glm::vec3& pos, const glm::vec3& direction, const glm::vec3& color,
			const glm::vec3& attenuation, float cutOff, float outerCutOff);

		// 获取聚光灯
		TRSpotLight& getSpotLight(int index);  // 返回引用
		//--------------
		TRPointLight &getPointLight(const int &index);

		glm::mat4 getMVPMatrix();

		//Draw call
		void renderAllDrawableMeshes();

		//Commit rendered result
		unsigned char* commitRenderedColorBuffer();
		unsigned int getNumberOfClipFaces() const;
		unsigned int getNumberOfCullFaces() const;



	private:

		//Homogeneous space clipping - Sutherland Hodgeman algorithm
		std::vector<TRShadingPipeline::VertexData> clipingSutherlandHodgeman(
			const TRShadingPipeline::VertexData &v0,
			const TRShadingPipeline::VertexData &v1,
			const TRShadingPipeline::VertexData &v2) const;

		//Cliping auxiliary functions
		std::vector<TRShadingPipeline::VertexData> clipingSutherlandHodgeman_aux(
			const std::vector<TRShadingPipeline::VertexData> &polygon,
			const int &axis, 
			const int &side) const;
		bool isPointInsideInClipingFrustum(const glm::vec4 &p) const
		{
			return (p.x <= p.w && p.x >= -p.w)
				&& (p.y <= p.w && p.y >= -p.w)
				&& (p.z <= p.w && p.z >= -p.w)
				&& (p.w <= m_frustum_near_far.y && p.w >= m_frustum_near_far.x);
		}

		//Back face culling
		bool isBackFacing(const glm::ivec2 &v0, const glm::ivec2 &v1, const glm::ivec2 &v2, TRCullFaceMode mode) const;

	private:
		//新添加-----------------------------------------------------
		std::vector<TRSpotLight> spotLights;  // 存储聚光灯的容器
		//Drawable mesh array
		std::vector<TRDrawableMesh::ptr> m_drawableMeshes;

		//MVP transformation matrices
		glm::mat4 m_viewMatrix = glm::mat4(1.0f);
		glm::mat4 m_modelMatrix = glm::mat4(1.0f);
		glm::mat4 m_projectMatrix = glm::mat4(1.0f);
		glm::mat4 m_mvp_matrix = glm::mat4(1.0f);
		bool m_mvp_dirty = false;

		//Near plane & far plane
		glm::vec2 m_frustum_near_far;

		//Viewport transformation (ndc space -> screen space)
		glm::mat4 m_viewportMatrix = glm::mat4(1.0f);

		//Shader pipeline handler
		TRShadingPipeline::ptr m_shader_handler = nullptr;

		//Double buffers
		TRFrameBuffer::ptr m_backBuffer;                      // The frame buffer that's going to be written.
		TRFrameBuffer::ptr m_frontBuffer;                     // The frame buffer that's going to be displayed.

		struct Profile
		{
			unsigned int m_num_cliped_triangles = 0;
			unsigned int m_num_culled_triangles = 0;
		};
		Profile m_clip_cull_profile;
	};
}

#endif