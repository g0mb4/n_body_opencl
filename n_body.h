#ifndef __N_BODY_H__
#define __N_BODY_H__

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#ifdef _MSC_VER

	#define PACKED
	#pragma pack(push,1)

	typedef struct _body_movement_state
	{
		cl_float3 pos;
		cl_float3 vel;
	} body_mstate;

	#pragma pack(pop)
	#undef PACKED

	#define PACKED
	#pragma pack(push,1)

		typedef struct _body_properties
		{
			cl_float mass;
			cl_float radius;
			cl_float coll_coef;
		} body_props;

	#pragma pack(pop)
	#undef PACKED

#endif

#endif