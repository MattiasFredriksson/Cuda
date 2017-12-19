#pragma once

#include "GL/glew.h"

GLuint generateTraceContext(unsigned int buffer_width, unsigned int buffer_height, unsigned int num_light, const char *obj_file);
void closeContext();
void rayTrace(unsigned int buffer_width, unsigned int buffer_height, float fov, float trace_depth, unsigned int kernel_dim);