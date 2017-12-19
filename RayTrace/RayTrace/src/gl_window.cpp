//--------------------------------------------------------------------------------------
// BTH - Stefan Petersson 2014.
// Modified - Mattias Fredriksson 2017
//--------------------------------------------------------------------------------------
#include "gl_window.h"
#include "resource.h"
#include <string>
#include <fstream>
#include <streambuf>
#include "GL/glew.h"
#include "RayTrace.h"
#include <assert.h>
#include <iostream>
//#include "../GL/GL.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")

using namespace std;

// Forward declare:
HWND InitWindow(HINSTANCE hInstance, unsigned int width, unsigned int height);
HGLRC CreateOpenGLContext(HWND wndHandle);
#define BUFFER_OFFSET(i) ((char *)nullptr + (i))

// Params:
GLuint gVertexBuffer = 0;
GLuint gVertexAttribute = 0;
GLuint gShaderProgram = 0;

void glError()
{
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
		std::cout << "Err: " << err << std::endl;
	assert(err == GL_NO_ERROR);
}

void CreateShaders()
{
	//create vertex shader
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	// open glsl file and put it in a string
	ifstream shaderFile("VertexShader.glsl");
	std::string shaderText((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
	shaderFile.close();
	// make a double pointer (only valid here)
	const char* shaderTextPtr = shaderText.c_str();
	// ask GL to load this
	glShaderSource(vs, 1, &shaderTextPtr, nullptr);
	// ask GL to compile it
	glCompileShader(vs);

	//create fragment shader | same process.
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	shaderFile.open("Fragment.glsl");
	shaderText.assign((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
	shaderFile.close();
	shaderTextPtr = shaderText.c_str();
	glShaderSource(fs, 1, &shaderTextPtr, nullptr);
	glCompileShader(fs);

	//link shader program (connect vs and ps)
	gShaderProgram = glCreateProgram();
	glAttachShader(gShaderProgram, fs);
	glAttachShader(gShaderProgram, vs);
	glLinkProgram(gShaderProgram);
	glUseProgram(gShaderProgram);
	glError();

	GLint bufferUniform = glGetUniformLocation(gShaderProgram, "tex");
	glUniform1i(bufferUniform, 0); //Assigns sampling slot
	glError();
}

void CreateTriangleData()
{
	// this is how we will structure the input data for the vertex shader
	// every six floats, is one vertex.
	struct TriangleVertex
	{
		float x, y, z;
		float u,v;
	};
	// create the actual data in plane Z = 0
	TriangleVertex triangleVertices[6] =
	{
		// pos and uv for each vertex
		{ -1.f, -1.f, 0.0f,	0.0f, 0.0f },
		{ -1.f, 1.f, 0.0f, 0.0f, 1.0f },
		{ 1.f, -1.f, 0.0f, 1.0f, 0.0f },

		{ 1.f, 1.f, 0.0f, 1.0f, 1.0f },
		{ 1.f, -1.f, 0.0f, 1.0f, 0.0f },
		{ -1.f, 1.f, 0.0f, 0.0f, 1.0f }
	};

	// Vertex Array Object (VAO) 
	glGenVertexArrays(1, &gVertexAttribute);
	// bind == enable
	glBindVertexArray(gVertexAttribute);
	// this activates the first and second attributes of this VAO
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	// create a vertex buffer object (VBO) id
	glGenBuffers(1, &gVertexBuffer);
	// Bind the buffer ID as an ARRAY_BUFFER
	glBindBuffer(GL_ARRAY_BUFFER, gVertexBuffer);
	// This "could" imply copying to the GPU, depending on what the driver wants to do...
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);

	// query where which slot corresponds to the input vertex_position in the Vertex Shader 
	GLuint vertexPos = glGetAttribLocation(gShaderProgram, "vertex_position");
	// specify that: the vertex attribute "vertexPos", of 3 elements of type FLOAT, not normalized, with STRIDE != 0,
	//               starts at offset 0 of the gVertexBuffer (it is implicitly bound!)
	glVertexAttribPointer(vertexPos, 3, GL_FLOAT, GL_FALSE, sizeof(TriangleVertex), BUFFER_OFFSET(0));

	// query where which slot corresponds to the input vertex_uv in the Vertex Shader 
	GLuint vertexUV = glGetAttribLocation(gShaderProgram, "vertex_uv");
	// specify that: the vertex attribute "vertexUV", of 2 elements of type FLOAT, not normalized, with STRIDE != 0,
	//               starts at offset (12 bytes) of the gVertexBuffer 
	glVertexAttribPointer(vertexUV, 2, GL_FLOAT, GL_FALSE, sizeof(TriangleVertex), BUFFER_OFFSET(sizeof(float) * 3));
	glError();
}

void setUpGL(unsigned int width, unsigned int height)
{
	glViewport(0, 0, width, height);
	glEnable(GL_TEXTURE_2D);
}

void Render(GLuint tex)
{
	// set the color TO BE used
	glClearColor(0, 0, 0, 1);
	// use the color to clear the color buffer
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(gShaderProgram);
	glBindVertexArray(gVertexAttribute);
	glBindTexture(GL_TEXTURE0, tex);

	// draw 6 vertices starting from index 0 in the vertex array currently bound (VAO), with current in-use shader
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	unsigned int width = 1280, height = 960;
	MSG msg = { 0 };
	HWND wndHandle = InitWindow(hInstance, width, height); //1. Skapa fönster
	HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_LAUNCHWIN32WINDOWFROMCONSOLE)); // Keyboard shortcut table for application commands.
	GLuint cuFrameBuffer;

	if (wndHandle)
	{
		HDC hDC = GetDC(wndHandle);
		HGLRC hRC = CreateOpenGLContext(wndHandle); //2. Skapa och koppla OpenGL context

		glewInit(); //3. Initiera The OpenGL Extension Wrangler Library (GLEW)

		setUpGL(width, height); //4. Sätt viewport

		CreateShaders(); //5. Skapa vertex- och fragment-shaders

		CreateTriangleData(); //6. Definiera triangelvertiser, 7. Skapa vertex buffer object (VBO), 8.Skapa vertex array object (VAO)

		cuFrameBuffer = generateTraceContext(width, height, 10, "Suzanne.obj");
		if (!cuFrameBuffer) return -1;

		ShowWindow(wndHandle, nCmdShow);


		rayTrace(width, height, 30.f, 100.f, 16);

		while (WM_QUIT != msg.message)
		{
			if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
			{
				if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
				{
					TranslateMessage(&msg);
					DispatchMessage(&msg);
				}
			}
			else
			{
				Render(cuFrameBuffer); //9. Rendera

				SwapBuffers(hDC); //10. Växla front- och back-buffer
			}
		}

		wglMakeCurrent(NULL, NULL);
		ReleaseDC(wndHandle, hDC);
		wglDeleteContext(hRC);
		DestroyWindow(wndHandle);
	}

	return (int)msg.wParam;
}

HWND InitWindow(HINSTANCE hInstance, unsigned int width, unsigned int height)
{
	const char CLASS_NAME[] = "RAY_TRACE_DEMO";
	const char WINDOW_DESEC[] = "Raytrace Demo";
	WNDCLASSEX wcex = { 0 };
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.hInstance = hInstance;
	wcex.lpszClassName = CLASS_NAME;
	if (!RegisterClassEx(&wcex))
		return false;

	RECT rc = { 0, 0, (int)width, (int)height };
	AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

	HWND handle = CreateWindow(
		CLASS_NAME,
		WINDOW_DESEC,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		rc.right - rc.left,
		rc.bottom - rc.top,
		nullptr,
		nullptr,
		hInstance,
		nullptr);

	return handle;
}


HGLRC CreateOpenGLContext(HWND wndHandle)
{
	//get handle to a device context (DC) for the client area
	//of a specified window or for the entire screen
	HDC hDC = GetDC(wndHandle);

	//details: http://msdn.microsoft.com/en-us/library/windows/desktop/dd318286(v=vs.85).aspx
	static  PIXELFORMATDESCRIPTOR pixelFormatDesc =
	{
		sizeof(PIXELFORMATDESCRIPTOR),    // size of this pfd  
		1,                                // version number  
		PFD_DRAW_TO_WINDOW |              // support window  
		PFD_SUPPORT_OPENGL |              // support OpenGL  
		PFD_DOUBLEBUFFER |                // double buffered
		PFD_DEPTH_DONTCARE,               // disable depth buffer <-- added by Stefan
		PFD_TYPE_RGBA,                    // RGBA type  
		32,                               // 32-bit color depth  
		0, 0, 0, 0, 0, 0,                 // color bits ignored  
		0,                                // no alpha buffer  
		0,                                // shift bit ignored  
		0,                                // no accumulation buffer  
		0, 0, 0, 0,                       // accum bits ignored  
		0,                                // 0-bits for depth buffer <-- modified by Stefan      
		0,                                // no stencil buffer  
		0,                                // no auxiliary buffer  
		PFD_MAIN_PLANE,                   // main layer  
		0,                                // reserved  
		0, 0, 0                           // layer masks ignored  
	};

	//attempt to match an appropriate pixel format supported by a
	//device context to a given pixel format specification.
	int pixelFormat = ChoosePixelFormat(hDC, &pixelFormatDesc);

	//set the pixel format of the specified device context
	//to the format specified by the iPixelFormat index.
	SetPixelFormat(hDC, pixelFormat, &pixelFormatDesc);

	//create a new OpenGL rendering context, which is suitable for drawing
	//on the device referenced by hdc. The rendering context has the same
	//pixel format as the device context.
	HGLRC hRC = wglCreateContext(hDC);

	//makes a specified OpenGL rendering context the calling thread's current
	//rendering context. All subsequent OpenGL calls made by the thread are
	//drawn on the device identified by hdc. 
	wglMakeCurrent(hDC, hRC);

	return hRC;
}