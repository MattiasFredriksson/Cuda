// Exclude the min and max macros from Windows.h
#include "window.h"
#include "WinMainParameters.h"
#include "resource.h"
#include <iostream>

using namespace WinMainParameters;

// ISO C++ conformant entry point. The project properties explicitly sets this as the entry point in the manner
// documented for the linker's /ENTRY option: http://msdn.microsoft.com/en-us/library/f9t8842e.aspx . As per
// the documentation, the value set as the entry point is "mainCRTStartup", not "main". Every C or C++ program
// must perform any initialization required by the language standards before it begins executing our code. In
// Visual C++, this is done by the *CRTStartup functions, each of which goes on to call the developer's entry
// point function.
int main(int /*argc*/, char* /*argv*/[]) {

	// Use the functions from WinMainParameters.h to get the values that would've been passed to WinMain.
	// Note that these functions are in the WinMainParameters namespace.
	HINSTANCE hInstance = GetHInstance();
	HINSTANCE hPrevInstance = GetHPrevInstance();
	LPWSTR lpCmdLine = GetLPCmdLine();
	int nCmdShow = GetNCmdShow();

	// Assert that the values returned are expected.
	assert(hInstance != nullptr);
	assert(hPrevInstance == nullptr);
	assert(lpCmdLine != nullptr);

	// Close the console window. This is not required, but if you do not need the console then it should be
	// freed in order to release the resources it is using. If you wish to keep the console open and use it
	// you can remove the call to FreeConsole. If you want to create a new console later you can call
	// AllocConsole. If you want to use an existing console you can call AttachConsole.
	// * FreeConsole();

	// ***********************
	// If you want to avoid creating a console in the first place, you can change the linker /SUBSYSTEM
	// option in the project properties to WINDOWS as documented here:
	// http://msdn.microsoft.com/en-us/library/fcc1zstk.aspx . If you do that you should comment out the
	// above call to FreeConsole since there will not be any console to free. The program will still
	// function properly. If you want the console back, change the /SUBSYSTEM option back to CONSOLE.
	// ***********************

	// Note: The remainder of the code in this file comes from the default Visual C++ Win32 Application
	// template (with a few minor alterations). It serves as an example that the program works, not as an
	// example of good, modern C++ code style.

	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);


	// Perform application initialization:
	return wWinMain(hInstance, hPrevInstance, lpCmdLine, nCmdShow);
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND	- process the application menu
//  WM_PAINT	- Paint the main window
//  WM_DESTROY	- post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int wmId, wmEvent;
	//PAINTSTRUCT ps;
	//HDC hdc;
	RECT clientRect;
	HINSTANCE hInstance = (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE);
	if (hInstance == NULL) return 0;

	switch (message)
	{
	case WM_COMMAND:
		wmId = LOWORD(wParam);
		wmEvent = HIWORD(wParam);
		// Parse the menu selections:
		switch (wmId)
		{
		case IDM_ABOUT:
			DialogBox(hInstance, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case IDM_EXIT:
			DestroyWindow(hWnd);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		break;
	case WM_SIZE:
		//Triggered when resized, continously during resize operation.
		GetClientRect(hWnd, &clientRect);
		break;
	case WM_EXITSIZEMOVE:
		// Triggered when moving/resize operation is finished.
		GetClientRect(hWnd, &clientRect);
		break;
	case WM_PAINT:
		//	Window API draw ('paint') command, the function is a default draw function found at:
		//	https://msdn.microsoft.com/en-us/library/windows/desktop/ff381401(v=vs.85).aspx
		//	The function seem to 'initialize' the window on creation, resize etc. 
		{
			PAINTSTRUCT ps;
			HDC hdc = BeginPaint(hWnd, &ps);

			// All painting occurs here, between BeginPaint and EndPaint.

			FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));

			EndPaint(hWnd, &ps);
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	HINSTANCE hInstance = (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE);
	if (hInstance)
	{
		UNREFERENCED_PARAMETER(lParam);
		switch (message)
		{
		case WM_INITDIALOG:
			return (INT_PTR)TRUE;

		case WM_COMMAND:
			if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
			{
				EndDialog(hWnd, LOWORD(wParam));
				return (INT_PTR)TRUE;
			}
			break;
		}
	}
	return (INT_PTR)FALSE;
}
