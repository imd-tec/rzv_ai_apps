cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(05_Age_gender_detection)

# Link against TVM runtime
add_library(tvm_headers INTERFACE)
target_link_libraries(tvm_headers INTERFACE tvm_runtime) # Closed source library that is provided in the Yocto SDK
set(EXE_NAME "age_gender_detection_app")

file(GLOB SRC ./*.cpp ./*.h)
add_executable(${EXE_NAME} ${SRC})
# Use SDL2 and OpenGL3 backend
add_library(ImGui STATIC
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
    imgui/backends/imgui_impl_opengl3.cpp
    imgui/backends/imgui_impl_sdl2.cpp
	imgui/imgui_demo.cpp
    
)
add_executable(memcpy_test gl_demo/memcpy.cpp)
add_library(ImPlot STATIC
	implot/implot_items.cpp
	implot/implot.cpp
	implot/implot.h
)
find_package(SDL2 REQUIRED)
find_library(DL_LIBRARIES NAMES dl)
find_package(OpenGL )
# SDL2 should be installed into the Yocto SDK, SDL2 include dirs will normally be set to /usr/include/SDL2
# Which won't work if you are using the Yocto SDK, hence you need to fix this using a cmake toolchain file 
target_include_directories(ImGui PUBLIC ${SDL2_INCLUDE_DIRS}  imgui imgui/backends)
target_include_directories(ImPlot PUBLIC implot/)
target_link_libraries(ImGui PUBLIC ${SDL2_LIBRARIES} GLESv2  ${OPENGL_LIBRARIES} )
target_link_libraries(ImGui PUBLIC ${DL_LIBRARIES})
target_link_libraries(ImPlot PUBLIC ImGui)

add_executable(demo gl_demo/gl_demo.cpp )
target_compile_definitions(demo PUBLIC IMGUI_IMPL_OPENGL_ES2)
target_compile_definitions(ImGui PUBLIC IMGUI_IMPL_OPENGL_ES2)
target_compile_definitions(${EXE_NAME} PUBLIC IMGUI_IMPL_OPENGL_ES2)
target_compile_definitions(ImPlot PUBLIC IMGUI_IMPL_OPENGL_ES2)
message(STATUS "Using OpenGL libraries ${OPENGL_LIBRARIES}")
target_link_libraries(demo PUBLIC  ImGui)

TARGET_LINK_LIBRARIES(${EXE_NAME} pthread glib-2.0 mmngr gobject-2.0 gstreamer-1.0  gstbase-1.0 gstapp-1.0 
	opencv_imgcodecs opencv_imgproc opencv_core opencv_highgui
	jpeg webp tiff z tbb gtk-3 png16 gdk-3 cairo
	lzma rt cairo-gobject
	xkbcommon wayland-cursor wayland-egl wayland-client epoxy
	fribidi harfbuzz fontconfig
	glib-2.0 gobject-2.0 gdk_pixbuf-2.0 gmodule-2.0 pangocairo-1.0
	atk-1.0 gio-2.0 pango-1.0 freetype pixman-1 uuid pcre
	mount resolv expat pangoft2-1.0 blkid
	EGL GLESv2 mmngrbuf ImGui ImPlot)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(${EXE_NAME} ${OpenCV_LIBS})
target_link_libraries(${EXE_NAME} tvm_headers)
# Add in SDL2
target_compile_definitions(${EXE_NAME} PRIVATE -DIMAGE_MODE)
if(V2H)
    target_compile_definitions(${EXE_NAME} PRIVATE V2H)
endif()
