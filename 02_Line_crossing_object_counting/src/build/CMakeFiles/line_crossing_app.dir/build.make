# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build

# Include any dependencies generated for this target.
include CMakeFiles/line_crossing_app.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/line_crossing_app.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/line_crossing_app.dir/flags.make

CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.o: ../Line_crossing.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/Line_crossing.cpp

CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/Line_crossing.cpp > CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.i

CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/Line_crossing.cpp -o CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.s

CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.o: ../MeraDrpRuntimeWrapper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/MeraDrpRuntimeWrapper.cpp

CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/MeraDrpRuntimeWrapper.cpp > CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.i

CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/MeraDrpRuntimeWrapper.cpp -o CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.s

CMakeFiles/line_crossing_app.dir/box.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/box.cpp.o: ../box.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/line_crossing_app.dir/box.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/box.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/box.cpp

CMakeFiles/line_crossing_app.dir/box.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/box.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/box.cpp > CMakeFiles/line_crossing_app.dir/box.cpp.i

CMakeFiles/line_crossing_app.dir/box.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/box.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/box.cpp -o CMakeFiles/line_crossing_app.dir/box.cpp.s

CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.o: ../kalman_filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/kalman_filter.cpp

CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/kalman_filter.cpp > CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.i

CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/kalman_filter.cpp -o CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.s

CMakeFiles/line_crossing_app.dir/matrix.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/matrix.cpp.o: ../matrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/line_crossing_app.dir/matrix.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/matrix.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/matrix.cpp

CMakeFiles/line_crossing_app.dir/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/matrix.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/matrix.cpp > CMakeFiles/line_crossing_app.dir/matrix.cpp.i

CMakeFiles/line_crossing_app.dir/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/matrix.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/matrix.cpp -o CMakeFiles/line_crossing_app.dir/matrix.cpp.s

CMakeFiles/line_crossing_app.dir/munkres.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/munkres.cpp.o: ../munkres.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/line_crossing_app.dir/munkres.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/munkres.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/munkres.cpp

CMakeFiles/line_crossing_app.dir/munkres.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/munkres.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/munkres.cpp > CMakeFiles/line_crossing_app.dir/munkres.cpp.i

CMakeFiles/line_crossing_app.dir/munkres.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/munkres.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/munkres.cpp -o CMakeFiles/line_crossing_app.dir/munkres.cpp.s

CMakeFiles/line_crossing_app.dir/track.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/track.cpp.o: ../track.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/line_crossing_app.dir/track.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/track.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/track.cpp

CMakeFiles/line_crossing_app.dir/track.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/track.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/track.cpp > CMakeFiles/line_crossing_app.dir/track.cpp.i

CMakeFiles/line_crossing_app.dir/track.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/track.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/track.cpp -o CMakeFiles/line_crossing_app.dir/track.cpp.s

CMakeFiles/line_crossing_app.dir/tracker.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/tracker.cpp.o: ../tracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/line_crossing_app.dir/tracker.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/tracker.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/tracker.cpp

CMakeFiles/line_crossing_app.dir/tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/tracker.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/tracker.cpp > CMakeFiles/line_crossing_app.dir/tracker.cpp.i

CMakeFiles/line_crossing_app.dir/tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/tracker.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/tracker.cpp -o CMakeFiles/line_crossing_app.dir/tracker.cpp.s

CMakeFiles/line_crossing_app.dir/wayland.cpp.o: CMakeFiles/line_crossing_app.dir/flags.make
CMakeFiles/line_crossing_app.dir/wayland.cpp.o: ../wayland.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/line_crossing_app.dir/wayland.cpp.o"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/line_crossing_app.dir/wayland.cpp.o -c /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/wayland.cpp

CMakeFiles/line_crossing_app.dir/wayland.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/line_crossing_app.dir/wayland.cpp.i"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/wayland.cpp > CMakeFiles/line_crossing_app.dir/wayland.cpp.i

CMakeFiles/line_crossing_app.dir/wayland.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/line_crossing_app.dir/wayland.cpp.s"
	/opt/poky/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++ --sysroot=/opt/poky/3.1.21/sysroots/aarch64-poky-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/wayland.cpp -o CMakeFiles/line_crossing_app.dir/wayland.cpp.s

# Object files for target line_crossing_app
line_crossing_app_OBJECTS = \
"CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.o" \
"CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.o" \
"CMakeFiles/line_crossing_app.dir/box.cpp.o" \
"CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.o" \
"CMakeFiles/line_crossing_app.dir/matrix.cpp.o" \
"CMakeFiles/line_crossing_app.dir/munkres.cpp.o" \
"CMakeFiles/line_crossing_app.dir/track.cpp.o" \
"CMakeFiles/line_crossing_app.dir/tracker.cpp.o" \
"CMakeFiles/line_crossing_app.dir/wayland.cpp.o"

# External object files for target line_crossing_app
line_crossing_app_EXTERNAL_OBJECTS =

line_crossing_app: CMakeFiles/line_crossing_app.dir/Line_crossing.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/MeraDrpRuntimeWrapper.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/box.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/kalman_filter.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/matrix.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/munkres.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/track.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/tracker.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/wayland.cpp.o
line_crossing_app: CMakeFiles/line_crossing_app.dir/build.make
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_gapi.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_stitching.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_aruco.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_bgsegm.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_bioinspired.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_ccalib.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_dpm.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_face.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_fuzzy.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_hfs.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_img_hash.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_line_descriptor.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_quality.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_reg.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_rgbd.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_saliency.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_sfm.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_stereo.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_structured_light.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_superres.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_surface_matching.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_tracking.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_videostab.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_xfeatures2d.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_xobjdetect.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_xphoto.so.4.1.0
line_crossing_app: /home/sharady/ai_sdk_work/drp-ai_tvm/tvm/build_runtime/libtvm_runtime.so
line_crossing_app: /home/sharady/ai_sdk_work/drp-ai_tvm/tvm/build_runtime/libtvm_runtime.so
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_shape.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_datasets.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_ml.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_plot.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_phase_unwrapping.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_optflow.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_ximgproc.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_video.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_objdetect.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_calib3d.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_features2d.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_highgui.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_flann.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_videoio.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_imgcodecs.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_photo.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_imgproc.so.4.1.0
line_crossing_app: /opt/poky/3.1.21/sysroots/aarch64-poky-linux/usr/lib64/libopencv_core.so.4.1.0
line_crossing_app: CMakeFiles/line_crossing_app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable line_crossing_app"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/line_crossing_app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/line_crossing_app.dir/build: line_crossing_app

.PHONY : CMakeFiles/line_crossing_app.dir/build

CMakeFiles/line_crossing_app.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/line_crossing_app.dir/cmake_clean.cmake
.PHONY : CMakeFiles/line_crossing_app.dir/clean

CMakeFiles/line_crossing_app.dir/depend:
	cd /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build /home/sharady/ai_sdk_work/drp-ai_tvm/02_Line_crossing_yolov3_wayland_02_feb_working/02_Line_crossing_yolov3_wayland_02_feb/02_Line_crossing/src/build/CMakeFiles/line_crossing_app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/line_crossing_app.dir/depend

