# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\bowet\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7442.42\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\bowet\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7442.42\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\bowet\Documents\GitHub\projetAnnuel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\projetAnnuel.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\projetAnnuel.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\projetAnnuel.dir\flags.make

CMakeFiles\projetAnnuel.dir\library.c.obj: CMakeFiles\projetAnnuel.dir\flags.make
CMakeFiles\projetAnnuel.dir\library.c.obj: ..\library.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/projetAnnuel.dir/library.c.obj"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) /FoCMakeFiles\projetAnnuel.dir\library.c.obj /FdCMakeFiles\projetAnnuel.dir\ /FS -c C:\Users\bowet\Documents\GitHub\projetAnnuel\library.c
<<

CMakeFiles\projetAnnuel.dir\library.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/projetAnnuel.dir/library.c.i"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe > CMakeFiles\projetAnnuel.dir\library.c.i @<<
 /nologo $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\bowet\Documents\GitHub\projetAnnuel\library.c
<<

CMakeFiles\projetAnnuel.dir\library.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/projetAnnuel.dir/library.c.s"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) /FoNUL /FAs /FaCMakeFiles\projetAnnuel.dir\library.c.s /c C:\Users\bowet\Documents\GitHub\projetAnnuel\library.c
<<

CMakeFiles\projetAnnuel.dir\library.cpp.obj: CMakeFiles\projetAnnuel.dir\flags.make
CMakeFiles\projetAnnuel.dir\library.cpp.obj: ..\library.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/projetAnnuel.dir/library.cpp.obj"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\projetAnnuel.dir\library.cpp.obj /FdCMakeFiles\projetAnnuel.dir\ /FS -c C:\Users\bowet\Documents\GitHub\projetAnnuel\library.cpp
<<

CMakeFiles\projetAnnuel.dir\library.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/projetAnnuel.dir/library.cpp.i"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe > CMakeFiles\projetAnnuel.dir\library.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\bowet\Documents\GitHub\projetAnnuel\library.cpp
<<

CMakeFiles\projetAnnuel.dir\library.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/projetAnnuel.dir/library.cpp.s"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\projetAnnuel.dir\library.cpp.s /c C:\Users\bowet\Documents\GitHub\projetAnnuel\library.cpp
<<

CMakeFiles\projetAnnuel.dir\main.cpp.obj: CMakeFiles\projetAnnuel.dir\flags.make
CMakeFiles\projetAnnuel.dir\main.cpp.obj: ..\main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/projetAnnuel.dir/main.cpp.obj"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\projetAnnuel.dir\main.cpp.obj /FdCMakeFiles\projetAnnuel.dir\ /FS -c C:\Users\bowet\Documents\GitHub\projetAnnuel\main.cpp
<<

CMakeFiles\projetAnnuel.dir\main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/projetAnnuel.dir/main.cpp.i"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe > CMakeFiles\projetAnnuel.dir\main.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\bowet\Documents\GitHub\projetAnnuel\main.cpp
<<

CMakeFiles\projetAnnuel.dir\main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/projetAnnuel.dir/main.cpp.s"
	C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\projetAnnuel.dir\main.cpp.s /c C:\Users\bowet\Documents\GitHub\projetAnnuel\main.cpp
<<

# Object files for target projetAnnuel
projetAnnuel_OBJECTS = \
"CMakeFiles\projetAnnuel.dir\library.c.obj" \
"CMakeFiles\projetAnnuel.dir\library.cpp.obj" \
"CMakeFiles\projetAnnuel.dir\main.cpp.obj"

# External object files for target projetAnnuel
projetAnnuel_EXTERNAL_OBJECTS =

projetAnnuel.exe: CMakeFiles\projetAnnuel.dir\library.c.obj
projetAnnuel.exe: CMakeFiles\projetAnnuel.dir\library.cpp.obj
projetAnnuel.exe: CMakeFiles\projetAnnuel.dir\main.cpp.obj
projetAnnuel.exe: CMakeFiles\projetAnnuel.dir\build.make
projetAnnuel.exe: CMakeFiles\projetAnnuel.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable projetAnnuel.exe"
	C:\Users\bowet\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7442.42\bin\cmake\win\bin\cmake.exe -E vs_link_exe --intdir=CMakeFiles\projetAnnuel.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\mt.exe --manifests -- C:\PROGRA~2\MICROS~4\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\link.exe /nologo @CMakeFiles\projetAnnuel.dir\objects1.rsp @<<
 /out:projetAnnuel.exe /implib:projetAnnuel.lib /pdb:C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug\projetAnnuel.pdb /version:0.0 /machine:x64 /debug /INCREMENTAL /subsystem:console  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\projetAnnuel.dir\build: projetAnnuel.exe

.PHONY : CMakeFiles\projetAnnuel.dir\build

CMakeFiles\projetAnnuel.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\projetAnnuel.dir\cmake_clean.cmake
.PHONY : CMakeFiles\projetAnnuel.dir\clean

CMakeFiles\projetAnnuel.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\bowet\Documents\GitHub\projetAnnuel C:\Users\bowet\Documents\GitHub\projetAnnuel C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug C:\Users\bowet\Documents\GitHub\projetAnnuel\cmake-build-debug\CMakeFiles\projetAnnuel.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\projetAnnuel.dir\depend
