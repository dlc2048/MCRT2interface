
macro(makePythonLink src dest)
  execute_process(
    COMMAND ln -sfn ${src} ${dest}
	RESULT_VARIABLE _EXIT_CODE
  )
  if (NOT _EXIT_CODE EQUAL 0)
    message(FATAL_ERROR "makePythonLink failed; output see ${CMAKE_SOURCE_DIR}/error.log")
  endif()
endmacro()

macro(makeWindowsBat src)
  set( target_file "${CMAKE_CURRENT_BINARY_DIR}/${src}.bat" )
  file(WRITE  ${target_file} "@echo off \n")
  file(APPEND ${target_file} "set file=%MCRT2_HOME%/RT2interface/${src} \n")
  file(APPEND ${target_file} "python \"%file%\" %*")
  install(FILES ${target_file} DESTINATION bin)
endmacro()

project(MCRT2interface LANGUAGES NONE)
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

# find_package(Python REQUIRED)

option(MCRT2_SET_SYMLINK "Set symbolic links for RT2 binary in UNIX system (need root)" OFF)
option(MCRT2_INSTALL_PYTHON_API "Install python api (rt2)" OFF)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fsource_cache.txt" ${VCPKG_PATH})

file(COPY "setup.py"         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY "requirements.txt" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY "src"              DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY "RT2dicom"         DESTINATION ${CMAKE_CURRENT_BINARY_DIR} 
  FILE_PERMISSIONS 
  OWNER_READ OWNER_WRITE OWNER_EXECUTE 
  GROUP_READ GROUP_EXECUTE 
  WORLD_READ WORLD_EXECUTE
)
file(COPY "RT2fluka"         DESTINATION ${CMAKE_CURRENT_BINARY_DIR} 
  FILE_PERMISSIONS 
  OWNER_READ OWNER_WRITE OWNER_EXECUTE 
  GROUP_READ GROUP_EXECUTE 
  WORLD_READ WORLD_EXECUTE
)
file(COPY "RT2geant4"         DESTINATION ${CMAKE_CURRENT_BINARY_DIR} 
  FILE_PERMISSIONS 
  OWNER_READ OWNER_WRITE OWNER_EXECUTE 
  GROUP_READ GROUP_EXECUTE 
  WORLD_READ WORLD_EXECUTE
)
file(COPY "RT2nifti"         DESTINATION ${CMAKE_CURRENT_BINARY_DIR} 
  FILE_PERMISSIONS 
  OWNER_READ OWNER_WRITE OWNER_EXECUTE 
  GROUP_READ GROUP_EXECUTE 
  WORLD_READ WORLD_EXECUTE
)
file(COPY "RT2fsource"         DESTINATION ${CMAKE_CURRENT_BINARY_DIR} 
  FILE_PERMISSIONS 
  OWNER_READ OWNER_WRITE OWNER_EXECUTE 
  GROUP_READ GROUP_EXECUTE 
  WORLD_READ WORLD_EXECUTE
)
file(COPY "RT2neutron"         DESTINATION ${CMAKE_CURRENT_BINARY_DIR} 
  FILE_PERMISSIONS 
  OWNER_READ OWNER_WRITE OWNER_EXECUTE 
  GROUP_READ GROUP_EXECUTE 
  WORLD_READ WORLD_EXECUTE
)

install(FILES 
  "RT2dicom" 
  "RT2fluka" 
  "RT2geant4" 
  "RT2nifti" 
  "RT2fsource" 
  "RT2neutron" 
  "RT2fsource_CMakeLists.txt" 
  "RT2fsource_template.cpp" 
  DESTINATION ${INSTALL_ROOT}RT2interface
)

file(COPY "RT2fsource_template.cpp"   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS 
  OWNER_READ OWNER_WRITE
  GROUP_READ
  WORLD_READ
)
file(COPY "RT2fsource_CMakeLists.txt" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS 
  OWNER_READ OWNER_WRITE
  GROUP_READ
  WORLD_READ
)

if (MCRT2_INSTALL_PYTHON_API)
    execute_process(
      COMMAND pip install -e .
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE   _EXIT_CODE
      )
    if (NOT _EXIT_CODE EQUAL 0)
        message(FATAL_ERROR "RT2 interface installation failed; output see ${CMAKE_SOURCE_DIR}/error.log")
    endif()
endif()

if( MCRT2_SET_SYMLINK )
  if( WIN32 )
    makeWindowsBat("RT2dicom"  )
    makeWindowsBat("RT2fluka"  )
    makeWindowsBat("RT2geant4" )
    makeWindowsBat("RT2nifti"  )
    makeWindowsBat("RT2fsource")
    makeWindowsBat("RT2neutron")
  else( WIN32 )
    if( UNIX AND NOT APPLE )
      makePythonLink(
        "${CMAKE_CURRENT_BINARY_DIR}/RT2dicom" 
        "/usr/local/bin"
      )
      makePythonLink(
        "${CMAKE_CURRENT_BINARY_DIR}/RT2fluka" 
        "/usr/local/bin"
      )
      makePythonLink(
        "${CMAKE_CURRENT_BINARY_DIR}/RT2geant4" 
        "/usr/local/bin"
      )
      makePythonLink(
        "${CMAKE_CURRENT_BINARY_DIR}/RT2nifti" 
        "/usr/local/bin"
      )
      makePythonLink(
        "${CMAKE_CURRENT_BINARY_DIR}/RT2fsource" 
        "/usr/local/bin"
      )
      makePythonLink(
        "${CMAKE_CURRENT_BINARY_DIR}/RT2neutron" 
        "/usr/local/bin"
      )
    else()
      # No rule for MAC OS
    endif()
  endif()
endif()
