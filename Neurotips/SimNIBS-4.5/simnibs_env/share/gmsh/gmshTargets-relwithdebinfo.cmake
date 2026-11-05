#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gmsh::shared" for configuration "RelWithDebInfo"
set_property(TARGET gmsh::shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(gmsh::shared PROPERTIES
  IMPORTED_IMPLIB_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/gmsh.dll.lib"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/gmsh-4.15.dll"
  )

list(APPEND _cmake_import_check_targets gmsh::shared )
list(APPEND _cmake_import_check_files_for_gmsh::shared "${_IMPORT_PREFIX}/lib/gmsh.dll.lib" "${_IMPORT_PREFIX}/lib/gmsh-4.15.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
