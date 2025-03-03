project(stimage_tests)

include_directories(${STIMAGE_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(test_file_ext ".c")
set(test_file_pattern "test_*${test_file_ext}")

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/${test_file_pattern}")
foreach(test_srcfile ${files})
    get_filename_component(test_filename ${test_srcfile} NAME ABSOLUTE)
    string(REPLACE ".c" "" test_target ${test_filename})
    add_executable(${test_target} ${test_srcfile})
    target_link_libraries(${test_target} stimage)
    message(STATUS "Adding test: ${test_target}")
    add_test(target_${test_target} ${test_target} ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(target_${test_target} PROPERTIES SKIP_RETURN_CODE 127)
endforeach()
