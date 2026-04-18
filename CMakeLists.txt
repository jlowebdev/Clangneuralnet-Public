cmake_minimum_required(VERSION 3.10)
project(clangneuralnet C)

# Add an executable

include(FetchContent)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(a
    main.c
    structs.c
    functions.c
    
)
# target_link_libraries(a PRIVATE raylib)