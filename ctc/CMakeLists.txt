include_directories(include)
file(GLOB SRC src/*.cpp)

add_library(ctc SHARED ${SRC})

