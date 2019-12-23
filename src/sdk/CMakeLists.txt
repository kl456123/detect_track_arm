
aux_source_directory(. MODULE_SRCS)

add_library(SDK ${MODULE_SRCS})
target_link_libraries(SDK -lindem -lusbdriver)
