add_executable(build_graph build_graph.cpp)

llvm_update_compile_flags(build_graph)

target_link_libraries(build_graph
  PRIVATE
  MLIRAttnDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(build_graph)
