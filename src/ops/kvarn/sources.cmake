target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/codec.cu"
  "${CMAKE_CURRENT_LIST_DIR}/attention.cu"
  "${CMAKE_CURRENT_LIST_DIR}/decode.cu")
