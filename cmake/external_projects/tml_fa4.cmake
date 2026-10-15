include(FetchContent)

if(DEFINED ENV{TML_FA4_SRC_DIR})
  set(TML_FA4_SRC_DIR $ENV{TML_FA4_SRC_DIR})
endif()

if(TML_FA4_SRC_DIR)
  FetchContent_Declare(
    tml_fa4
    SOURCE_DIR ${TML_FA4_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND "")
else()
  FetchContent_Declare(
    tml_fa4
    GIT_REPOSITORY https://github.com/HelloKS/tml-fa4.git
    # feat/sm120-sm121-native-fp8: native FP8 MMA and relative bias on SM12x.
    GIT_TAG 31062b9a7b3a896e13b6949f12e9a405c7bb2651
    GIT_PROGRESS TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND "")
endif()

FetchContent_GetProperties(tml_fa4)
if(NOT tml_fa4_POPULATED)
  FetchContent_Populate(tml_fa4)
endif()
message(STATUS "tml-fa4 is available at ${tml_fa4_SOURCE_DIR}")

add_custom_target(tml_fa4)

# Install into a private namespace so this implementation cannot shadow the
# flash_attn package used by vLLM's standard attention backends.
install(CODE "
  file(GLOB_RECURSE TML_FA4_PY_FILES
    \"${tml_fa4_SOURCE_DIR}/flash_attn/cute/*.py\")
  foreach(SRC_FILE \${TML_FA4_PY_FILES})
    file(RELATIVE_PATH REL_PATH
      \"${tml_fa4_SOURCE_DIR}/flash_attn/cute\" \${SRC_FILE})
    set(DST_FILE
      \"\${CMAKE_INSTALL_PREFIX}/vllm/third_party/tml_fa4/\${REL_PATH}\")
    get_filename_component(DST_DIR \${DST_FILE} DIRECTORY)
    file(MAKE_DIRECTORY \${DST_DIR})
    file(READ \${SRC_FILE} FILE_CONTENTS)
    string(REPLACE
      \"flash_attn.cute\"
      \"vllm.third_party.tml_fa4\"
      FILE_CONTENTS \"\${FILE_CONTENTS}\")
    file(WRITE \${DST_FILE} \"\${FILE_CONTENTS}\")
  endforeach()
" COMPONENT tml_fa4)
