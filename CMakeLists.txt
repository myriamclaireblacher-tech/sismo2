cmake_minimum_required(VERSION 3.18)

project(sismo2 VERSION 1.0 LANGUAGES CXX)

# Force le standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. CHEMIN DE SUNDIALS
set(SUNDIALS_DIR "F:/stage/sundials_install")
set(CMAKE_PREFIX_PATH ${SUNDIALS_DIR})

#adding eigen
# (Note : Un dossier de build est requis par CMake car Eigen est en dehors du projet)
add_subdirectory("F:/stage/eigen" "${CMAKE_CURRENT_BINARY_DIR}/eigen_build")

# 2. CONFIGURATION DES SOURCES
set(SOURCES
    main.cpp
    ODE.cpp
)

set(HEADERS
    ODE.hpp
    variables.hpp
    param.hpp
    Surface_responce.hpp
)

# 3. RECHERCHE DE SUNDIALS
find_package(SUNDIALS REQUIRED COMPONENTS cvode nvecserial)

# 4. CRÉATION DE L'EXÉCUTABLE
add_executable(sismo2 ${SOURCES} ${HEADERS})

# 5. INCLUSIONS ET LIENS
target_include_directories(sismo2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sismo2 PUBLIC SUNDIALS::cvode SUNDIALS::nvecserial)

# Lier Eigen à votre exécutable (Corrigé pour cibler sismo2 après sa création)
target_link_libraries(sismo2 PRIVATE Eigen3::Eigen)

# =========================================================================
# AUTOMATISATION DE LA COPIE DES DLLs (OPTION 2 COMPLET)
# =========================================================================
set(DLL_CORE  "F:/stage/sundials_install/bin/libsundials_core.dll")
set(DLL_CVODE "F:/stage/sundials_install/bin/libsundials_cvode.dll")
set(DLL_NVEC  "F:/stage/sundials_install/bin/libsundials_nvecserial.dll")

add_custom_command(TARGET sismo2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${DLL_CORE}"
    "${DLL_CVODE}"
    "${DLL_NVEC}"
    $<TARGET_FILE_DIR:sismo2>
    COMMENT "CMake : Copie automatique des DLLs de SUNDIALS (Core, Cvode, Nvec) dans le dossier build..."
)

# Gestion des flags d'optimisation selon le compilateur pour éviter les erreurs
if(MSVC)
    target_compile_options(sismo2 PRIVATE /O2 /arch:AVX2)
    target_compile_definitions(sismo2 PRIVATE NDEBUG)
else()
    target_compile_options(sismo2 PRIVATE -O3 -march=native)
    target_compile_definitions(sismo2 PRIVATE NDEBUG)
endif()