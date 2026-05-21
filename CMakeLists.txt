cmake_minimum_required(VERSION 3.18)

project(sismo2 VERSION 1.0 LANGUAGES CXX)

# Force le standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. CHEMIN DE SUNDIALS
set(SUNDIALS_DIR "F:/stage/sundials_install")
set(CMAKE_PREFIX_PATH ${SUNDIALS_DIR})

# 2. CONFIGURATION DES SOURCES
set(SOURCES
    main.cpp
    ODE.cpp
)

set(HEADERS
    ODE.hpp
    variables.hpp
    param.hpp
)

# 3. RECHERCHE DE SUNDIALS
find_package(SUNDIALS REQUIRED COMPONENTS cvode nvecserial)

# 4. CRÉATION DE L'EXÉCUTABLE
add_executable(sismo2 ${SOURCES} ${HEADERS})

# 5. INCLUSIONS ET LIENS
target_include_directories(sismo2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sismo2 PUBLIC SUNDIALS::cvode SUNDIALS::nvecserial)

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