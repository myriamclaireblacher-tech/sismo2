#ifndef SURFACE_RESPONSE
#define SURFACE_RESPONSE

#include "ODE.hpp"


int signal_surface(std::vector<Param>*pP, std::vector<sunrealtype>& t_list,  std::vector<sunrealtype>& V_list,
                        Eigen::Matrix<double,,NSubFaults, 3*Nstations>& G, 
                        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix, 
                        Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){
    for (size_t i = 0; i < NSubFaults; i++)
    {
        ODE_solver(t_list,V_list, (*pP)[i]);
        compute_slip( storage_matrix.row(i) , t_list , V_list);
    }
    RES_matrix.noalias() = G * storage_matrix ;
    return 0;
    }

#endif