#ifndef SURFACE_RESPONSE
#define SURFACE_RESPONSE

#include "ODE.hpp"
#include <Eigen/Dense>


int signal_surface(vector<Param>*pP, Eigen::Matrix<double,3*Nstations,NsubFaults> G, vector<sunrealtype>& t_list,  vector<sunrealtype>& V_list, Eigen::Matrix<double, 3*Nstations, t_list.size()>& RES_matrix, Eigen::Matrix<double, 3*Nstations, t_list.size()>& INT_matrix
                           Eigen::Matrix <double, 3*Nstations, NSubFaults > & useless_matrix ){
    for (size_t i = 0; i < NSubFaults; i++)
    {
        ODE_solver(t_list,V_list, pP[i]);
        compute_slip( useless_matrix( t_list.size()*i ) , t_list , V_list);
    }
    RES_matrix.noalias() = G * useless_matrix ;
    
    };


#endif