#ifndef SA_HPP
#define SA_HPP

#include "inversion.hpp"

std::vector<Param> SA (int niter, int npert,  const std::vector<sunrealtype>& t_list, const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, 
    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const PT_param PT,
       double T = 0.01, int seed= 3456789);

#endif