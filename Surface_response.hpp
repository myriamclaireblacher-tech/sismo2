#ifndef SURFACE_RESPONSE
#define SURFACE_RESPONSE

#include "ODE.hpp"
#include <omp.h>

//one for each thread
struct ThreadWorkspace {
    Fault fault_solver;
    std::vector<sunrealtype> V_list;
};

int signal_surface(std::vector<Param>*pP, std::vector<sunrealtype>& t_list,  
                        std::vector<ThreadWorkspace>& workspaces,
                        Eigen::Matrix<double,3*Nstations,NSubFaults>& G, 
                        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix, 
                        Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

    


    #pragma omp parallel 
    {
        //get thread data
        int tid = omp_get_thread_num(); 
        ThreadWorkspace& ws = workspaces[tid];
        #pragma omp for
            for (size_t i = 0; i < NSubFaults; i++)
            {   ws.fault_solver.ODE_solver(t_list, ws.V_list, (*pP)[i]);
                compute_slip( storage_matrix.row(i) , t_list , ws.V_list);}

            
            
     }
    RES_matrix.noalias() = G * storage_matrix ;
    return 0;
        
    }
#endif