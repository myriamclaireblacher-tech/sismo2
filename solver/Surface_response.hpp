#ifndef SURFACE_RESPONSE
#define SURFACE_RESPONSE

#include "ODE.hpp"
#include <omp.h>


/*

int surface_response_parallel(std::vector<Param>& pP, std::vector<sunrealtype>& t_list,  
                        std::vector<Fault>& fault_per_thread,
                        Eigen::Matrix<double,3*Nstations,NSubFaults>& G, 
                        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix, 
                        Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

    #pragma omp parallel 
    {
        //get thread data
        
        
        #pragma omp for schedule(dynamic, 10) 
            for (size_t i = 0; i < NSubFaults; i++)
            {   int tid = omp_get_thread_num(); 
                fault_per_thread[tid].ODE_solver(t_list, storage_matrix.row(i), pP[i]);}
            
    }
    
    RES_matrix.noalias() = G * storage_matrix ;
        
   return 0;       
};
   
*/

int surface_response(std::vector<Param>& pP, const std::vector<sunrealtype>& t_list,  
                        Fault & fault,
                        const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, 
                        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix, 
                        Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix);

int surface_response_subfault(int subfault, std::vector<Param>& pP, const std::vector<sunrealtype>& t_list,  
                        Fault & fault,
                        const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, 
                        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix, 
                        Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix);

#endif