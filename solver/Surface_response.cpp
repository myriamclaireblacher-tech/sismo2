#include "Surface_response.hpp" 


int surface_response(std::vector<Param>& pP, const std::vector<sunrealtype>& t_list,  
                        Fault & fault,
                        const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, 
                        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix, 
                        Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){
        
    for (size_t i = 0; i < NSubFaults; i++)
    {   
        int retval = fault.ODE_solver(t_list, storage_matrix.row(i), pP[i]);
        if (retval < 0) return retval; }
            
    RES_matrix.noalias() = G * storage_matrix ;
        
   return 0;       
}



int surface_response_subfault(int subfault, std::vector<Param>& pP, const std::vector<sunrealtype>& t_list,  
                        Fault & fault,
                        const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, 
                        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix, 
                        Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){
        
     
    int retval = fault.ODE_solver(t_list, storage_matrix.row(subfault), pP[subfault]);
    if (retval < 0) return retval; 
    RES_matrix.noalias() = G * storage_matrix ;
        
   return 0;       
}