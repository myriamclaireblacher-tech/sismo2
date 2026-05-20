#include "ODE.hpp"
#include "variables.hpp"

//additional general librairies
#include <iostream> 
#include <iomanip>

//additional CVCode  librairies
#include <stdio.h>
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */


int ODE_solver(std::ofstream& file, const std::vector<sunrealtype>& t_list, Param fault_param){


};


static int f(sunrealtype t, N_Vector y, N_Vector y_dot, void *user_data){
    Param* p = static_cast<Param*>(user_data);

    const sunrealtype* y_data = N_VGetArrayPointer(y);
    const sunrealtype V = y_data[0];
    const sunrealtype theta = y_data[1];

    sunrealtype* ydot_data = N_VGetArrayPointer(y_dot);
    const sunrealtype evolution_term = V * theta * p->D_c_inv;
    ydot_data[0] = p->k_a_simga * V * (Vinf - V) + p->b_a * V * (1.0 - evolution_term) ; //dV/dt
    ydot_data[1] = 1.0 - evolution_term ;                                                // Dtheta/dt
    return 0;
};


static int jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3){
    
    Param* p = static_cast<Param*>(user_data);
    const sunrealtype* y_data = N_VGetArrayPointer(y);
    const sunrealtype V = y_data[0];              //retrieve V
    const sunrealtype theta = y_data[1];          //retrieve theta
    sunrealtype* J_data = SUNDenseMatrix_Data(J) ;
    j_data[0] = 
    
};
