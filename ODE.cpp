#include "ODE.hpp"
#include "variables.hpp"
#include "param.hpp"

//additional general librairies
#include <iostream> 
#include <iomanip>
#include <cmath>
#include <stdio.h>

//additional CVCode  librairies
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */


int ODE_solver(std::ofstream& file, const std::vector<sunrealtype>& t_list, Param fault_param){
    const sunrealtype V1 = fault_param.V0_ * std::exp(fault_param.Dtau_asigma) ; //initial slip rate 
    const sunrealtype theta1 = 1.0/(fault_param.D_c_inv * fault_param.V0_ ); //initial theta
    const sunrealtype T0 = SUN_RCONST(0.0) ;
    SUNContext sunctx ;
    sunrealtype t ; 
    SUNLinearSolver Ls ;
    FILE* FID ; 
    int retval=SUNContext_Create


}


static int f(sunrealtype t, N_Vector y, N_Vector y_dot, void *user_data){
    Param* p = static_cast<Param*>(user_data);

    const sunrealtype* y_data = N_VGetArrayPointer(y);
    const sunrealtype V = y_data[0];
    const sunrealtype theta = y_data[1];

    sunrealtype* ydot_data = N_VGetArrayPointer(y_dot);
    const sunrealtype evolution_term = V * theta * p->D_c_inv;
    ydot_data[0] = p->k_a_sigma * V * (Vinf - V) + p->b_a * V * (1.0 - evolution_term) ; //dV/dt
    ydot_data[1] = 1.0 - evolution_term ;                                                // Dtheta/dt
    return 0;
}


static int jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3){
    
    Param* p = static_cast<Param*>(user_data);
    const sunrealtype* y_data = N_VGetArrayPointer(y);
    const sunrealtype V = y_data[0];              //retrieve V
    const sunrealtype theta = y_data[1];          //retrieve theta
    sunrealtype* J_data = SUNDenseMatrix_Data(J) ;
    J_data[0] =   p->k_a_sigma * (-2.0 * V + Vinf ) + p->b_a - 2.0 * theta * p->coeff1 ;
    J_data[1] =  -p->D_c_inv * theta ;
    J_data[2] =  -V * V * p->coeff1 ;
    J_data[3] =  -p->D_c_inv * V ;
    return 0;
}
