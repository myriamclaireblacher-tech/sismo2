#ifndef RESOL_EDO
#define RESOL_EDO

//general librairies
#include <list>
#include <vector>
#include <iostream> 
#include <iomanip>

//CVCode  librairies
#include <cvode/cvode.h>            /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <stdio.h>
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */

//Solution of rate&state ODE for fixed t
struct SolValues{
    sunrealtype t;
    sunrealtype V;
    sunrealtype theta;
};

//Parameters of the fault
struct Param{
    double k_a_simga;           // k/ (a sigma)
    double b_a;                 //b sigma / (a simag)
    double D_c_inv;             // 1/D_c
    double exp_Dtau_asigma;     //exp (Delta Tau/ (a sigma) )
    double V0_;                 // V0_
};                              //vinf in VARIABLES.hpp

//right hand side of the EDO to resolve
static int f(sunrealtype t, N_Vector y, N_Vector y_dot, void *user_data);

int ODE_solver(std::ofstream& file, const std::vector<sunrealtype>& t_list, Param fault_param)


#endif
