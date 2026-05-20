#ifndef RESOL_EDO
#define RESOL_EDO

//general librairies
#include <vector>
#include <fstream>


//CVCode  librairies
#include <cvode/cvode.h>            /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */



//Parameters of the fault
struct Param{
    double k_a_simga;           // k/ (a sigma)
    double b_a;                 //b sigma / (a simag)
    double D_c_inv;             // 1/D_c
    double exp_Dtau_asigma;     //exp (Delta Tau/ (a sigma) )
    double V0_;                 // V0_
};    

//ODE solver
int ODE_solver(std::ofstream& file, const std::vector<sunrealtype>& t_list, Param fault_param);


//Solution of rate&state ODE for fixed t
struct SolValues{
    sunrealtype t;
    sunrealtype V;
    sunrealtype theta;
};

                          //vinf in VARIABLES.hpp

//right hand side (RHS) of the EDO to resolve
static int f(sunrealtype t, N_Vector y, N_Vector y_dot, void *user_data);

//Jacobian matrix of the RHS

static int jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);



#endif
