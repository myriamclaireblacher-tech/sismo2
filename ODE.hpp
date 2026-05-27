#ifndef RESOL_EDO
#define RESOL_EDO

//general librairies
#include <vector>
#include <fstream>
#include <stdexcept>
#include "param.hpp"


//CVCode  librairies
#include <cvode/cvode.h>            /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */

//Eigen librairies
#include <Eigen/Dense>

constexpr sunindextype NEQ = 2 ; //number of equations to solve

//precision for EDO resolution
constexpr double RTOL = SUN_RCONST(10e-4);
constexpr double ATOL1 = SUN_RCONST(10e-8);
constexpr double ATOL2 = SUN_RCONST(10e-8);

class Fault{
    private :
        //tools to resolve EDO
        SUNContext sunctx; //
        N_Vector y; //
        N_Vector abstol;
        SUNMatrix A;
        SUNLinearSolver LS;
        void* cvode_mem ; 
    public:
        Fault();
    
        ~Fault();
        
        int ODE_solver(const std::vector<sunrealtype>& t_list, std::vector<sunrealtype>& V_list, const Param& fault_param);
        int ODE_solver(std::ofstream& file, const std::vector<sunrealtype>& t_list, const Param& fault_param);
        int ODE_solver(const std::vector<sunrealtype>& t_list, Eigen::Ref<Eigen::RowVectorXd> slip_list, const Param& fault_param);
};

//slip_list computed with V_list, starting at 0
void compute_slip(std::vector<sunrealtype>& slip_list, const std::vector<sunrealtype>& t_list, const std::vector<sunrealtype>& V_list);

void compute_slip(Eigen::Ref<Eigen::RowVectorXd> slip_list, const std::vector<sunrealtype>& t_list, const std::vector<sunrealtype>& V_list);


//t_list, slip_list and v_list to csv
void save_to_csv(std::ofstream& file, const std::vector<sunrealtype>& t_list, const std::vector<sunrealtype>& slip_list, const std::vector<sunrealtype>& V_list);
//right hand side (RHS) of the EDO to resolve
int f(sunrealtype t, N_Vector y, N_Vector y_dot, void *user_data);

//Jacobian matrix of the RHS

int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);


//accessories for EDO resolution

int check_retval(void* returnvalue, const char* funcname, int opt);

struct SolValues{
    sunrealtype t;
    sunrealtype V;
    sunrealtype theta;
};


#endif