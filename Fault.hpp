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
        Fault(){
                int retval = SUNContext_Create(SUN_COMM_NULL, &this->sunctx);
                if (check_retval(&retval, "SUNContext_Create", 1)){ throw std::runtime_error("\nError : cannot create SUNContext.");}
                this->y=N_VNew_Serial(NEQ, this->sunctx);
                if (check_retval((void*)y, "N_VNew_Serial", 0)) {throw std::runtime_error("\nError : cannot create N_vector y."); }
                this->abstol = N_VNew_Serial(NEQ, this->sunctx) ;
                if (check_retval((void*)abstol, "N_VNew_Serial", 0)) {throw std::runtime_error("\nError : cannot create N_vector abstol."); }
                sunrealtype* y_dot = N_VGetArrayPointer(this->abstol);
                y_dot[0]=ATOL1 ; //error on V
                y_dot[1]=ATOL2 ; //error on theta
                this->cvode_mem = CVodeCreate(CV_BDF, this->sunctx);
                if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) {throw std::runtime_error("\nError : cannot create cvode_mem."); }
                retval = CVodeInit(cvode_mem, f, SUN_RCONST(0.0), this->y);
                if (check_retval(&retval, "CVodeInit", 1)) { throw std::runtime_error("\nError : cannot init cvode_mem."); }
                retval = CVodeSVtolerances(cvode_mem, RTOL, abstol);
                if (check_retval(&retval, "CVodeSVtolerances", 1)) {  throw std::runtime_error("\nError : issues with tolerance."); }
                
                
                this->A = SUNDenseMatrix(NEQ, NEQ, this->sunctx);
                if (check_retval((void*)A, "SUNDenseMatrix", 0)) {throw std::runtime_error("\nError : cannot create SunMatrix A."); };
                this->LS = SUNLinSol_Dense(this->y, this->A, this->sunctx);
                if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { throw std::runtime_error("\nError : cannot create SUNLinsol LS."); }

        };
        ~Fault(){    
            N_VDestroy(y);            /* Free y vector */
            N_VDestroy(abstol);       /* Free abstol vector */
            CVodeFree(&cvode_mem);    /* Free CVODE memory */
            SUNLinSolFree(LS);        /* Free the linear solver memory */
            SUNMatDestroy(A);         /* Free the matrix memory */
            SUNContext_Free(&sunctx); /* Free the SUNDIALS context */};

};


//right hand side (RHS) of the EDO to resolve
int f(sunrealtype t, N_Vector y, N_Vector y_dot, void *user_data);

//Jacobian matrix of the RHS

int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);


//accessories for EDO resolution

int check_retval(void* returnvalue, const char* funcname, int opt);




#endif