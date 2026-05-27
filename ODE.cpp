#include <cstdio>
#include "ODE.hpp"

//additional general librairies

#include <cmath>
#include <stdio.h>
#include <fstream> 

//additional CVCode  librairies
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */

Fault::Fault(){
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
                retval = CVodeSetLinearSolver(this->cvode_mem, this->LS, this->A);
                if (check_retval(&retval, "CVodeSetLinearSolver", 1)) { throw std::runtime_error("\nError : cannot attach Linear Solver."); }
                retval = CVodeSetJacFn(this->cvode_mem, Jac);
                if (check_retval(&retval, "CVodeSetJacFn", 1)) { throw std::runtime_error("\nError : cannot attach Jacobian."); }
        }

Fault::~Fault(){    
            if (this->LS)        SUNLinSolFree(this->LS);           /* Free the linear solver memory */
            if (this->A)         SUNMatDestroy(this->A);            /* Free the matrix memory */
            if (this->cvode_mem) CVodeFree(&this->cvode_mem);       /* Free CVODE memory */
            if (this->abstol)    N_VDestroy(this->abstol);          /* Free abstol vector */
            if (this->y)         N_VDestroy(this->y);               /* Free y vector */
            if (this->sunctx)    SUNContext_Free(&this->sunctx);    /* Free the SUNDIALS context */}


int Fault::ODE_solver(const std::vector<sunrealtype>& t_list, Eigen::Ref<Eigen::RowVectorXd> slip_list, const Param& fault_param){
    const sunrealtype V1 = fault_param.V0_ * std::exp(fault_param.Dtau_asigma) ; //initial slip rate 
    const sunrealtype theta1 = 1.0/(fault_param.D_c_inv * fault_param.V0_ ); //initial theta

    sunrealtype* y_data = N_VGetArrayPointer(this->y); 
    y_data[0] = V1 ;   //slip rate component for fixed time t=1
    y_data[1] = theta1 ;  //state variable component for fixed time t=0

    int retval = CVodeReInit(cvode_mem, SUN_RCONST(0.0), this->y);
    if (check_retval(&retval, "CVodeReInit", 1)) { return retval; }

    // parameters of the fault
    retval = CVodeSetUserData(this->cvode_mem, const_cast<Param*>(&fault_param)); 
    if (check_retval(&retval, "CVodeSetUserData", 1)) { return 1; }

    sunrealtype t;
    sunrealtype Vprev=V1;
    slip_list[0] = 0.0 ;

    for (int i=1; i<t_list.size(); ++i){
        retval = CVode(cvode_mem, t_list[i], y, &t, CV_NORMAL);
        if (check_retval(&retval, "CVode", 1)) { break; }
        if (retval == CV_SUCCESS) {
            sunrealtype V_new = y_data[0] ;
            slip_list[i]=slip_list[i-1]+ 0.5 * (V_new + Vprev) * (t_list[i] - t_list[i-1]);
            Vprev=V_new; }
    }
    /* Print final statistics to the screen */
    if (rapport_EDO)
    {
        printf("\nFinal Statistics:\n");
        retval = CVodePrintAllStats(cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
    }

    return (retval);
}

int Fault::ODE_solver(const std::vector<sunrealtype>& t_list, std::vector<sunrealtype>& V_list, const Param& fault_param){
    const sunrealtype V1 = fault_param.V0_ * std::exp(fault_param.Dtau_asigma) ; //initial slip rate 
    const sunrealtype theta1 = 1.0/(fault_param.D_c_inv * fault_param.V0_ ); //initial theta
    //V_list.clear();                 // 1. On vide COMPLÈTEMENT le vecteur
    //V_list.reserve(t_list.size());  // 2. On alloue la mémoire brute pour 300 points
    V_list[0]=V1;

    sunrealtype* y_data = N_VGetArrayPointer(this->y); 
    y_data[0] = V1 ;   //slip rate component for fixed time t=1
    y_data[1] = theta1 ;  //state variable component for fixed time t=0

    int retval = CVodeReInit(cvode_mem, SUN_RCONST(0.0), this->y);
    if (check_retval(&retval, "CVodeReInit", 1)) { return retval; }

    // parameters of the fault
    retval = CVodeSetUserData(this->cvode_mem, const_cast<Param*>(&fault_param)); 
    if (check_retval(&retval, "CVodeSetUserData", 1)) { return 1; }

    sunrealtype t;

    for (int i=1; i<t_list.size(); ++i){
        retval = CVode(cvode_mem, t_list[i], y, &t, CV_NORMAL);
        if (check_retval(&retval, "CVode", 1)) { break; }
        if (retval == CV_SUCCESS) V_list[i]=y_data[0];
    }
    /* Print final statistics to the screen */
    if (rapport_EDO)
    {
        printf("\nFinal Statistics:\n");
        retval = CVodePrintAllStats(cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
    }

    return (retval);
}

int Fault::ODE_solver(std::ofstream& file, const std::vector<sunrealtype>& t_list,  const Param& fault_param){

    const sunrealtype V1 = fault_param.V0_ * std::exp(fault_param.Dtau_asigma) ; //initial slip rate 
    const sunrealtype theta1 = 1.0/(fault_param.D_c_inv * fault_param.V0_ ); //initial theta


    sunrealtype* y_data = N_VGetArrayPointer(this->y); 
    y_data[0] = V1 ;   //slip rate component for fixed time t=1
    y_data[1] = theta1 ;  //state variable component for fixed time t=0

    int retval = CVodeReInit(cvode_mem, SUN_RCONST(0.0), this->y);
    if (check_retval(&retval, "CVodeReInit", 1)) { return retval; }

    // parameters of the fault
    retval = CVodeSetUserData(this->cvode_mem, const_cast<Param*>(&fault_param)); 
    if (check_retval(&retval, "CVodeSetUserData", 1)) { return 1; }

    sunrealtype t;

    std::vector<SolValues> history_res;
    history_res.reserve(t_list.size());
    history_res.push_back({0.0, V1, theta1}); // Sauvegarde de t=0
    for (int i=1; i<t_list.size(); ++i){
        retval = CVode(this->cvode_mem, t_list[i], this->y, &t, CV_NORMAL);
        if (check_retval(&retval, "CVode", 1)) { break; }
        if (retval == CV_SUCCESS) history_res.push_back({t,y_data[0],y_data[1]});
    }
    /* Print final statistics to the screen */
    if (rapport_EDO)
    {
        printf("\nFinal Statistics:\n");
        retval = CVodePrintAllStats(this->cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
    }


    /* Write data out to the CSV file */
    if (file.is_open()) {
        file << "Time,V,Theta\n"; // CSV Header
        for (const auto& sol : history_res) {
            file << sol.t << "," << sol.V << "," << sol.theta << "\n";
        }
    } else {
        printf("\nWarning: Output file stream is not open. Data not saved.\n");
    }
    return (retval);

}

//compute slip
void compute_slip(std::vector<sunrealtype>& slip_list, const std::vector<sunrealtype>& t_list, const std::vector<sunrealtype>& V_list){
    slip_list[0]=0.0 ; //slip value at value t=0
    for (int i=0; i<t_list.size()-1; ++i){
        slip_list[i+1]=slip_list[i]+ 0.5 * (V_list[i] + V_list[i+1]) * ( t_list[i+1] - t_list [i])  ;
    }
    ;}

void compute_slip(Eigen::Ref<Eigen::RowVectorXd> slip_list, 
                  const std::vector<sunrealtype>& t_list, 
                  const std::vector<sunrealtype>& V_list) {
    slip_list[0] = 0.0;
    for (size_t i = 0; i < t_list.size() - 1; ++i) {
        slip_list[i+1] = slip_list[i] + 0.5 * (V_list[i] + V_list[i+1]) * (t_list[i+1] - t_list[i]);
    }
};

//save to csv
void save_to_csv(std::ofstream& file, const std::vector<sunrealtype>& t_list, const std::vector<sunrealtype>& slip_list, const std::vector<sunrealtype>& V_list){
    if (file.is_open()) {
        file << "Time,slip,V\n"; // CSV Header
        for (size_t i = 0; i < t_list.size(); ++i) {
            file << t_list[i] << "," << slip_list[i]<< "," << V_list[i] << "\n";
        }
        file.close(); 
    } else {
        fprintf(stderr, "\nWarning: Could not open file. Data not saved.\n");
    }
}

//RHS
int f(sunrealtype t, N_Vector y, N_Vector y_dot, void *user_data){
    Param* p = static_cast<Param*>(user_data);

    const sunrealtype* y_data = N_VGetArrayPointer(y);
    const sunrealtype V = y_data[0];
    const sunrealtype theta = y_data[1];

    sunrealtype* ydot_data = N_VGetArrayPointer(y_dot);
    const sunrealtype evolution_term = V * theta * p->D_c_inv;
    ydot_data[0] = p->k_a_sigma * V * (Vinf - V) - p->b_a * V * (1.0 - evolution_term) / theta ; //dV/dt
    ydot_data[1] = 1.0 - evolution_term ;                                                // Dtheta/dt
    return 0;
}

//Jacobian of the RHS
int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3){
    
    Param* p = static_cast<Param*>(user_data);
    const sunrealtype* y_data = N_VGetArrayPointer(y);
    const sunrealtype V = y_data[0];              //retrieve V
    const sunrealtype theta = y_data[1];          //retrieve theta
    sunrealtype* J_data = SUNDenseMatrix_Data(J) ;
    const sunrealtype coeffA=1.0/theta ;
    const sunrealtype coeffB=coeffA *p->b_a;
    J_data[0] =   p->coeff1 * V + p->coeff2 - coeffB ;
    J_data[1] =  -p->D_c_inv * theta ;
    J_data[2] =  V * coeffA * coeffB ;
    J_data[3] =  -p->D_c_inv * V ;
    return 0;
}

//accessories for EDO resolution
int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  /* Check if retval < 0 */
  else if (opt == 1)
  {
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return (1);
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL)
  {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}
