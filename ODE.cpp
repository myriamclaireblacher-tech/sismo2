#include <cstdio>
#include "ODE.hpp"

//additional general librairies

#include <cmath>
#include <stdio.h>
#include <fstream> 

//additional CVCode  librairies
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */


//with file output
int ODE_solver(std::ofstream& file, const std::vector<sunrealtype>& t_list, Param& fault_param){
    const sunrealtype V1 = fault_param.V0_ * std::exp(fault_param.Dtau_asigma) ; //initial slip rate 
    const sunrealtype theta1 = 1.0/(fault_param.D_c_inv * fault_param.V0_ ); //initial theta
    const sunrealtype T0 = SUN_RCONST(0.0) ;
    SUNContext sunctx ;

    /* Create the SUNDIALS context */
    int retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) { return (1); }

    
    /* Initial conditions */
    N_Vector y = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)y, "N_VNew_Serial", 0)) { return (1); }

    sunrealtype* y_data = N_VGetArrayPointer(y); /////////////////////////////////
    y_data[0] = V1 ;   //slip rate component for fixed time t=1
    y_data[1] = theta1 ;  //state variable component for fixed time t=0

    N_Vector abstol = N_VNew_Serial(NEQ, sunctx) ;
    if (check_retval((void*)abstol, "N_VNew_Serial", 0)) { return (1); }

    sunrealtype* y_dot = N_VGetArrayPointer(abstol);
    y_dot[0]=ATOL1 ; //error on V
    y_dot[1]=ATOL2 ; //error on theta

    /* Call CVodeCreate to create the solver memory and specify the
    * Backward Differentiation Formula */
    void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) { return (1); }

        /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in y'=f(t,y), the initial time T0, and
    * the initial dependent variable vector y. */
    retval = CVodeInit(cvode_mem, f, T0, y);
    if (check_retval(&retval, "CVodeInit", 1)) { return (1); }

        /* Call CVodeSVtolerances to specify the scalar relative tolerance
    * and vector abstolute tolerances */
    retval = CVodeSVtolerances(cvode_mem, RTOL, abstol);
    if (check_retval(&retval, "CVodeSVtolerances", 1)) { return (1); }


    // parameters of the fault
    retval = CVodeSetUserData(cvode_mem, (void*)&fault_param); ///////////////
    if (check_retval(&retval, "CVodeSetUserData", 1)) { return 1; }

    /* Create dense SUNMatrix for use in linear solves */
    SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, sunctx);
    if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (1); }

    /* Create dense SUNLinearSolver object for use by CVode */
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (1); }

    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolver(cvode_mem, LS, A); /////////////////////////
    if (check_retval(&retval, "CVodeSetLinearSolver", 1)) { return (1); }

    /* Set the user-supplied Jacobian routine Jac */
    retval = CVodeSetJacFn(cvode_mem,Jac); ////////////////////////////////////////
    if (check_retval(&retval, "CVodeSetJacFn", 1)) { return (1); }

    /* In loop, call CVode, print results, and test for error.
        Break out of loop when NOUT preset output times have been reached.  */

    sunrealtype t;

    std::vector<SolValues> history_res;
    history_res.reserve(t_list.size());
    for (int i=1; i<t_list.size(); ++i){
        retval = CVode(cvode_mem, t_list[i], y, &t, CV_NORMAL);
        if (check_retval(&retval, "CVode", 1)) { break; }
        if (retval == CV_SUCCESS) history_res.push_back({t,y_data[0],y_data[1]});
    }
    /* Print final statistics to the screen */
    if (rapport_EDO)
    {
        printf("\nFinal Statistics:\n");
        retval = CVodePrintAllStats(cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
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

    //destroy
    N_VDestroy(y);            /* Free y vector */
    N_VDestroy(abstol);       /* Free abstol vector */
    CVodeFree(&cvode_mem);    /* Free CVODE memory */
    SUNLinSolFree(LS);        /* Free the linear solver memory */
    SUNMatDestroy(A);         /* Free the matrix memory */
    SUNContext_Free(&sunctx); /* Free the SUNDIALS context */

    return (retval);


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

int ODE_solver(SUNContext sunctx, N_Vector y, N_Vector abstol, SUNMatrix A , SUNLinearSolver LS, void* cvode_mem , const std::vector<sunrealtype>& t_list, std::vector<sunrealtype>& y_list, Param& fault_param){
    const sunrealtype V1 = fault_param.V0_ * std::exp(fault_param.Dtau_asigma) ; //initial slip rate 
    const sunrealtype theta1 = 1.0/(fault_param.D_c_inv * fault_param.V0_ ); //initial theta
    const sunrealtype T0 = SUN_RCONST(0.0) ;

    /* Create the SUNDIALS context */
    int retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) { return (1); }

    
    /* Initial conditions */
    N_Vector y = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)y, "N_VNew_Serial", 0)) { return (1); }

    sunrealtype* y_data = N_VGetArrayPointer(y);
    y_data[0] = V1 ;   //slip rate component for fixed time t=1
    y_data[1] = theta1 ;  //state variable component for fixed time t=0

    N_Vector abstol = N_VNew_Serial(NEQ, sunctx) ;
    if (check_retval((void*)abstol, "N_VNew_Serial", 0)) { return (1); }

    sunrealtype* y_dot = N_VGetArrayPointer(abstol);
    y_dot[0]=ATOL1 ; //error on V
    y_dot[1]=ATOL2 ; //error on theta

    /* Call CVodeCreate to create the solver memory and specify the
    * Backward Differentiation Formula */
    void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) { return (1); }

        /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in y'=f(t,y), the initial time T0, and
    * the initial dependent variable vector y. */
    retval = CVodeInit(cvode_mem, f, T0, y);
    if (check_retval(&retval, "CVodeInit", 1)) { return (1); }

        /* Call CVodeSVtolerances to specify the scalar relative tolerance
    * and vector abstolute tolerances */
    retval = CVodeSVtolerances(cvode_mem, RTOL, abstol);
    if (check_retval(&retval, "CVodeSVtolerances", 1)) { return (1); }


    // parameters of the fault
    retval = CVodeSetUserData(cvode_mem, (void*)&fault_param);
    if (check_retval(&retval, "CVodeSetUserData", 1)) { return 1; }

    /* Create dense SUNMatrix for use in linear solves */
    SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, sunctx);
    if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (1); }

    /* Create dense SUNLinearSolver object for use by CVode */
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (1); }

    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_retval(&retval, "CVodeSetLinearSolver", 1)) { return (1); }

    /* Set the user-supplied Jacobian routine Jac */
    retval = CVodeSetJacFn(cvode_mem,Jac);
    if (check_retval(&retval, "CVodeSetJacFn", 1)) { return (1); }

    /* In loop, call CVode, print results, and test for error.
        Break out of loop when NOUT preset output times have been reached.  */

    sunrealtype t;

    std::vector<sunrealtype> V_list;
    V_list.reserve(t_list.size());
    for (int i=1; i<t_list.size(); ++i){
        retval = CVode(cvode_mem, t_list[i], y, &t, CV_NORMAL);
        if (check_retval(&retval, "CVode", 1)) { break; }
        if (retval == CV_SUCCESS) V_list.push_back({y_data[0]});
    }
    /* Print final statistics to the screen */
    if (rapport_EDO)
    {
        printf("\nFinal Statistics:\n");
        retval = CVodePrintAllStats(cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
    }



    //destroy
    N_VDestroy(y);            /* Free y vector */
    N_VDestroy(abstol);       /* Free abstol vector */
    CVodeFree(&cvode_mem);    /* Free CVODE memory */
    SUNLinSolFree(LS);        /* Free the linear solver memory */
    SUNMatDestroy(A);         /* Free the matrix memory */
    SUNContext_Free(&sunctx); /* Free the SUNDIALS context */

    return (V_list);

}
