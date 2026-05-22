#include "ODE.hpp"
#include <iostream>
#include <chrono>


int main(){
    Param P_test(0.01, 0.4, 0.17, 0.1, 0.08*100/(365*24), 2.0);
    // double k_enter, double a_sigma_enter, double b_sigma_enter, double D_c_enter, double V0_enter, double Dtau_enter
    std::vector<sunrealtype> t_list(300);
    for (int i=0; i<300; ++i){
        t_list[i]=5.0/300.0*i;
    }
    std::vector<sunrealtype> V_list;
    std::vector<sunrealtype> slip_list;

    std::ofstream fichier_sortie("test.csv");
    
    
    
    Fault F;

    auto timeStart = std::chrono::high_resolution_clock::now();
    for (int j=0 ; j<10000; ++j){
        F.ODE_solver(t_list, V_list,  P_test);
        compute_slip(slip_list, t_list, V_list);
    }
    //solver
    
    //ODE_solver(fichier_sortie, t_list, P_test);
    //ODE_solver(t_list, P_test);}

    
    
    auto timeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = timeEnd - timeStart;
    double timeTotal = duration.count();
    std::cout<<"\nTIME  : "<<timeTotal<<"\n";

    save_to_csv(fichier_sortie, t_list, slip_list,V_list);

    
        
    // 5. Fermeture du fichier
    
    
    std::cout << "Simulation terminee avec succes !" << std::endl;
    return 0;
}