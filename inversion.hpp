#ifndef PARALLEL_TEMPERING
#define PARALLEL_TEMPERING

#include "Surface_response.hpp"
#include <random>
#include <cmath>

struct PT_param {
    double k_a_sigma_inf;           // k/ (a sigma) lower bound
    double k_a_sigma_sup;           //upper bound
    double b_a_inf;                 //b sigma / (a simag)
    double b_a_sup;
    double D_c_inv_inf;             // 1/D_c
    double D_c_inv_sup;
    double Dtau_asigma_inf;     //Delta Tau/ (a sigma)
    double Dtau_asigma_sup;     
    double T_max ;
    int nchains;
    int ncold;
    PT_param(double k_a_sigma_inf_ent, double k_a_sigma_sup_ent, double b_a_inf_ent, double b_a_sup_ent, double D_c_inv_inf_ent, double D_c_inv_sup_ent, double Dtau_asigma_inf_ent, double Dtau_asigma_sup_ent, double Tmax_ent,
            int nchains_ent, int ncold_ent)
    :k_a_sigma_inf(k_a_sigma_inf_ent),k_a_sigma_sup(k_a_sigma_sup_ent),b_a_inf(b_a_inf_ent), b_a_sup (b_a_sup_ent), D_c_inv_inf(b_a_inf_ent),
    D_c_inv_sup (D_c_inv_sup_ent), Dtau_asigma_inf(Dtau_asigma_inf_ent), Dtau_asigma_sup (Dtau_asigma_sup_ent), T_max(T_max_ent),
    nchains (nchains_ent), ncold(ncold_ent)
    {}   
}

int parallel_tempering(const int maxint, const PT_param PT, int seed=42){

    //Design the proposal covariance matrix
    cov = np.identity(nparams) * sigma
    mu  = np.zeros(nparams)

    //Temperatures of the chains
    std::mt19937 gen(seed); //generate random
    std::vector<double> T_log(nchains); 
    std::uniform_real_distribution<double> log_temp(0.0 , std::log(PT.T_max));

    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T_log[i]=0.0 ;
        else T_log[i]= log_temp(gen) ;

    }

    //Initial models of the chains
    std::vector<Param> P;
    P.reserve(PT.nchains);
    for (int i=0;i<PT.nchains;i++){
        double p1 = PT.k_a_sigma_inf + unif_dist(gen) * (PT.k_a_sigma_sup - PT.k_a_sigma_inf );
        double p2 = PT.b_a_inf +     unif_dist(gen) * (PT.b_a_sup - PT.b_a_inf );
        double p3 = PT.D_c_inv_inf + unif_dist(gen) * (PT.D_c_inv_sup - PT.D_c_inv_inf );
        double p4 = PT.Dtau_asigma_inf + unif_dist(gen) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf ) ;
        P.push_back(Param(p1, p2, p3, p4));
    }




    for (int i=0; i<maxint ; i++){
        for (int j=0; j<PT.nchains ; j++){



        }
    }
    return 0;
}

#endif