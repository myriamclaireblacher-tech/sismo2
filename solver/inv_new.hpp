#ifndef Easy_PARALLEL_TEMPERING
#define EasyPARALLEL_TEMPERING

#include "inversion.hpp"


struct Easy_Param
{
    double a_sigma_k ;
    double dtau_asigma ;
    double V_V;
    double big_param ;
    Easy_Param(double a_sigma_k_ent, double dtau_asigma_ent, double V_V_ent, double big_pram_ent ) : 
    a_sigma_k(a_sigma_k_ent), dtau_asigma (dtau_asigma_ent), V_V ( V_V_ent), big_param (big_pram_ent){} ;
    Easy_Param() = default ;
} ;

struct Bounds{
    double a_sigma_k_inf; double a_sigma_k_sup;
    double dtau_asigma_inf ; double dtau_asigma_sup;
    double V_V_inf ; double V_V_sup ;
    double big_param_inf; double big_param_sup;
    Bounds(double a_ent, double a_ent_sup, double d_ent, double d_ent_sup, double V_V_ent, double V_V_ent_sup, double big_ent, double big_ent_sup) :
    a_sigma_k_inf(a_ent), a_sigma_k_sup(a_ent_sup), dtau_asigma_inf(d_ent), dtau_asigma_sup(d_ent_sup), V_V_inf(V_V_ent), V_V_sup(V_V_ent_sup), 
    big_param_inf(big_ent), big_param_sup(big_ent_sup) {};
};

void direct(std::vector<sunrealtype> t_list, Easy_Param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix,
    Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

    for (int j=0; j<NSubFaults; j++){
        for (int i=0; i<t_list.size(); i++)
            storage_matrix(j, i) = PT.a_sigma_k * std::log(1.0 + std::exp(PT.dtau_asigma) * PT.V_V * (std::exp(PT.big_param) * t_list[i] + 1.0));
    }
    RES_matrix.noalias() = G * storage_matrix ;
}

double llk_easy (Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, std::vector<sunrealtype> t_list, Easy_Param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix,
    Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

        direct(t_list, PT, G, RES_matrix, storage_matrix) ;
        double llk=  - (data - RES_matrix).array().abs().sum();
        return llk ;
    }

Easy_Param inversion_easy_PT(Bounds bds, int seed){

    #pragma region starting_model

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);

    double p1 = unif_dist_intern (gen) * (bds.a_sigma_k_sup -bds.a_sigma_k_inf) + bds.a_sigma_k_inf;
    double p2 = unif_dist_intern(gen) *(bds.dtau_asigma_sup- bds.dtau_asigma_inf) + bds.a_sigma_k_inf ;
    double p3 = unif_dist_intern(gen) * (bds.V_V_sup-bds.V_V_inf) +bds.V_V_inf ;
    double p4 = unif_dist_intern(gen) * (bds.big_param_sup - bds.big_param_inf ) + bds.big_param_inf ;
    Easy_Param ehdsqhc = Easy_Param (p1,p2,p3,p4); 

    #pragma endregion

}


#endif