#ifndef PARALLEL_TEMPERING
#define PARALLEL_TEMPERING

#include "Surface_response.hpp"
#include <random>
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <memory>

struct PT_param {
    double k_a_sigma_inf;           // k/ (a sigma) lower bound
    double k_a_sigma_sup;           //upper bound
    double b_a_inf;                 //b sigma / (a simag)
    double b_a_sup;
    double D_c_inv_inf;             // 1/D_c
    double D_c_inv_sup;
    double Dtau_asigma_inf;     //Delta Tau/ (a sigma)
    double Dtau_asigma_sup;     
    double V0_inf; 
    double V0_sup;
    double T_max ;
    int nchains;
    int ncold;
    PT_param(double k_a_sigma_inf_ent, double k_a_sigma_sup_ent, double b_a_inf_ent, double b_a_sup_ent, double D_c_inv_inf_ent, double D_c_inv_sup_ent, double Dtau_asigma_inf_ent, double Dtau_asigma_sup_ent, double Tmax_ent,
            int nchains_ent, int ncold_ent)  ;
};

struct ThreadWorkspace {
    std::vector<Param> pP;
    Fault fault;
    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic> RES_matrix;
    Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> storage_matrix;

    ThreadWorkspace(int t_list_size) ;
    ThreadWorkspace() ; 
};

struct ColdChainSaver {
    std::ofstream file;
    ColdChainSaver(int cold_idx) ;
    void save_step(const std::vector<Param>& pP, double energy) ;
};

double compute_llk(const std::vector<sunrealtype>& t_list, const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data,
                    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, ThreadWorkspace& work) ;


double compute_llk2(const std::vector<sunrealtype>& t_list, const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data,
                    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, ThreadWorkspace& work);

int parallel_tempering(const int maxint, const PT_param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list,  const int seed=42,const double sigma=0.005);

int parallel_tempering_test_out_of_bounds(const int maxint, const PT_param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list,  const int seed=42,const double sigma=0.005);

int parallel_tempering_miror(const int maxint, const PT_param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list,  const int seed=42,const double sigma=0.005);

int parallel_tempering_new(const int maxint, const PT_param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list,  const int seed=42,const double sigma=0.005);
   
#endif