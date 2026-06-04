#ifndef PARALLEL_TEMPERING
#define PARALLEL_TEMPERING

#include "Surface_response.hpp"
#include <random>
#include <cmath>
#include <string>
#include <fstream>

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
};

struct ThreadWorkspace {
    std::vector<Param> pP;
    std::vector<Fault> Faults;
    Eigen::MatrixXd RES_matrix;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> storage_matrix;

    ThreadWorkspace(int t_list_size) {
        pP.reserve(NSubFaults);
        Faults.resize(NSubFaults);
        RES_matrix.resize(3 * Nstations, t_list_size);
        storage_matrix.resize(NSubFaults, t_list_size);
    }
    ThreadWorkspace() = default; 
};

struct ColdChainSaver {
    std::ofstream model_file;
    std::ofstream llk_file;

    // One file for each cold chain
    ColdChainSaver(int cold_idx) {
        model_file.open("models_cold_" + std::to_string(cold_idx) + ".txt");
        llk_file.open("llk_cold_" + std::to_string(cold_idx) + ".txt");   
    }

    void save_step(const Param& m, double energy) {
        //save model
        model_file << m.k_a_sigma << " " << m.b_a << " " << m.D_c_inv << " " << m.Dtau_asigma << "\n";
        //save log-likelyhood
        llk_file << energy << "\n";
    }
};

double compute_llk(const std::vector<sunrealtype>& t_list, const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data,
                    Eigen::Matrix<double,3*Nstations,NSubFaults>& G, ThreadWorkspace& work,double T){

    int SR = surface_response(work.pP, t_list, work.Faults, G, work.RES_matrix, work.storage_matrix);
    if (SR!=0) return -std::numeric_limits<double>::infinity();
    double llk = - (data - work.RES_matrix).array().square().sum()/T;
    return llk;
};




int parallel_tempering(const int maxint, const PT_param PT, Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, std::vector<double> t_list,  int seed=42, double sigma=0.05){

    //check t_list size, data size

    if (t_list.size()!=data.size()) cout<<"\nt_list and data are not matching \n";

    //Create storage space for each CPU
    std::vector<ThreadWorkspace> workspaces;
    workspaces.reserve(NCPU);
    for (int t = 0; t < NCPU; ++t) {
        workspaces.emplace_back(NSubFaults, Nstations, t_list.size());
    }
    //storage of the models of the cold chains
    std::vector<ColdChainSaver> savers;
    for (int i = 0; i < PT.ncold; ++i) {
        savers.emplace_back(i);
    }
    
    //Temperatures of the chains
    std::mt19937 gen(seed); //generate random
    std::vector<double> T(nchains); 
    std::uniform_real_distribution<double> unif_dist(0.0 , std::log(PT.T_max));

    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T[i]=1.0 ;
        else T[i]= std :: exp( unif_dist(gen) * std::log(PT.T_max) ) ; //loguniform repartition

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

    //save initial models in files for cold chains



    //Parallel tempering
    for (int i=0; i<maxint ; i++){

        if (it % (itmax / 10) == 0) {
            std::cout << std::setw(3) << (100 * it / itmax) << "% done" << std::endl;
        }

        //add CPU
        for (int ichain=0; ichain<PT.nchains ; ichain++){
            Param Mnew = P[ichain];
            Mnew.k_a_sigma   += unif_dist(gen) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigma;
            Mnew.b_a         += unif_dist(gen) * (PT.b_a_sup         - PT.b_a_inf) * sigma ;
            Mnew.D_c_inv     += unif_dist(gen) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigma ;
            Mnew.Dtau_asigma += unif_dist(gen) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigma;

            bool accept = false ;

            if (Mnew.k_a_sigma   < PT.k_a_sigma_inf   || Mnew.k_a_sigma   > PT.k_a_sigma_sup   ||
            Mnew.b_a         < PT.b_a_inf         || Mnew.b_a         > PT.b_a_sup         ||
            Mnew.D_c_inv     < PT.D_c_inv_inf     || Mnew.D_c_inv     > PT.D_c_inv_sup     ||
            Mnew.Dtau_asigma < PT.Dtau_asigma_inf || Mnew.Dtau_asigma > PT.Dtau_asigma_sup) 
            {
                int ret = compute_llk(const Param *P)
                //résoudre + llk
                if (!std::isnan(Enew))


            }

            Mnew = Param(Mnew.k_a_sigma, Mnew.b_a, Mnew.D_c_inv, Mnew.Dtau_asigma);


        }

        
    }



    
    return 0;
}

#endif