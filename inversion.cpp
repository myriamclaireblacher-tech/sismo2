#include "inversion.hpp"

PT_param::PT_param(double k_a_sigma_inf_ent, double k_a_sigma_sup_ent, double b_a_inf_ent, double b_a_sup_ent, double D_c_inv_inf_ent, double D_c_inv_sup_ent, double Dtau_asigma_inf_ent, double Dtau_asigma_sup_ent, double Tmax_ent,
            int nchains_ent, int ncold_ent)
    :k_a_sigma_inf(k_a_sigma_inf_ent),k_a_sigma_sup(k_a_sigma_sup_ent),b_a_inf(b_a_inf_ent), b_a_sup (b_a_sup_ent), D_c_inv_inf(D_c_inv_inf_ent),
    D_c_inv_sup (D_c_inv_sup_ent), Dtau_asigma_inf(Dtau_asigma_inf_ent), Dtau_asigma_sup (Dtau_asigma_sup_ent), T_max(Tmax_ent),
    nchains (nchains_ent), ncold(ncold_ent)
    {}   


ThreadWorkspace::ThreadWorkspace(int t_list_size) {
    pP.resize(NSubFaults);
    RES_matrix.resize(3 * Nstations, t_list_size);
    storage_matrix.resize(NSubFaults, t_list_size);
}

ThreadWorkspace::ThreadWorkspace() = default; 



ColdChainSaver::ColdChainSaver(int cold_idx) {
    file.open("chain_cold_" + std::to_string(cold_idx) + ".bin", std::ios::binary);
}

void ColdChainSaver::save_step(const std::vector<Param>& pP, double energy) {
    file.write(reinterpret_cast<const char*>(&energy), sizeof(double));
    file.write(reinterpret_cast<const char*>(pP.data()), pP.size() * sizeof(Param));
}

double compute_llk(const std::vector<sunrealtype>& t_list, const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data,
                    Eigen::Matrix<double,3*Nstations,NSubFaults>& G, ThreadWorkspace& work){

    int SR = surface_response(work.pP, t_list, work.fault, G, work.RES_matrix, work.storage_matrix);
    if (SR!=0) return -std::numeric_limits<double>::infinity();
    double llk = - (data - work.RES_matrix).array().square().sum();
    return llk;
}

int parallel_tempering(const int maxint, const PT_param PT, Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, std::vector<sunrealtype>& t_list,  int seed, double sigma){

    //check t_list size, data size
    if (t_list.size()!=(data.size()/3/Nstations)) std::cout<<"\nt_list and data size are not matching \n";

    //Create storage space for each CPU
    std::vector<ThreadWorkspace> workspaces;
    workspaces.reserve(NCPU);
    for (int t = 0; t < NCPU; ++t) {
        workspaces.emplace_back(t_list.size());
    }

    //storage of the models of the cold chains
    std::vector<ColdChainSaver> savers;
    for (int i = 0; i < PT.ncold; ++i) {
        savers.emplace_back(i);
    }
    
    //Temperatures of the chains
    std::mt19937 gen(seed); //generate random
    std::vector<double> T(PT.nchains); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0);

    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T[i]=1.0 ;
        else T[i]= std :: exp( unif_dist(gen) * std::log(PT.T_max) ) ; //loguniform repartition

    }

    //random integer 
    std::uniform_int_distribution<int> rand_chain(0, PT.nchains - 2); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(PT.nchains);

    //Initial models of the chains    
    std::vector<Param> M;
    M.resize(PT.nchains*NSubFaults);
    #pragma omp parallel 
    {
        ThreadWorkspace& work = workspaces[omp_get_thread_num()]; 
        std::mt19937 gen2(seed+omp_get_thread_num());

        #pragma omp for schedule(dynamic)
        for (int j=0;j<PT.nchains;j++){
            bool reject = true;
            double llk_i ;
            //propose new model 
            while (reject)
            {

            for (int i=0; i<NSubFaults ; i++){
                double p1 = PT.k_a_sigma_inf + unif_dist(gen2) * (PT.k_a_sigma_sup - PT.k_a_sigma_inf );
                double p2 = PT.b_a_inf +     unif_dist(gen2) * (PT.b_a_sup - PT.b_a_inf );
                double p3 = PT.D_c_inv_inf + unif_dist(gen2) * (PT.D_c_inv_sup - PT.D_c_inv_inf );
                double p4 = PT.Dtau_asigma_inf + unif_dist(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf ) ;
                M[j * NSubFaults + i]=Param(p1, p2, p3, p4);
            }

            //test llk
            work.pP.assign(M.begin() + (j * NSubFaults), 
                           M.begin() + ((j + 1) * NSubFaults));
            llk_i = compute_llk(t_list, data, G , work);
            if (llk_i != -std::numeric_limits<double>::infinity() ) reject = false ;
            }
            llk[j]=llk_i;
            if (j<PT.ncold) savers[j].save_step(work.pP, llk_i);

        }


    //Parallel tempering


    for (int it = 0; it < maxint ; it ++ ){

        #pragma omp single
        {
        if (it % (maxint / 10) == 0) {
            std::cout << std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
        }}

        #pragma omp for schedule(dynamic)
        for (int ichain = 0; ichain < PT.nchains ; ichain++){

            //copy model values
            work.pP.assign(M.begin() + (ichain * NSubFaults), M.begin() + ((ichain + 1) * NSubFaults));

            //New model proposal
            for (int i=0; i<NSubFaults; i++){
                work.pP[i].k_a_sigma   += unif_dist(gen2) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigma;
                work.pP[i].b_a         += unif_dist(gen2) * (PT.b_a_sup         - PT.b_a_inf) * sigma ;
                work.pP[i].D_c_inv     += unif_dist(gen2) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigma ;
                work.pP[i].Dtau_asigma += unif_dist(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigma;
            }

            // Check if out of bounds
            bool InBound = true;
            for (int i=0; i<NSubFaults; i++){
                if (work.pP[i].k_a_sigma   < PT.k_a_sigma_inf   || work.pP[i].k_a_sigma   > PT.k_a_sigma_sup   ||
                    work.pP[i].b_a         < PT.b_a_inf         || work.pP[i].b_a         > PT.b_a_sup         ||
                    work.pP[i].D_c_inv     < PT.D_c_inv_inf     || work.pP[i].D_c_inv     > PT.D_c_inv_sup     ||
                    work.pP[i].Dtau_asigma < PT.Dtau_asigma_inf || work.pP[i].Dtau_asigma > PT.Dtau_asigma_sup) 
                {
                    InBound = false;
                    break; 
                }
                }

            //accept rate of the new model
            bool accept = false ;
            double Enew = -std::numeric_limits<double>::infinity() ; 

            if (InBound) {
                Enew = compute_llk(t_list, data, G , work);
                if (Enew!=-std::numeric_limits<double>::infinity()){
                    double delta  = (Enew - llk[ichain])/T[ichain] ;
                    double alpha  = std::min(0.0, delta);
                    double u      = std::log(unif_dist(gen2) );
                    accept = (u <= alpha);
                }
            }

            if (accept){
                std::copy(work.pP.begin(), work.pP.end(), M.begin() + (ichain * NSubFaults));
                llk[ichain] = Enew;
                if (ichain<PT.ncold) savers[ichain].save_step(work.pP, Enew);            //plus tard save que sur certaines itérations
            }

        }} //parallel end

        #pragma omp single
        { std::cout<<"\nswap ";

        

        for (int s = 0; s < PT.nchains - 1; s++) {
            int p = rand_chain(gen);
            int q = rand_chain(gen);
            if (p == q) continue;

            // Formule théorique du Parallel Tempering
            double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
            double u_swap     = std::log(unif_dist(gen));

            if (u_swap <= alpha_swap) {
                std::swap_ranges(M.begin() + (p * NSubFaults), M.begin() + ((p + 1) * NSubFaults), M.begin() + (q * NSubFaults));
                std::swap(llk[p], llk[q]);
            }
        }
        
    }}
    std::cout<<"\n 100% done";
    return 0;
}

