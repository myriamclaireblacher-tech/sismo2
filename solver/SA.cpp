#include <algorithm>


void SA (int niter, const std::vector<sunrealtype>& t_list,  const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, c
    onst Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
      int npert  = 20, double T = 0.01){

    double alp    = (0.000004/0.01) ** (1.0/niter) ;
    const double nparam = 5 ;

    //Workspace
    ThreadWorkspace work(t_list.size()) ;

    //modèle initial 

    double p1 = (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 10.0;
    double p2 = (PT.b_a_inf + PT.b_a_sup) / 10.0;
    double p3 = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 20.0;
    double p4 = (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;
    double p5 = (PT.V0_inf + PT.V0_sup) /2.0 ;

    work.pP = Param(p1, p2, p3 , p4 , p5);

    //best llk & params
    double best_llk=compute_llk2(t_list, data, G, work) ;

    std::vector<Param> best_param;
    best_param.resize(NSubFaults);
    best_param = work.pP ;
    int best_it=0;

    //subfault shuffle && random
    int i=1 ;
    double geometric_proba = 1.0 - std::exp(std::log(0.5)/1) ;

    std::mt19937 gen(seed+omp_get_thread_num());
    std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
    /////////////////////////////////
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0);
    std::uniform_int_distribution<int> random_index(0, NSubFaults-1); 
    std::random_device rd;
    std::default_random_engine rng(rd());

    for (int it=0; i<niter ; i++){
        double p = unif_dist(gen);
        if (p<geometric_proba) {i=random_index(gen);}
        int num_param = random_param ;

        std::vector <int> num_param = {0, 1, 2, 3 , 4};
        std::shuffle(num_param.begin(), num_param.end(), rng);

        for (int j=0; j<nparam ; j++){
            double alpha = unif_dist(gen);
            // -> Perturbation distance 
            double y = T * std::tan(std::pi * (alpha - 0.5)) ;
            // direction
            double v = unif_dist_plus(gen) ;
            v= v/std::abs(v);

            for (int ipert=0; i<npert; i++){
                //new model
                m_new[isub, ipm] = m[isub, ipm] + y[ipert] * v[ipert] * (ubounds[ipm] - lbounds[ipm])
                //within bounds

                 while m_new[isub, ipm] > ubounds[ipm] or m_new[isub, ipm] < lbounds[ipm]:

                    # -> New amplitude of the perturbation
                    alpha[ipert] = np.random.uniform(0.0, 1.0)

                    # -> New perturbation distance
                    y[ipert] = T * np.tan(np.pi * (alpha[ipert] - 0.5))

                    # -> New new model
                    m_new[isub, ipm] = m[isub, ipm] + y[ipert] * v[ipert] * (ubounds[ipm] - lbounds[ipm])

                # -> Compute the predictions
                pred = forward.calc_pred(m_new, t, vl, G)

                # -> Compute the cost-function
                err_new[ipert] = forward.calc_cost(obs, pred)
                
            }
        }
        if (num_param==0) ka
        else if (num_param==1) as
        else if (num_param ==2) ba
        else if (num_param == 3) V0
        else if (num_param == 4 )
    }




}