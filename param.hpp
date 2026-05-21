#ifndef PARAM_HPP
#define PARAM_HPP


//Parameters of the fault
class Param{  
    //param
    public:
        double k, a_sigma, b_sigma, D_c, V0_, Dtau;

        //precalculation of parameters
        double k_a_sigma;           // k/ (a sigma)
        double b_a;                 //b sigma / (a simag)
        double D_c_inv;             // 1/D_c
        double Dtau_asigma;     //Delta Tau/ (a sigma) 
        double coeff1;               


        //constructor with  6 parameters 
        Param( double k_enter, double a_sigma_enter, double b_sigma_enter, double D_c_enter, double V0_enter, double Dtau_enter)

        : k(k_enter), a_sigma(a_sigma_enter), b_sigma(b_sigma_enter), D_c(D_c_enter) ,V0_(V0_enter),Dtau(Dtau_enter),    
        k_a_sigma(k_enter/a_sigma_enter), b_a(b_sigma_enter/a_sigma_enter), D_c_inv(1.0/D_c_enter), 
        Dtau_asigma(Dtau_enter/a_sigma_enter), coeff1(b_sigma_enter/a_sigma_enter/D_c_enter)
        {}

        //constructor with 4 parameters
        Param( double k_a_sigma_enter, double b_a_enter, double D_c_inv_enter, double Dtau_asigma_enter) :
        k(-1.0), a_sigma(-1.0), b_sigma(-1.0), D_c(-1.0), V0_(-1.0), Dtau(-1.0), //random intiialisation, not useful
        k_a_sigma (k_a_sigma_enter), b_a (b_a_enter) , D_c_inv (D_c_inv_enter), Dtau_asigma(Dtau_asigma_enter) ,
        coeff1(D_c_inv_enter*b_a_enter) 
        {}

};

#endif