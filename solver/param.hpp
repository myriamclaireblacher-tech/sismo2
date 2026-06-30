#ifndef PARAM_HPP
#define PARAM_HPP

#include "variables.hpp"

//Parameters of the fault
class Param{  
    //param
    public:
        double k, a_sigma, b_sigma, D_c, Dtau;

        //precalculation of parameters
        double V0_;
        double k_a_sigma;           // k/ (a sigma)
        double b_a;                 //b sigma / (a simag)
        double D_c_inv;             // 1/D_c
        double Dtau_asigma;     //Delta Tau/ (a sigma) 
        
        double coeff1;          //2.0 * (- k/ (a sigma) +  b sigma / (a simag) /D_c )
        double coeff2;          //Vinf * k/ (a sigma) 


        //constructor with  6 parameters 
        Param( double k_enter, double a_sigma_enter, double b_sigma_enter, double D_c_enter, double V0_enter, double Dtau_enter)

        : k(k_enter), a_sigma(a_sigma_enter), b_sigma(b_sigma_enter), D_c(D_c_enter) ,V0_(V0_enter),Dtau(Dtau_enter),    
        k_a_sigma(k_enter/a_sigma_enter), b_a(b_sigma_enter/a_sigma_enter), D_c_inv(1.0/D_c_enter), 
        Dtau_asigma(Dtau_enter/a_sigma_enter), coeff1(2.0 * ( - k_enter/a_sigma_enter + b_sigma_enter/ a_sigma_enter /D_c_enter )), coeff2(Vinf * k_enter/a_sigma_enter )
        {}

        //constructor with 5 parameters
        Param( double k_a_sigma_enter, double b_a_enter, double D_c_inv_enter, double Dtau_asigma_enter, double V0_enter) :
        k(-1.0), a_sigma(-1.0), b_sigma(-1.0), D_c(-1.0), V0_(V0_enter), Dtau(-1), //random intiialisation, not useful
        k_a_sigma (k_a_sigma_enter), b_a (b_a_enter) , D_c_inv (D_c_inv_enter), Dtau_asigma(Dtau_asigma_enter) ,
        coeff1( 2.0 * (-k_a_sigma_enter + b_a_enter * D_c_inv_enter) ), coeff2(Vinf * k_a_sigma_enter  )
        {}

        Param() = default ;

        friend std::ostream& operator<<(std::ostream& os, const Param& p) {
            os << "Param {\n"
               
               << "    k/asigma   : " << p.k_a_sigma << "\n"
               << "    b/a         : " << p.b_a << "\n"
               << "    1/D_c     : " << p.D_c_inv << "\n"
               << "    Dtau/asigma : " << p.Dtau_asigma << "\n"
               << "    V0_         : " << p.V0_ << "\n"
               << "}";
            return os;
        }
};

#endif