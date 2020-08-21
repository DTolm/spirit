#pragma once
#ifndef SOLVER_LBFGS_OSO_HPP
#define SOLVER_LBFGS_OSO_HPP

#include <utility/Constants.hpp>
// #include <utility/Exception.hpp>
#include <algorithm>
#include <fstream> 
using namespace Utility;

template <> inline
void Method_Solver<Solver::LBFGS_OSO>::Initialize ()
{
    this->n_lbfgs_memory = 3; // how many previous iterations are stored in the memory
    /*this->delta_a    = std::vector<field<vectorfield>>( this->noi, field<vectorfield>( this->n_lbfgs_memory, vectorfield(this->nos, { 0,0,0 } ) ));
    this->delta_grad = std::vector<field<vectorfield>>( this->noi, field<vectorfield>( this->n_lbfgs_memory, vectorfield(this->nos, { 0,0,0 } ) ));*/
    this->rho        = scalarfield( this->n_lbfgs_memory, 0 );
    this->alpha      = scalarfield( this->n_lbfgs_memory, 0 );
    /*this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->searchdir = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->grad      = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->grad_pr   = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->q_vec     = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );*/
    this->local_iter = 0;
    this->maxmove = Constants::Pi / 200.0;
    this->iterations = 0;
    this->systems[0]->app.freeLastSolver();
    this->systems[0]->app.init_solver(0);
};   

/*
    Implemented according to Aleksei Ivanov's paper: https://arxiv.org/abs/1904.02669
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121).
*/

template <> inline
void Method_Solver<Solver::LBFGS_OSO>::Iteration()
{
    // update forces which are -dE/ds
   // auto time0 = std::chrono::steady_clock::now();
    
    
    //this->Calculate_Force(this->configurations, this->forces);
    //scalar energy=this->systems[0]->app.getEnergy();
    //this->systems[0]->hamiltonian->Update_Energy_Contributions();
    //auto times0 = std::chrono::steady_clock::now();
    //this->systems[0]->app.writeGradient((*this->configurations[0]).data());
   // auto time1 = std::chrono::steady_clock::now();
    //printf("Copy: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time1 - times0).count() * 0.001);

   // printf("Calculate_Force: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() * 0.001);
    // calculate gradients for OSO
    /*for( int img=0; img < this->noi; img++ )
    {
        auto& image = *this->configurations[img];
        auto& grad_ref = this->grad[img];

        auto fv = this->forces_virtual[img].data();
        auto f = this->forces[img].data();
        auto s = image.data();

        //Backend::par::apply( this->nos, [f,fv,s] SPIRIT_LAMBDA (int idx) {
        //    fv[idx] = s[idx].cross(f[idx]);
        //} );

        //Solver_Kernels::oso_calc_gradients(grad_ref, image, this->forces[img]);
    }*/
    //this->systems[0]->app.readDataStream2(this->systems[0]->app.vulkanFFTTransferGrad, this->grad[0].data());
    //this->systems[0]->app.oso_calc_gradients();
    //auto time2 = std::chrono::steady_clock::now();
    //printf("oso_calc_gradients: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() * 0.001);
    //this->systems[0]->app.writeGradient((*this->configurations[0]).data());
   // this->systems[0]->app.lbfgs_get_searchdir(this->local_iter,
     //   this->rho, this->alpha, this->n_lbfgs_memory, maxmove);
   // auto time3 = std::chrono::steady_clock::now();
   // printf("lbfgs_get_searchdir: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count() * 0.001);
    //this->systems[0]->app.writeDataStream(this->systems[0]->app.vulkanFFTTransferSearchDir, searchdir[0].data());
    // calculate search direction
   /* Solver_Kernels::lbfgs_get_searchdir(this->local_iter,
            this->rho, this->alpha, this->q_vec,
            this->searchdir, this->delta_a,
            this->delta_grad, this->grad, this->grad_pr,
            this->n_lbfgs_memory, maxmove);*/

            // Scale direction
           /* this->systems[0]->app.writeDataStream(this->systems[0]->app.vulkanFFTTransferSearchDir, searchdir[0].data());
            scalar scaling = 1;
            for(int img=0; img<noi; img++)
                scaling = std::min(Solver_Kernels::maximum_rotation(searchdir[img], maxmove), scaling);

            for(int img=0; img<noi; img++)
            {
                Vectormath::scale(searchdir[img], scaling);
            }*/
   // this->systems[0]->app.scale(maxmove);
    //auto time4 = std::chrono::steady_clock::now();
    //printf("scale: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count() * 0.001);
    // this->systems[0]->app.writeDataStream(this->systems[0]->app.vulkanFFTTransferSearchDir, searchdir[0].data());
    // this->systems[0]->app.writeDataStream(this->systems[0]->app.vulkanFFTTransferSearchDir, searchdir[0].data());
    // this->systems[0]->app.writeSpins((*this->configurations[0]).data());
     //this->systems[0]->app.readDataStream2(this->systems[0]->app.vulkanFFTTransferSearchDir, searchdir[0].data());
     // rotate spins
    //this->systems[0]->app.oso_rotate();
    /*if (iterations == 0) {
        this->systems[0]->app.setIteration0();
    }*/
    this->systems[0]->app.runLBFGS();
    //auto time5 = std::chrono::steady_clock::now();
    //printf("oso_rotate: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4).count() * 0.001);
   // printf("all: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time5 - time0).count() * 0.001);
    //std::cout << this->systems[0]->hamiltonian->gradient_contributions_per_spin[3*this->systems[0]->geometry->nos*5] << "\n";
    //this->systems[0]->app.writeSpins((*this->systems[0]->spins).data());
    //#ifdef SPIRIT_UI_CXX_USE_QT
    //if (iterations % (this->systems[0]->app.launchConfiguration.savePeriod/ this->systems[0]->app.launchConfiguration.groupedIterations) == 0) {
    
    if (iterations % this->systems[0]->app.launchConfiguration.savePeriod  == 0) {
        int hopf_radii[256];
        int k = 0;
        if (allow_copy == true) {
            allow_copy = false;
           /* for (int i = 1; i < this->systems[0]->geometry->n_cells[0]-1; i++) {
                int idx = i + this->systems[0]->geometry->n_cells[0] * this->systems[0]->geometry->n_cells[1] / 2 + this->systems[0]->geometry->n_cells[0] * this->systems[0]->geometry->n_cells[1] * this->systems[0]->geometry->n_cells[2] / 2;
                scalar * temp=(scalar*)((*this->configurations[0]).data());
                if ((abs(temp[3*idx+2]) < abs(temp[3 * idx -3+ 2])) && (abs(temp[3 * idx + 2]) < abs(temp[3 * idx +3+ 2])))
                {
                    hopf_radii[k] = i;
                    k++;
                }
            }
            std::cout << "hopfion coordinates: ";
            for (int i = 0; i < k; i++)
                std::cout << hopf_radii[i] << " ";
            std::cout << "\n";*/
            this->systems[0]->app.writeSpins((scalar*)((*this->configurations[0]).data()), &allow_copy);
            
        }
        if (allow_copy2 == true) {
            allow_copy2 = false;
            this->systems[0]->app.writeGradient(this->systems[0]->hamiltonian->gradient_contributions_per_spin, &allow_copy2);
        }
        scalar energy[5];
        scalar energy_full = 0;
        Vector3 meanMag;
        scalar max_Force = 0;
        scalar maxmove = 0;
       
        this->systems[0]->app.getEnergy(energy, &meanMag, &max_Force, &maxmove);
        energy[0] *= 2;
        for (int i = 0; i < 5; i++) {
            energy_full += energy[i] ;
            this->systems[0]->hamiltonian->energy_array[i].second = energy[i];
        }
        //this->systems[0]->hamiltonian->energy_contributions_per_spin[0].second[0] = energy_full;
        this->systems[0]->M = meanMag/ (this->systems[0]->geometry->nos);
        //scalar max_Force =this->systems[0]->app.getMaxForce();
        this->force_max_abs_component = sqrt(max_Force);
        //std::cout << "maxTorque: " << this->force_max_abs_component<<" Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << " m_sum: " << this->systems[0]->M[0]+ this->systems[0]->M[1]+this->systems[0]->M[2] <<" Efull: " << energy_full / this->systems[0]->geometry->nos << " Ezeeman: " << energy[0] / this->systems[0]->geometry->nos << " Eanis: " << energy[1] / this->systems[0]->geometry->nos << " Eexch: " << energy[2] / this->systems[0]->geometry->nos << " Edmi: " << energy[3] / this->systems[0]->geometry->nos << " Eddi: " << energy[4] / this->systems[0]->geometry->nos << "\n";
        /*if (iterations == 0) {
            std::ofstream outfile;
         
            outfile.open("output/2506/energy.txt", std::ios_base::app);
            outfile << "init field: "<< " maxTorque: " << this->force_max_abs_component << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << " m_sum: " << this->systems[0]->M[0] + this->systems[0]->M[1] + this->systems[0]->M[2] << " Efull: " << energy_full / this->systems[0]->geometry->nos << " Ezeeman: " << energy[0] / this->systems[0]->geometry->nos << " Eanis: " << energy[1] / this->systems[0]->geometry->nos << " Eexch: " << energy[2] / this->systems[0]->geometry->nos << " Edmi: " << energy[3] / this->systems[0]->geometry->nos << " Eddi: " << energy[4] / this->systems[0]->geometry->nos << "\n";
           
        }
        */
        std::cout << "iteration: " << iterations << " maxTorque: " << this->force_max_abs_component << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << " Efull: " << energy_full / this->systems[0]->geometry->nos << " Ezeeman: " << energy[0] / this->systems[0]->geometry->nos << " Eanis: " << energy[1] / this->systems[0]->geometry->nos << " Eexch: " << energy[2] / this->systems[0]->geometry->nos << " Edmi: " << energy[3] / this->systems[0]->geometry->nos << " Eddi: " << energy[4] / this->systems[0]->geometry->nos << "\n";

        //std::cout <<"Efull: "<< energy_full / this->systems[0]->geometry->nos << " Ezeeman: " << energy[0] / this->systems[0]->geometry->nos << " Eanis: " << energy[1] / this->systems[0]->geometry->nos << " Eexch: " << energy[2] / this->systems[0]->geometry->nos << " Edmi: " << energy[3] / this->systems[0]->geometry->nos << " Eddi: " << energy[4] / this->systems[0]->geometry->nos << "\n";
        //std::cout <<"maxForce: " << sqrt(max_Force) << " maxmove: " << maxmove << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << "\n";

        //this->systems[0]->app.writeGradient((*this->configurations[0]).data());
    }
    //#endif

        iterations++;
   // Solver_Kernels::oso_rotate( this->configurations, this->searchdir);
   // this->systems[0]->app.readInitialSpins((*this->configurations[0]).data());
}


template <> inline
std::string Method_Solver<Solver::LBFGS_OSO>::SolverName()
{
    return "LBFGS_OSO";
}

template <> inline
std::string Method_Solver<Solver::LBFGS_OSO>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno using exponential transforms";
}

#endif