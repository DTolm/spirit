template <> inline
void Method_Solver<Solver::CG_OSO>::Initialize ()
{
    this->iterations = 0;
    this->systems[0]->app.freeLastSolver();
    this->systems[0]->app.init_solver(4);

};



template <> inline
void Method_Solver<Solver::CG_OSO>::Iteration ()
{
   
    this->systems[0]->app.CG_iterate();
    
    if (iterations % this->systems[0]->app.launchConfiguration.savePeriod == 0) {

        int hopf_radii[256];
        int k = 0;
        if (allow_copy == true) {
            allow_copy = false;
            this->systems[0]->app.writeSpins((scalar*)((*this->configurations[0]).data()), &allow_copy);

        }
        if (allow_copy2 == true) {
            allow_copy2 = false;
            this->systems[0]->app.writeGradient(this->systems[0]->hamiltonian->gradient_contributions_per_spin, &allow_copy2);
        }
        scalar energy[5];
        scalar energy_full = 0;
        Vector3 meanMag;
        scalar max_Force;
        scalar time;
        this->systems[0]->app.getEnergy(energy, &meanMag, &max_Force, &time);
        energy[0] *= 2;
        for (int i = 0; i < 5; i++) {
            energy_full += energy[i];
            this->systems[0]->hamiltonian->energy_array[i].second = energy[i];
        }
       
        this->systems[0]->M = meanMag / (this->systems[0]->geometry->nos);
        //scalar max_Force =this->systems[0]->app.getMaxForce();
        if (this->force_max_abs_component == sqrt(max_Force))  this->systems[0]->iteration_allowed = false;
        this->force_max_abs_component = sqrt(max_Force);
        if (this->force_max_abs_component < this->systems[0]->app.launchConfiguration.maxTorque) this->systems[0]->iteration_allowed = false;
        std::cout << "iteration: " << iterations << " maxTorque: " << this->force_max_abs_component << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << " Efull: " << energy_full / this->systems[0]->geometry->nos << " Ezeeman: " << energy[0] / this->systems[0]->geometry->nos << " Eanis: " << energy[1] / this->systems[0]->geometry->nos << " Eexch: " << energy[2] / this->systems[0]->geometry->nos << " Edmi: " << energy[3] / this->systems[0]->geometry->nos << " Eddi: " << energy[4] / this->systems[0]->geometry->nos << "\n";

    }

    iterations++;
   
}

template <> inline
std::string Method_Solver<Solver::CG_OSO>::SolverName()
{
	return "CG_OSO";
};

template <> inline
std::string Method_Solver<Solver::CG_OSO>::SolverFullName()
{
	return "CG using exponential transforms";
};