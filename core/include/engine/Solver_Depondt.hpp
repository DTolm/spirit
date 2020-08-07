template <> inline
void Method_Solver<Solver::Depondt>::Initialize ()
{
    this->iterations = 0;
    this->systems[0]->app.freeLastSolver();
    this->systems[0]->hamiltonian->picoseconds_passed = 0;
    dt_init = 1.76085964411e11 * (1e-16);//in seconds, scaled by gamma
    this->systems[0]->app.init_solver(2);
   /* this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->forces_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->rotationaxis = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->angle = scalarfield( this->nos, 0 );
    this->forces_virtual_norm = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
        configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );

    this->temp1 = vectorfield( this->nos, {0, 0, 0} );*/
};


/*
    Template instantiation of the Simulation class for use with the Depondt Solver.
        The Depondt method is an improvement of Heun's method for spin systems. It applies
        rotations instead of finite displacements and thus avoids re-normalizations.
    Paper: Ph. Depondt et al., Spin dynamics simulations of two-dimensional clusters with
           Heisenberg and dipole-dipole interactions, J. Phys. Condens. Matter 21, 336005 (2009).
*/
template <> inline
void Method_Solver<Solver::Depondt>::Iteration ()
{
    // Generate random vectors for this iteration
   /* this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force(this->configurations, this->forces);
    this->Calculate_Force_Virtual(this->configurations, this->forces, this->forces_virtual);

    // Predictor for each image
    for (int i = 0; i < this->noi; ++i)
    {
        auto& conf           = *this->configurations[i];
        auto& conf_predictor = *this->configurations_predictor[i];

        // For Rotation matrix R := R( H_normed, angle )
        Vectormath::norm( forces_virtual[i], angle );   // angle = |forces_virtual|

        Vectormath::set_c_a( 1, forces_virtual[i], rotationaxis[i] );  // rotationaxis = |forces_virtual|
        Vectormath::normalize_vectors( rotationaxis[i] );            // normalize rotation axis

        // Get spin predictor n' = R(H) * n
        Vectormath::rotate( conf, rotationaxis[i], angle, conf_predictor );
    }

    // Calculate_Force for the Corrector
    this->Calculate_Force(this->configurations_predictor, this->forces_predictor);
    this->Calculate_Force_Virtual(this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor);

    // Corrector step for each image
    for (int i=0; i < this->noi; i++)
    {
        auto& conf   = *this->configurations[i];

        // Calculate the linear combination of the two forces_virtuals
        Vectormath::set_c_a( 0.5, forces_virtual[i], temp1);   // H = H/2
        Vectormath::add_c_a( 0.5, forces_virtual_predictor[i], temp1 ); // H = (H + H')/2

        // Get the rotation angle as norm of temp1 ...For Rotation matrix R' := R( H'_normed, angle' )
        Vectormath::norm( temp1, angle );   // angle' = |forces_virtual lin combination|

        // Normalize temp1 to get rotation axes
        Vectormath::normalize_vectors( temp1 );

        // Get new spin conf n_new = R( (H+H')/2 ) * n
        Vectormath::rotate( conf, temp1, angle, conf );
    }*/
    this->systems[0]->app.runDepondt();
    iterations++;
    if (iterations % this->systems[0]->app.launchConfiguration.savePeriod == 0) {
        int hopf_radii[256];
        int k = 0;
        if (allow_copy == true) {
            allow_copy = false;
            /*for (int i = 1; i < this->systems[0]->geometry->n_cells[0] - 1; i++) {
                int idx = i + this->systems[0]->geometry->n_cells[0] * this->systems[0]->geometry->n_cells[1] / 2 + this->systems[0]->geometry->n_cells[0] * this->systems[0]->geometry->n_cells[1] * this->systems[0]->geometry->n_cells[2] / 2;
                scalar* temp = (scalar*)((*this->configurations[0]).data());
                if ((abs(temp[3 * idx + 2]) < abs(temp[3 * idx - 3 + 2])) && (abs(temp[3 * idx + 2]) < abs(temp[3 * idx + 3 + 2])))
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
        scalar time;
        this->systems[0]->app.getEnergy(energy, &meanMag, &max_Force, &time);
        energy[0] *= 2;
        for (int i = 0; i < 5; i++) {
            energy_full += energy[i];
            this->systems[0]->hamiltonian->energy_array[i].second = energy[i];
        }
        //this->systems[0]->hamiltonian->energy_contributions_per_spin[0].second[0] = energy_full;
        this->systems[0]->M = meanMag / this->systems[0]->geometry->nos;
        //scalar max_Force =this->systems[0]->app.getMaxForce();
        this->force_max_abs_component = sqrt(max_Force);
        std::cout << "time_const_dt(ps): " << iterations * this->systems[0]->app.launchConfiguration.groupedIterations * this->systems[0]->app.launchConfiguration.gamma / 0.176085964411 << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << " Efull: " << energy_full / this->systems[0]->geometry->nos << " Ezeeman: " << energy[0] / this->systems[0]->geometry->nos << " Eanis: " << energy[1] / this->systems[0]->geometry->nos << " Eexch: " << energy[2] / this->systems[0]->geometry->nos << " Edmi: " << energy[3] / this->systems[0]->geometry->nos << " Eddi: " << energy[4] / this->systems[0]->geometry->nos << "\n";

        //std::cout << "Efull: " << energy_full / this->systems[0]->geometry->nos << " Ezeeman: " << energy[0] / this->systems[0]->geometry->nos << " Eanis: " << energy[1] / this->systems[0]->geometry->nos << " Eexch: " << energy[2] / this->systems[0]->geometry->nos << " Edmi: " << energy[3] / this->systems[0]->geometry->nos << " Eddi: " << energy[4] / this->systems[0]->geometry->nos << "\n";
        //std::cout << "time_const_dt(ps): " << iterations* this->systems[0]->app.launchConfiguration.groupedIterations* this->systems[0]->app.launchConfiguration.gamma/ 0.176085964411 << " time GPU(ps): "<< time <<" maxForce: " << max_Force << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << "\n";
        this->systems[0]->hamiltonian->picoseconds_passed = time;
    }
    //this->systems[0]->hamiltonian->picoseconds_passed += dt_init*this->systems[0]->app.launchConfiguration.groupedIterations;
    
};

template <> inline
std::string Method_Solver<Solver::Depondt>::SolverName()
{
    return "Depondt";
};

template <> inline
std::string Method_Solver<Solver::Depondt>::SolverFullName()
{
    return "Depondt";
};