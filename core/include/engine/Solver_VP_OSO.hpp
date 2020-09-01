template <> inline
void Method_Solver<Solver::VP_OSO>::Initialize ()
{
    this->iterations = 0;
    this->systems[0]->app.freeLastSolver();
    this->systems[0]->app.init_solver(1);
    /*this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->configurations_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      configurations_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->velocities          = std::vector<vectorfield>(this->noi, vectorfield(this->nos, Vector3::Zero()));	// [noi][nos]
    this->velocities_previous = velocities;	// [noi][nos]
    this->forces_previous     = velocities;	// [noi][nos]
    this->grad                = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->grad_pr             = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->projection          = std::vector<scalar>(this->noi, 0);	// [noi]
    this->force_norm2         = std::vector<scalar>(this->noi, 0);	// [noi]
    this->searchdir           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );*/
};


/*
    Template instantiation of the Simulation class for use with the VP Solver.
		The velocity projection method is often efficient for direct minimization,
		but deals poorly with quickly varying fields or stochastic noise.
	Paper: P. F. Bessarab et al., Method for finding mechanism and activation energy
		   of magnetic transitions, applied to skyrmion and antivortex annihilation,
		   Comp. Phys. Comm. 196, 335 (2015).

    Instead of the cartesian update scheme with re-normalization, this implementation uses the orthogonal spin optimization scheme,
    described by A. Ivanov in https://arxiv.org/abs/1904.02669.
*/

template <> inline
void Method_Solver<Solver::VP_OSO>::Iteration ()
{
    /*scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for (int img = 0; img < noi; ++img)
    {
        auto g    = grad[img].data();
        auto g_pr = grad_pr[img].data();
        auto v    = velocities[img].data();
        auto v_pr = velocities_previous[img].data();

        Backend::par::apply( nos, [g, g_pr, v, v_pr] SPIRIT_LAMBDA (int idx) {
            g_pr[idx] = g[idx]; //don't ferget prvious step
            //v_pr[idx] = v[idx];
        } );
    }*/

    // Get the forces on the configurations
    
    //this->Calculate_Force_Virtual(configurations, forces, forces_virtual);

    /*for( int img=0; img < this->noi; img++ )
    {
        auto& image = *this->configurations[img];
        auto& grad = this->grad[img];
       // Solver_Kernels::oso_calc_gradients(grad, image, this->forces[img]);
        Vectormath::scale(grad, -1.0); // we won't do that
    }*/
   // auto time0 = std::chrono::steady_clock::now();
    //this->Calculate_Force(configurations, forces);
    //this->systems[0]->app.run();
    //this->systems[0]->app.writeGradient((*this->configurations[0]).data());
    //scalar energy = this->systems[0]->app.getEnergy();
   // auto time1 = std::chrono::steady_clock::now();
    //this->systems[0]->app.oso_calc_gradients();
   // auto time2 = std::chrono::steady_clock::now();
    this->systems[0]->app.vp_get_searchdir(1.0 / this->m, this->systems[0]->llg_parameters->dt, iterations);
    //auto time3 = std::chrono::steady_clock::now();
    //if (iterations > 0) {
        //this->systems[0]->app.oso_rotate();

    //}
   // auto time4 = std::chrono::steady_clock::now();
   /* printf("Calculate_Force: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() * 0.001);
    printf("oso_calc_gradients: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() * 0.001);
    printf("vp_get_searchdir: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count() * 0.001);
    printf("oso_rotate: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count() * 0.001);
    printf("all: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time4 - time0).count() * 0.001);*/
    //#ifdef SPIRIT_UI_CXX_USE_QT
    //if (iterations % (this->systems[0]->app.launchConfiguration.savePeriod / this->systems[0]->app.launchConfiguration.groupedIterations) == 0) {
    //std::cout << this->systems[0]->spins_vector[0].x << "\n";
    if (iterations % this->systems[0]->app.launchConfiguration.savePeriod == 0) {

        int hopf_radii[256];
        int k = 0;
        if (allow_copy == true) {
            allow_copy = false;
            //this->systems[0]->app.writeSpins((*this->configurations[0]).data(), &allow_copy);
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
        scalar max_Force;
        scalar time;
        this->systems[0]->app.getEnergy(energy, &meanMag, &max_Force, &time);
        energy[0] *= 2;
        for (int i = 0; i < 5; i++) {
            this->systems[0]->hamiltonian->energy_array[i].second = energy[i];
            energy[i] /= this->systems[0]->app.launchConfiguration.num_nonzero_Ms;
            energy_full += energy[i];
        }
        //this->systems[0]->hamiltonian->energy_contributions_per_spin[0].second[0] = energy_full;
        //this->systems[0]->M = meanMag / num_nonzero_Ms;
       
        this->systems[0]->M = meanMag / this->systems[0]->app.launchConfiguration.num_nonzero_Ms;
        //scalar max_Force =this->systems[0]->app.getMaxForce();
        //if (this->force_max_abs_component == sqrt(max_Force))  this->systems[0]->iteration_allowed = false;
        this->force_max_abs_component = sqrt(max_Force);
        if (this->force_max_abs_component < this->systems[0]->app.launchConfiguration.maxTorque) this->systems[0]->iteration_allowed = false;
       /* if (iterations == 0) {
            std::ofstream outfile;

            outfile.open("output/3008/1st_sim/energy.txt", std::ios_base::app);
            outfile << "init iteration: " << iterations << " maxTorque: " << this->force_max_abs_component << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << " Efull: " << energy_full / num_nonzero_Ms << " Ezeeman: " << energy[0] / num_nonzero_Ms << " Eanis: " << energy[1] / num_nonzero_Ms << " Eexch: " << energy[2] / num_nonzero_Ms << " Edmi: " << energy[3] / num_nonzero_Ms << " Eddi: " << energy[4] / num_nonzero_Ms << " Hopf radii: ";
            for (int i = 0; i < k; i++)
                outfile << hopf_radii[i] << " ";
            outfile << "\n";
        }
        if (this->force_max_abs_component < this->systems[0]->app.launchConfiguration.maxTorque) {
            std::ofstream outfile;

            outfile.open("output/3008/1st_sim/energy.txt", std::ios_base::app);
            outfile << "final iteration: " << iterations <<" maxTorque: " << this->force_max_abs_component << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << " Efull: " << energy_full / num_nonzero_Ms << " Ezeeman: " << energy[0] / num_nonzero_Ms << " Eanis: " << energy[1] / num_nonzero_Ms << " Eexch: " << energy[2] / num_nonzero_Ms << " Edmi: " << energy[3] / num_nonzero_Ms << " Eddi: " << energy[4] / num_nonzero_Ms << " Hopf radii: ";
            for (int i = 0; i < k; i++)
                outfile << hopf_radii[i] << " ";
            outfile << "\n";
        }*/
        std::cout << "iteration: " << iterations << " maxTorque: " << this->force_max_abs_component << " Mx: " << this->systems[0]->M[0] << " My: " << this->systems[0]->M[1] << " Mz: " << this->systems[0]->M[2] << " Efull: " << energy_full << " Ezeeman: " << energy[0] << " Eanis: " << energy[1] << " Eexch: " << energy[2] << " Edmi: " << energy[3] << " Eddi: " << energy[4]  << "\n";

        //this->systems[0]->app.writeGradient((*this->configurations[0]).data());
    }
    //#endif   
    iterations++;
    /*for (int img = 0; img < noi; ++img )
    {
        auto& velocity = velocities[img];
        auto g        = this->grad[img].data();
        auto g_pr     = this->grad_pr[img].data();
        auto v        = velocities[img].data();
        auto m_temp   = this->m;

        // Calculate the new velocity
        Backend::par::apply(nos, [g,g_pr,v,m_temp] SPIRIT_LAMBDA (int idx) {
            v[idx] += 0.5/m_temp * (g_pr[idx] + g[idx]); // no need to change sign - depends on grad Apply0VP
        });

        // Get the projection of the velocity on the force
        projection[img]  = Vectormath::dot(velocity, this->grad[img]); //correct dot0
        force_norm2[img] = Vectormath::dot(this->grad[img], this->grad[img]); //correct dot1
    }
    for (int img = 0; img < noi; ++img)
    {
        projection_full  += projection[img];
        force_norm2_full += force_norm2[img];
    }
    for (int img = 0; img < noi; ++img)
    {
        auto sd     = this->searchdir[img].data();
        auto v      = this->velocities[img].data();
        auto g      = this->grad[img].data();
        auto m_temp = this->m;

        scalar dt = this->systems[img]->llg_parameters->dt;
        scalar ratio = projection_full/force_norm2_full;

        // Calculate the projected velocity
        if (projection_full <= 0)
        {
            Vectormath::fill(velocities[img], { 0,0,0 });
        } else {
            Backend::par::apply(nos, [g,v,ratio] SPIRIT_LAMBDA (int idx) {
                v[idx] = g[idx] * ratio; //merge with apply1
            });
        }

        Backend::par::apply( nos, [sd, dt, m_temp, v, g] SPIRIT_LAMBDA (int idx) {
            sd[idx] = dt * v[idx] + 0.5/m_temp * dt * g[idx]; // add minus - the only part that matters Apply1VP, conditional from previous
        }); 
    }*/
    //Solver_Kernels::oso_rotate( this->configurations, this->searchdir);
   
}

template <> inline
std::string Method_Solver<Solver::VP_OSO>::SolverName()
{
	return "VP_OSO";
};

template <> inline
std::string Method_Solver<Solver::VP_OSO>::SolverFullName()
{
	return "Velocity Projection using exponential transforms";
};