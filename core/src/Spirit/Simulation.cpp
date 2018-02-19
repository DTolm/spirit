#include <Spirit/State.h>
#include <Spirit/Simulation.h>
#include <Spirit/Chain.h>

#include <data/State.hpp>
#include <engine/Method_LLG.hpp>
#include <engine/Method_MC.hpp>
#include <engine/Method_GNEB.hpp>
#include <engine/Method_EMA.hpp>
#include <engine/Method_MMF.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>


bool Get_Method( State *state, const char * c_method_type, const char * c_solver_type, 
                 int n_iterations, int n_iterations_log, int idx_image, int idx_chain, 
                 std::shared_ptr<Engine::Method> & method ) noexcept
{
    try
    {
        // Translate to string
        std::string method_type(c_method_type);
        std::string solver_type(c_solver_type);

        // Determine the Solver kind
        Engine::Solver solver;
        if (solver_type == "SIB")
            solver = Engine::Solver::SIB;
        else if (solver_type == "Heun")
            solver = Engine::Solver::Heun;
        else if (solver_type == "Depondt")
            solver = Engine::Solver::Depondt;
        else if (solver_type == "NCG")
            solver = Engine::Solver::NCG;
        else if (solver_type == "VP")
            solver = Engine::Solver::VP;
        else
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API, "Invalid Solver selected: " + 
                    solver_type);
            return false;
        }

        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Determine wether to stop or start a simulation
        if (image->iteration_allowed)
        {
            // Currently iterating image, so we stop
            image->Lock();
            image->iteration_allowed = false;
            image->Unlock();
            return false;
        }
        else if (chain->iteration_allowed)
        {
            // Currently iterating chain, so we stop
            chain->Lock();
            chain->iteration_allowed = false;
            chain->Unlock();
            return false;
        }
        else
        {
            // ------ Nothing is iterating, so we could start a simulation ------

            // Lock the chain in order to prevent unexpected things
            chain->Lock();

            // Determine the method and chain(s) or image(s) involved
            if (method_type == "LLG")
            {
                image->iteration_allowed = true;
                if (n_iterations > 0) image->llg_parameters->n_iterations = n_iterations;
                if (n_iterations_log > 0) image->llg_parameters->n_iterations_log = n_iterations_log;

                if (solver == Engine::Solver::SIB)
                    method = std::shared_ptr<Engine::Method>( 
                        new Engine::Method_LLG<Engine::Solver::SIB>( image, idx_image, idx_chain ) );
                else if (solver == Engine::Solver::Heun)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_LLG<Engine::Solver::Heun>( image, idx_image, idx_chain ) );
                else if (solver == Engine::Solver::Depondt)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_LLG<Engine::Solver::Depondt>( image, idx_image, idx_chain ) );
                else if (solver == Engine::Solver::NCG)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_LLG<Engine::Solver::NCG>( image, idx_image, idx_chain ) );
                else if (solver == Engine::Solver::VP)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_LLG<Engine::Solver::VP>( image, idx_image, idx_chain ) );
            }
            else if (method_type == "MC")
            {
                image->iteration_allowed = true;
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_MC( image, idx_image, idx_chain ) );
            }
            else if (method_type == "EMA")
            {
                image->iteration_allowed = true;
                if (n_iterations > 0) image->ema_parameters->n_iterations = n_iterations;
                if (n_iterations_log > 0) image->ema_parameters->n_iterations_log = n_iterations_log;
                
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_EMA(image,idx_image,idx_chain));
            }
            else if (method_type == "GNEB")
            {
                if (Simulation_Running_Anywhere_Chain(state, idx_chain))
                {
                    Log( Utility::Log_Level::Error, Utility::Log_Sender::API, 
                            std::string( "There are still one or more simulations running on the specified chain!" ) +
                            std::string( " Please stop them before starting a GNEB calculation." ) );
                    chain->Unlock();
                    return false;
                }
                else if (Chain_Get_NOI(state, idx_chain) < 3)
                {
                    Log( Utility::Log_Level::Error, Utility::Log_Sender::API, 
                            std::string( "There are less than 3 images in the specified chain!" ) +
                            std::string( " Please insert more before starting a GNEB calculation." ) );
                    chain->Unlock();
                    return false;
                }
                else
                {
                    chain->iteration_allowed = true;
                    if (n_iterations > 0) 
                        chain->gneb_parameters->n_iterations = n_iterations;
                    if (n_iterations_log > 0) 
                        chain->gneb_parameters->n_iterations_log = n_iterations_log;

                    if (solver == Engine::Solver::SIB)
                        method = std::shared_ptr<Engine::Method>(
                            new Engine::Method_GNEB<Engine::Solver::SIB>( chain, idx_chain ) );
                    else if (solver == Engine::Solver::Heun)
                        method = std::shared_ptr<Engine::Method>(
                            new Engine::Method_GNEB<Engine::Solver::Heun>( chain, idx_chain ) );
                    else if (solver == Engine::Solver::Depondt)
                        method = std::shared_ptr<Engine::Method>(
                            new Engine::Method_GNEB<Engine::Solver::Depondt>( chain, idx_chain ) );
                    else if (solver == Engine::Solver::NCG)
                        method = std::shared_ptr<Engine::Method>(
                            new Engine::Method_GNEB<Engine::Solver::NCG>( chain, idx_chain ) );
                    else if (solver == Engine::Solver::VP)
                        method = std::shared_ptr<Engine::Method>(
                            new Engine::Method_GNEB<Engine::Solver::VP>( chain, idx_chain ) );
                }
            }
            else if (method_type == "MMF")
            {
                image->iteration_allowed = true;
                if (n_iterations > 0) 
                    image->mmf_parameters->n_iterations = n_iterations;
                if (n_iterations_log > 0) 
                    image->mmf_parameters->n_iterations_log = n_iterations_log;

                if (solver == Engine::Solver::SIB)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_MMF<Engine::Solver::SIB>( image, idx_chain ) );
                else if (solver == Engine::Solver::Heun)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_MMF<Engine::Solver::Heun>( image, idx_chain ) );
                else if (solver == Engine::Solver::Depondt)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_MMF<Engine::Solver::Depondt>( image, idx_chain ) );
                else if (solver == Engine::Solver::NCG)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_MMF<Engine::Solver::NCG>( image, idx_chain ) );
                else if (solver == Engine::Solver::VP)
                    method = std::shared_ptr<Engine::Method>(
                        new Engine::Method_MMF<Engine::Solver::VP>( image, idx_chain ) );
            }
            else
            {
                Log( Utility::Log_Level::Error, Utility::Log_Sender::API, 
                        "Invalid Method selected: " + method_type );
                chain->Unlock();
                return false;
            }
        }

        // Create Simulation Information
        auto info = std::shared_ptr<Engine::Method>(method);

        // Add to correct list
        if (method_type == "LLG")
            state->method_image[idx_image] = info;
        else if (method_type == "MC")
        {
            state->method_image[idx_image] = info;
        }
        else if (method_type == "EMA")
            state->method_image[idx_image] = info;
        else if (method_type == "GNEB")
            state->method_chain = info;
        else if (method_type == "MMF")
            state->method_image[idx_image] = info;
        
        // Unlock chain in order to be able to iterate
        chain->Unlock();

        return true;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return false;
    } 
}


void Simulation_SingleShot(State *state, const char * c_method_type, const char * c_solver_type, 
    int n_iterations, int n_iterations_log, int idx_image, int idx_chain) noexcept
{
    try
    {
        // One Iteration
        std::shared_ptr<Engine::Method> method;
        if ( Get_Method( state, c_method_type, c_solver_type, 1, n_iterations_log, 
                         idx_image, idx_chain, method) )
            method->Iterate();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);        
    }
}

void Simulation_PlayPause(State *state, const char * c_method_type, const char * c_solver_type,
    int n_iterations, int n_iterations_log, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Iterate
        std::shared_ptr<Engine::Method> method;
        if ( Get_Method( state, c_method_type, c_solver_type, n_iterations, n_iterations_log, 
                            idx_image, idx_chain, method ) )
            method->Iterate();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);        
    }
}

void Simulation_Stop_All(State *state) noexcept
{
    try
    {
        // GNEB
        state->active_chain->Lock();
        state->active_chain->iteration_allowed = false;
        state->active_chain->Unlock();
        for (int i=0; i<state->noc; ++i)
        {
            state->active_chain->Lock();
            state->active_chain->iteration_allowed = false;
            state->active_chain->Unlock();
        }

        // LLG, MC, EMA, MMF
        state->active_image->Lock();
        state->active_image->iteration_allowed = false;
        state->active_image->Unlock();
        for (int img = 0; img < state->active_chain->noi; ++img)
        {
            state->active_chain->images[img]->Lock();
            state->active_chain->images[img]->iteration_allowed = false;
            state->active_chain->images[img]->Unlock();
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(-1, -1);
    }
}


float Simulation_Get_MaxTorqueComponent(State * state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (Simulation_Running_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return (float) state->method_image[idx_image]->getForceMaxAbsComponent();
        }
        else if (Simulation_Running_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return (float) state->method_chain->getForceMaxAbsComponent();
        }

        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}


void Simulation_Get_Chain_MaxTorqueComponents(State * state, float * torques, int idx_chain) noexcept
{
    int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (Simulation_Running_Chain(state, idx_chain))
        {
            std::vector<scalar> t(chain->noi, 0);
            
            if (state->method_chain)
                t = state->method_chain->getForceMaxAbsComponent_All();

            for (int i=0; i<chain->noi; ++i)
            {
                torques[i] = t[i];
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}


float Simulation_Get_IterationsPerSecond(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return (float)state->method_image[idx_image]->getIterationsPerSecond();
        }
        else if (Simulation_Running_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return (float)state->method_chain->getIterationsPerSecond();
        }

        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}


int Simulation_Get_Iteration(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return (float)state->method_image[idx_image]->getNIterations();
        }
        else if (Simulation_Running_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return (float)state->method_chain->getNIterations();
        }

        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}


const char * Simulation_Get_Solver_Name(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        
        if (Simulation_Running_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return state->method_image[idx_image]->SolverName().c_str();
        }
        else if (Simulation_Running_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return state->method_chain->SolverName().c_str();
        }
        
        return "";
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}

const char * Simulation_Get_Method_Name(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return state->method_image[idx_image]->Name().c_str();
        }
        else if (Simulation_Running_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return state->method_chain->Name().c_str();
        }

        return "";
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}



bool Simulation_Running_Image(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (image->iteration_allowed) return true;
        else return false;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return false;
    }
}

bool Simulation_Running_Chain(State *state, int idx_chain) noexcept
{
    int idx_image=-1;

    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
            from_indices( state, idx_image, idx_chain, image, chain );
        
        if (state->active_chain->iteration_allowed)
            return true;
        else 
            return false;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return false;
    }
}

bool Simulation_Running_Anywhere_Chain(State *state, int idx_chain) noexcept
{
    int idx_image=-1;

    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_Chain(state, idx_chain)) 
            return true;
        
        for (int i=0; i<chain->noi; ++i)
            if (Simulation_Running_Image(state, i, idx_chain)) 
                return true;
        
        return false;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return false;
    }
}

void Simulation_Calculate_Eigenmodes(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if( Simulation_Running_Image(state, idx_image, idx_chain) )
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, 
                "Eigenmodes cannot be calculated while a simulation is running on the image" );
                return;
        }
        else
        {
            image->Lock();
            Engine::Calculate_Eigenmodes(image, idx_image, idx_chain );
            image->Unlock();
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

