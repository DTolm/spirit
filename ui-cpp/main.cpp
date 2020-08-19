#include "utility/CommandLineParser.hpp"
#include "utility/Handle_Signal.hpp"

#include "Spirit/State.h"
#include <data/State.hpp>

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include "Spirit/Hamiltonian.h"
#include "Spirit/Chain.h"
#include "Spirit/Configurations.h"
#include "Spirit/Transitions.h"
#include "Spirit/Simulation.h"
#include "Spirit/Log.h"
#include "Spirit/IO.h"
#include <engine/Hamiltonian_Micromagnetic.hpp>
#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef SPIRIT_UI_CXX_USE_QT
    #include "MainWindow.hpp"
#endif

#include <memory>
#include <string>

// Initialise global state pointer
std::shared_ptr<State> state;
// Main
int main(int argc, char ** argv)
{
    //--- Register SigInt
    signal(SIGINT, Utility::Handle_Signal::Handle_SigInt);

    //--- Default config file
    std::string cfgfile = "input/input.cfg";

    //--- Command line arguments
    CommandLineParser cmdline(argc, argv);
    // Quiet run
    bool quiet = cmdline.cmdOptionExists("-quiet");
    // Config file
    const std::string & filename = cmdline.getCmdOption("-f");
    if( !filename.empty() )
        cfgfile = filename;

    //--- Data Files
    // std::string spinsfile = "input/anisotropic/achiral.txt";
    // std::string chainfile = "input/chain.txt";
    //std::string ovf_file = "output/2506/1st_sim_2606/hopf_ddi_0.120000.ovf";
    //--- Initialise State
    state = std::shared_ptr<State>(State_Setup(cfgfile.c_str(), quiet), State_Delete);
   // IO_Image_Read(state.get(), ovf_file.c_str());
    //--- Initial spin configuration
    //Configuration_Random(state.get());
    // // Read Image from file
    // Configuration_from_File(state.get(), spinsfile, 0);
    /*for (float i = -0.3; i > -0.36; i -= 0.01) {
        std::string ovf_file = std::string("output/2506/6th_sim_3006/hopf_ddi_")+ std::to_string(2+i)+ std::string(".ovf");
        IO_Image_Read(state.get(), ovf_file.c_str());
        float normal[3] = { 0,0,1 };
        Hamiltonian_Set_Field_Regions(state.get(), i, normal, 0);
        Hamiltonian_Set_Field_Regions(state.get(), i, normal, 1);

       // std::string output_name = std::string("output/2506/hopf_ddi_")+ std::to_string(i+2)+ std::string(".ovf");
       // std::cout << output_name << "\n";
        Simulation_LLG_Start(state.get(), Solver_LBFGS_OSO);
       // IO_Image_Write(state.get(), output_name.c_str(), IO_Fileformat_OVF_text);
        
    }*/

    //--- Chain
    // // Read Chain from file
    // Chain_from_File(state.get(), chainfile);

    // // Set the chain length
    // Chain_Set_Length(state.get(), 12);

    // // First image is plus-z with a Bloch skyrmion at the center
     Configuration_PlusZ(state.get());
     std::shared_ptr<Data::Spin_System> image;
     std::shared_ptr<Data::Spin_System_Chain> chain;

     // Fetch correct indices and pointers
     int idx_image = -1;
     int idx_chain = -1;
     from_indices(state.get(), idx_image, idx_chain, image, chain);
     auto ham = (Engine::Hamiltonian_Micromagnetic*) image->hamiltonian.get();
     for (int k = 0; k < image->geometry->n_cells[2]; k++) {
         for (int j = 0; j < image->geometry->n_cells[1]; j++) {
             for (int i = 0; i < image->geometry->n_cells[0]; i++) {
                 ham->regions[i + j * image->geometry->n_cells[0] + k * image->geometry->n_cells[0] * image->geometry->n_cells[1]] = 1;//will set all spins to 1 - frozen.
             }
         }
     }
     //for (int k = 5; k < image->geometry->n_cells[2]-5; k++) {
         for (int j = 5; j < image->geometry->n_cells[1]-5; j++) {
             for (int i = 5; i < image->geometry->n_cells[0]-5; i++) {
                 ham->regions[i+j* image->geometry->n_cells[0]+0* image->geometry->n_cells[0]* image->geometry->n_cells[1]] = 0;//will set inner spins to 0 - free.
             }
         }
    // }
     ham->init_vulkan(&image->app);
     //float dir[3] = { 1,1,1};
     //Configuration_Domain(state.get(), dir);
     //Configuration_APStripe(state.get());

     //Configuration_Vortex(state.get(), 1000.0, 1.0, 0, false, false, false);
     Configuration_Skyrmion(state.get(), 150.0, 1.0, 0, false, false, false);
     //Configuration_Skyrmion(state.get(), 50.0, 1.0, 0, true, false, false);
     //Configuration_Hopfion(state.get(), 40, 1.0);
     //Configuration_SpinSpiral(State * state);//
    //std::string ovf_file = std::string("output/hopflllong") + std::string(".ovf");
    //IO_Image_Read(state.get(), ovf_file.c_str());
     //Hamiltonian_Set_Field_From_Python(state.get());
    // // Last image is plus-z
    // Chain_Jump_To_Image(state.get(), Chain_Get_NOI(state.get())-1);
    // Configuration_PlusZ(state.get());
    // Chain_Jump_To_Image(state.get(), 0);

    // // Create transition of images between first and last
    // Transition_Homogeneous(state.get(), 0, Chain_Get_NOI(state.get())-1);

    // // Update the Chain's Data'
    // Chain_Update_Data(state.get());
    //-------------------------------------------------------------------------------

    #ifdef _OPENMP
        int nt = omp_get_max_threads() - 1;
        Log_Send(state.get(), Log_Level_Info, Log_Sender_UI, ("Using OpenMP with n=" + std::to_string(nt) + " threads").c_str());
    #endif

    #ifdef SPIRIT_UI_CXX_USE_QT
        //------------------------ User Interface ---------------------------------------
        // Initialise Application and MainWindow
        QApplication app(argc, argv);
        //app.setOrganizationName("--");
        //app.setApplicationName("Spirit - Atomistic Spin Code - OpenGL with Qt");

        // Format for all GL Surfaces
        QSurfaceFormat format;
        format.setSamples(16);
        format.setVersion(3, 3);
        //format.setVersion(4, 2);
        //glFormat.setVersion( 3, 3 );
        //glFormat.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
        //glFormat.setSampleBuffers( true );
        format.setProfile(QSurfaceFormat::CoreProfile);
        format.setDepthBufferSize(24);
        format.setStencilBufferSize(8);
        QSurfaceFormat::setDefaultFormat(format);
        Log_Send(state.get(), Log_Level_Info, Log_Sender_UI, ("QSurfaceFormat version: " + std::to_string(format.majorVersion()) + "." + std::to_string(format.minorVersion())).c_str());

        MainWindow window(state);
        window.setWindowTitle(app.applicationName());
        window.show();
        window.control_set_solver("RK4");
        // Open the Application
        int exec = app.exec();
        // If Application is closed normally
        if (exec != 0) throw exec;
        // Finish
        state.reset();
        return exec;
        //-------------------------------------------------------------------------------
    #else
        //----------------------- LLG Iterations ----------------------------------------
        //std::thread t1 (&Simulation_LLG_Start, state.get(), Solver_VP_OSO, -1, -1, false, -1, -1);
        //t1.join();
        Simulation_LLG_Start(state.get(), Solver_RungeKutta4);

        //-------------------------------------------------------------------------------
    #endif

    state.reset();
    return 0;
}
