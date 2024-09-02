// SIMULIA includes

#include <aba_for_c.h>
#include <SMAAspUserSubroutines.h>
#include <ExtraUtilities.h>

// User includes
#include <filesystem>
#include <iostream>
#include <torch/script.h> // One-stop header.

#define MUTEX_LINE(mutex, line) \
    MutexLock(mutex);           \
    line                        \
    MutexUnlock(mutex);


extern "C" void FOR_NAME(uel,UEL) (
    Real*       _rhs,           //  rhs(mlvarx,ndofel) 
    Real*       _amatrx,
    Real*       _svars,
    Real*       _energy,
    const int&  ndofel,
    const int&  nrhs,
    const int&  nsvars,
    const Real* _props,
    const int&  nprops,
    const Real* _coords,
    const int&  mcrd,
    const int&  nnode,
    const Real* _u,
    const Real* _du,
    const Real* _v,
    const Real* _a,
    const int&  jtype,
    const Real(&time)[2],
    const Real& dtime,
    const int&  kstep,
    const int&  kinc,
    const int&  jelem,
    const Real* _params,
    const int&  ndload,
    const int*  _jdltyp,
    const Real* _adlmag,
    const Real* _predef,
    const int&  npredf,
    const int*  _lflags,
    const int&  mlvarx,
    const Real* _ddlmag,
    const int&  mdload,
    Real&       pnewdt,
    const int*  _jprops,
    const int&  njprop,
    const Real& period
) {
    SIMULIA::FortranArray2d<Real> amatrx(_amatrx, ndofel, ndofel);
    SIMULIA::FortranArray2d<Real> rhs(_rhs, mlvarx, ndofel);
    SIMULIA::FortranArray2d<const Real> du(_du, mlvarx, ndofel);
    SIMULIA::FortranArray1d<const Real> u(_u, ndofel);
    SIMULIA::FortranArray1d<const Real> props(_props, nprops);
    SIMULIA::FortranArray1d<const int> lflags(_lflags, 10);
    SIMULIA::FortranArray2d<const Real> coords(_coords, mcrd, nnode);

    const Real half = 0.5;
    auto area = props(1);
    auto e    = props(2);
    auto anu  = props(3);
    auto rho  = props(4);
    auto alen = abs(coords(1,2)-coords(1,1));
    auto ak   = area*e/alen;
    auto am   = half*area*rho*alen;

    for (size_t i = 1; i <= ndofel; i++)
    {
        for (size_t j = 1; j <= ndofel; j++)
        {
            amatrx(i, j) = 0.0;
        }
        rhs(i, 1) = 0.0;
    }

    for (size_t i = 1; i <= ndofel; i++)
    {
        for (size_t j = 1; j <= ndofel; j++)
        {
            if (i!=j)
            {
                amatrx(i, i) += 1.0;
                amatrx(j, j) += 1.0;
                amatrx(i, j) -= 1.0;
                amatrx(j, i) -= 1.0;
            }
        }
    }

    for (size_t i = 1; i <= ndofel; i++)
    {
        for (size_t j = 1; j <= ndofel; j++)
        {
            rhs(i, 1) -= du(j, 1) * amatrx(i, j);
        }
        
    }

    torch::jit::script::Module* module_array = (torch::jit::script::Module*) SMAStructArrayAccess(1); 
    torch::jit::script::Module& module = module_array[0];

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.
    MUTEX_LINE(1,
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    )

}

extern "C" void FOR_NAME(uexternaldb,UEXTERNALDB)(
    const int&  lop,       // operation code
    const int&  lrestart,  // whether restart file is being written in this increment
    const Real(&time)[2],  // (1) step time; (2) total time
    const Real& dtime,     // time increment
    const int&  kstep,     // current step
    const int&  kinc       // current increment 
) {
    std::cout<<"lop = "<<lop<<std::endl;
    
    if(lop==0){
        // Determine the path of the pytorch *.pt file for the module
        char outdir[256];
        int lenoutdir;
        CALL_NAME(getoutdir, GETOUTDIR)(outdir, lenoutdir, 256);
        std::string module_path = std::string(outdir).substr(0, lenoutdir) + "\\traced_resnet_model.pt";

        // Load the pytorch *.pt file to SMAStructArray
        SMAStructArrayCreate(1, 1, sizeof(torch::jit::script::Module));
        torch::jit::script::Module* module_array = (torch::jit::script::Module*) SMAStructArrayAccess(1); 
        torch::jit::script::Module& module = module_array[0];
        module = torch::jit::load(module_path);

        MutexInit(1);
    }
    else if (lop==1 || lop==5)
    {

    }
    
}

