
#include <aba_for_c.h>

extern "C" void FOR_NAME(getoutdir, GETOUTDIR) (CHNAME(outdir), int& lenoutdir CHLEN(outdir));

extern "C" void FOR_NAME(stdb_abqerr, STDB_ABQERR) (int& lop, CHNAME(format), int* intv, Real* realv, char* charv CHLEN(format));

extern "C" void FOR_NAME(xit, XIT)();