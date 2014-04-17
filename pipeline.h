#ifndef __ALIGNER_H
#define __ALIGNER_H

#include <nvbio/io/output/output_file.h>

#include "mem-search.h"

using namespace nvbio;

struct pipeline_context 
{
    io::OutputFile *output;
    struct mem_state mem;
};

#endif // ifndef __ALIGNER_H
