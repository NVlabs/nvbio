#ifndef __CMDLINE_H
#define __CMDLINE_H

#include <nvbio/basic/types.h>

using namespace nvbio;

struct runtime_options
{
    // genome file name or shared memory handle name
    const char *genome_file_name;
    // input reads file name
    const char *input_file_name;
    // output alignment file name
    const char *output_file_name;

    // whether to allow using mmap() to load the genome
    bool genome_use_mmap;
    // input batch size for reads
    uint64 batch_size;

    runtime_options()
    {
        genome_file_name = NULL;
        input_file_name = NULL;
        output_file_name = NULL;

        // default options
        genome_use_mmap = true;
        batch_size = 200000;
    };
};

extern struct runtime_options command_line_options;

void parse_command_line(int argc, char **argv);

#endif // ifndef __CMDLINE_H
