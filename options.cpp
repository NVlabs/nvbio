#include <stdio.h>
#include <stdlib.h>

#if !defined(WIN32)

#include <unistd.h>
#include <getopt.h>

#include <nvbio/basic/console.h>

#include "options.h"

struct runtime_options command_line_options;

static void usage(void)
{
    fprintf(stderr, "usage: nvmem [-f|--file-ref] <genome> <input.fastq> <output-file>\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  -f, --file-ref        Read the genome from file directly (do not use mmap)\n");
    fprintf(stderr, "\n");
    exit(1);
}

void parse_command_line(int argc, char **argv)
{
    static const char *options_short = "f:o";
    static struct option options_long[] = {
            { "file-ref",   no_argument,        NULL, 'f' },
            { NULL, 0, NULL, 0 },
    };

    int ch;
    while((ch = getopt_long(argc, argv, options_short, options_long, NULL)) != -1)
    {
        switch(ch)
        {
        case 'f':
            // -f, --file-ref
            command_line_options.genome_use_mmap = false;
            break;

        case '?':
        case ':':
        default:
            usage();
        }
    }

    if (optind != argc - 3)
    {
        // missing required arguments or too many arguments
        usage();
    }

    command_line_options.genome_file_name = argv[optind];
    command_line_options.input_file_name = argv[optind + 1];
    command_line_options.output_file_name = argv[optind + 2];
}

#endif