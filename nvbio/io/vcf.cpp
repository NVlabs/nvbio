/*
 * nvbio
 * Copyright (c) 2011-2014, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// loader for variant call format files, version 4.2

#include <nvbio/basic/console.h>
#include <nvbio/io/vcf.h>
#include <nvbio/io/bufferedtextfile.h>

#include <stdlib.h>

namespace nvbio {
namespace io {

// loads a VCF 4.2 file, appending the data to output
bool loadVCF(SNPDatabase& output, const char *file_name)
{
    BufferedTextFile file(file_name);
    char *line, *end;
    uint32 line_counter = 0;

    while((line = file.next_record(&end)))
    {
        line_counter++;
        *end = '\0';

        // strip out comments
        char *comment = strchr(line, '#');
        if (comment)
            *comment = '\0';

        // skip all leading whitespace
        while (*line == ' ' || *line == '\t' || *line == '\r')
        {
            line++;
        }

        if (*line == '\0')
        {
            // empty line, skip
            continue;
        }

        // parse the entries in each record
        char *chrom, *pos, *id, *ref, *alt, *qual, *filter;

// ugly macro to tokenize the string based on strchr
#define NEXT(prev, next)                        \
    {                                           \
        next = strchr(prev, '\t');              \
        if (!next) {                                                    \
            log_error(stderr, "Error parsing VCF file (line %d): incomplete variant\n", line_counter); \
            return false;                                               \
        }                                                               \
        *next = '\0';                                                   \
        next++;                                                         \
    }

        chrom = line;
        NEXT(chrom, pos);
        NEXT(pos, id);
        NEXT(id, ref);
        NEXT(ref, alt);
        NEXT(alt, qual);
        NEXT(qual, filter);

#undef NEXT

        // convert position and quality
        char *endptr = NULL;
        uint64 position = strtoll(pos, &endptr, 10);
        if (!endptr || endptr == pos || *endptr != '\0')
        {
            log_error(stderr, "VCF file error (line %d): invalid position\n", line_counter);
            return false;
        }

        uint8 quality;
        if (*qual == '.')
        {
            quality = 0xff;
        } else {
            quality = (uint8) strtol(qual, &endptr, 10);
            if (!endptr || endptr == qual || *endptr != '\0')
            {
                log_warning(stderr, "VCF file error (line %d): invalid quality\n", line_counter);
                quality = 0xff;
            }
        }

        // add an entry for each possible variant listed in this record
        do {
            char *next_base = strchr(alt, ',');
            if (next_base)
                *next_base = '\0';

            output.chromosome.push_back(std::string(chrom));
            output.position.push_back(position);
            output.reference.push_back(std::string(ref));

            // if this is a called monomorphic variant (i.e., a site which has been identified as always having the same allele)
            // we push the reference string as the variant
            if (strcmp(alt, ".") == 0)
                output.variant.push_back(std::string(ref));
            else
                output.variant.push_back(std::string(alt));

            output.variant_quality.push_back(quality);

            if (next_base)
                alt = next_base + 1;
            else
                alt = NULL;
        } while (alt && *alt != '\0');
    }

    return true;
}

} // namespace io
} // namespace nvbio
