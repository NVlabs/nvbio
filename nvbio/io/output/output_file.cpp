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

#include <nvbio/io/output/output_batch.h>
#include <nvbio/io/output/output_sam.h>
#include <nvbio/io/output/output_bam.h>
#include <nvbio/io/output/output_debug.h>

namespace nvbio {
namespace io {

OutputFile::OutputFile(const char *_file_name, AlignmentType _alignment_type, BNT _bnt)
    : file_name(_file_name),
      alignment_type(_alignment_type),
      bnt(_bnt),
      mapq_filter(-1),
      read_data_1(NULL),
      read_data_2(NULL)
{
}

OutputFile::~OutputFile()
{
}

void OutputFile::configure_mapq_evaluator(int mapq_filter)
{
    OutputFile::mapq_filter = mapq_filter;
}

void OutputFile::start_batch(const io::SequenceDataHost *read_data_1,
                             const io::SequenceDataHost *read_data_2)
{
    // stash the current host pointer for the read data
    OutputFile::read_data_1 = read_data_1;
    OutputFile::read_data_2 = read_data_2;
}

void OutputFile::process(struct DeviceOutputBatchSE& gpu_batch,
                         const AlignmentMate alignment_mate)
{
    // do nothing
}

void OutputFile::end_batch(void)
{
    // invalidate the read data pointers
    read_data_1 = NULL;
    read_data_2 = NULL;
}

void OutputFile::close(void)
{
}

IOStats& OutputFile::get_aggregate_statistics(void)
{
    return iostats;
}

void OutputFile::readback(struct HostOutputBatchPE& cpu_batch,
                          const struct DeviceOutputBatchSE& gpu_batch,
                          const AlignmentMate mate)
{
    Timer timer;
    timer.start();

    // read back this mate
    gpu_batch.readback_scores(cpu_batch.alignments[mate]);
    gpu_batch.readback_cigars(cpu_batch.cigar[mate]);
    gpu_batch.readback_mds(cpu_batch.mds[mate]);

    if (mate == MATE_1)
    {
        // mapq and read-ids come from MATE_1
        gpu_batch.readback_mapq(cpu_batch.mapq);
        gpu_batch.readback_ids(cpu_batch.read_ids);

        // stash the count for mate1, mate2 must match
        cpu_batch.count = gpu_batch.count;

        // set up the read data pointers
        // this is not strictly related to which mate or scoring pass we're processing,
        // but must be done once per batch, so we do it here
        cpu_batch.read_data[MATE_1] = read_data_1;
        cpu_batch.read_data[MATE_2] = read_data_2;
    }

    // sanity check to make sure the number of reads matches what we got previously
    // (for mate 1 best score this will always pass due to the assignment above)
    NVBIO_CUDA_ASSERT(cpu_batch.count == gpu_batch.count);

    timer.stop();
    iostats.output_process_timings.add(gpu_batch.count, timer.seconds());

    iostats.alignments_DtoH_time += timer.seconds();
    iostats.alignments_DtoH_count += gpu_batch.count;
}

OutputFile *OutputFile::open(const char *file_name, AlignmentType aln_type, BNT bnt)
{
    // parse out file extension; look for .sam, .bam suffixes
    uint32 len = uint32(strlen(file_name));

    if (strcmp(file_name, "/dev/null") == 0)
    {
        return new OutputFile(file_name, aln_type, bnt);
    }

    if (len >= strlen(".sam"))
    {
        if (strcmp(&file_name[len - strlen(".sam")], ".sam") == 0)
        {
            return new SamOutput(file_name, aln_type, bnt);
        }
    }

    if (len >= strlen(".bam"))
    {
        if (strcmp(&file_name[len - strlen(".bam")], ".bam") == 0)
        {
            return new BamOutput(file_name, aln_type, bnt);
        }
    }

    if (len >= strlen(".dbg"))
    {
        if (strcmp(&file_name[len - strlen(".dbg")], ".dbg") == 0)
        {
            return new DebugOutput(file_name, aln_type, bnt);
        }
    }

    log_warning(stderr, "could not determine file type for %s; guessing SAM\n", file_name);
    return new SamOutput(file_name, aln_type, bnt);
}

} // namespace io
} // namespace nvbio
