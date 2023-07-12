#!/bin/sh
#
# Batch submission script for a Dask workflow to run the deepmd optimization
#
#BSUB -P bif135-one
#BSUB -W 04:00
#BSUB -nnodes 2
#BSUB -J bif135-1-issue-18-debug
#BSUB -o out.deepmd.%J
#BSUB -e err.deepmd.%J
#BSUB -alloc_flags "gpumps"
#BSUB -q killable
#BSUB -B
#BSUB -N

date

# For wider logging
export COLUMNS=132

export SRC_DIR=/gpfs/alpine/proj-shared/chm187/mcoletti/deepmd_on_Summit/deepmd

# Make the run directory and move into it; we use the job ID to ensure we have  unique directory to catch all output.
# The run directory name is taken from the issue name on the kanban board.
export RUN_DIR=/gpfs/alpine/proj-shared/chm187/runs/18-new-runs-with-repaired-training-data/${LSB_JOBID}
if [ ! -d "$RUN_DIR" ]
then
	mkdir -p $RUN_DIR
fi
cd $RUN_DIR

# dask file for scheduler and workers to find each other
export SCHEDULER_FILE=${RUN_DIR}/scheduler_file.json

module purge
module load DefApps
module unload darshan-runtime
module load open-ce/1.5.2-py39-0 gcc/12.1.0
conda activate /gpfs/alpine/proj-shared/chm187/DPMD/deepmd-env

# We have to whack Summit on the nose with a newspaper to actually use the
# python in the activated environment
export PATH=/gpfs/alpine/proj-shared/chm187/DPMD/deepmd-env/bin/:$PATH

# Make sure gremlin can find our stuff
PYTHONPATH=${SRC_DIR}:$PYTHONPATH

# Copy over the hosts allocated for this job so that we can later verify
# that all the allocated nodes were busy with the correct worker allocation.
# Catches both the batch and compute nodes.
cat $LSB_DJOB_HOSTFILE | sort | uniq > $LSB_JOBID.hosts

# We need to figure out the number of nodes to later spawn the workers
NUM_NODES=$(cat $LSB_DJOB_HOSTFILE | sort | uniq | wc -l)
export NUM_NODES=$(expr $NUM_NODES - 1)

# hard coded number of Dask workers that will be spun up
#nWORKERS=2
#let x=$NUM_NODES y=$nWORKERS nWorker_Nodes=x/y
#let "nWORKERS=NUM_NODES*6"
#export nWORKERS=$nWORKERS
export nWORKERS=$NUM_NODES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=16
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=7
#export TF_GPU_ALLOCATOR=cuda_malloc_async

# Just echo stuff for reality check
echo "##########################################################################"
echo "Using python: " `which python3`
echo "PYTHONPATH: " $PYTHONPATH
echo "Source dir: $SRC_DIR"
echo "Run dir: $RUN_DIR"
echo "Dask scheduler file:" $SCHEDULER_FILE
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "NUMEXPR_MAX_THREADS: $NUMEXPR_MAX_THREADS"
echo "Number of nodes: $NUM_NODES"
echo "Number of workers: $nWORKERS"
echo "##########################################################################"

# gathering process ids for each step of the workflow.
# gotta kill them explicitly at the end of the script
# jskillall won't work
dask_pids=""

# Yes, running the scheduler on the batch node and not on some arbitrary compute
# node CPU. This allows for a homogenous compute node CPU allocation.  I.e.,
# if we did run the scheduler on a compute node, that's *one* node that will
# have so many cores running the scheduler, which means that the same number of
# cores on other nodes will be idle.
jsrun  --smpiargs="off" --gpu_per_rs 0 --nrs 1 --tasks_per_rs 1 --cpu_per_rs 2 --rs_per_host 1 dask scheduler --interface ib0 --no-dashboard --no-show --scheduler-file $SCHEDULER_FILE > dask-scheduler.out 2>&1 &
dask_pids="$dask_pids $!"

# Give the scheduler a chance to spin up.
sleep 5

# Now launch ALL the dask workers simultaneously.  They won't come up at the
# same time, though.  jsrun will be subprocess calls in Problem.evaluate() to
# get around stupid Summit/horovod MPI reset problem.
for ((i = 0; i < $NUM_NODES; i++)); do
  dask worker --nthreads 1 --nworkers 1 --interface ib0 \
  --no-dashboard --reconnect --scheduler-file $SCHEDULER_FILE &
  dask_pids="$dask_pids $!"
done

# Hopefully long enough for some workers to spin up and wait for work
echo Waiting for workers
sleep 5

# TODO Do we still need this nonsense?
export BIND="${SRC_DIR}/scripts/bind.sh --cpu=${SRC_DIR}/scripts/summit_map.sh --mem=${SRC_DIR}/scripts/summit_map.sh --"

# Run the dask client task manager on the launch/batch node with a single core.
python3 ${SRC_DIR}/deepmd-tuner.py ${SRC_DIR}/config/general.yaml ${SRC_DIR}/config/summit.yaml ${SRC_DIR}/config/debug.yaml

# shutting down dask scheduler and worker commands
# needed because these are running on the launch/batch nodes rather than through jsrun
for pid in $dask_pids
do
	kill -HUP $pid
done

# Finally, kill the job.
bkill $LSB_JOBID

echo Run finished.
date
