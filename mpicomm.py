"""
Create local and global communicators for coarse-grained parallelism
"""
import numpy as np
import dolfin as dl

def create_communicators():
    """
    Create communicators for simple coarse-grained parallelism
    where all PDEs are solved on a single core, and partition occured at the
    level of the source terms and in the time-summations for the grad and the 
    Hessian-vector product
    """
    mpicomm_local = dl.mpi_comm_self()
    mpicomm_global = dl.mpi_comm_world()

    return mpicomm_local, mpicomm_global


def partition_work(mpicomm_local, mpicomm_global, nb_sources, nb_timesteps):
    """
    Determine what source to use (locally), and what time-steps to integrate
    over in the gradient and Hessian-vector steps
    Arguments:
        mpicomm_local = local MPI communicator [out from create_communicators]
            all proc are assumed to have same size of mpicomm_local partition
        mpicomm_global = global MPI communicator [out from create_communicators]
        nb_sources = total number of source terms used
        nb_timesteps = total number of time steps for interval [0, T]
    """
    # ranks and sizes
    mpiworldsize = dl.MPI.size(dl.mpi_comm_world())
    mpilocalsize = dl.MPI.size(mpicomm_local)
    mpiglobalsize = dl.MPI.size(mpicomm_global)

    mpilocalrank = dl.MPI.rank(mpicomm_local)
    mpiglobalrank = dl.MPI.rank(mpicomm_global)

    # Checks and balances
    N = mpiworldsize / mpilocalsize
    assert mpiworldsize - N*mpilocalsize == 0
    assert N == mpiglobalsize

    if mpiglobalsize < nb_sources:
        nbsrc = int(np.ceil(float(nb_sources)/mpiglobalsize))
        firstindex = mpiglobalrank*nbsrc
        lastindex = min((mpiglobalrank+1)*nbsrc, nb_sources)
        return range(firstindex, lastindex), range(nb_timesteps)
    else:
        # Share local partitions over source terms
        ranks = np.array(range(mpiglobalsize), dtype=np.int)
        rank2src = (ranks*nb_sources)/mpiglobalsize
        mysrcnb = rank2src[mpiglobalrank]

        # Find nb of local partitions working on same source
        allglobranks = np.where(rank2src == mysrcnb)[0]
        nbproc = len(allglobranks)

        # Partition time-steps
        timestepspacketsize = int(np.ceil(float(nb_timesteps)/nbproc))
        timestepindex = int(np.where(allglobranks == mpiglobalrank)[0])
        t0 = timestepindex*timestepspacketsize
        tn = min((timestepindex+1)*timestepspacketsize, nb_timesteps)
        timesteps = range(t0, tn)

        return [mysrcnb], timesteps
