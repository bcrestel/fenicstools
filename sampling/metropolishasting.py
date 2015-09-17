from numpy.random import uniform

def MetropolisHasting(x0, proposal, posterior, nbsamples):
    """
    Run the Metropolis-Hasting algorithm for 'posterior', from starting point
    x0, with transition kernel 'proposal'
    Inputs:
        x0 = starting position
        proposal = object with methods sample(old) => generate sample and
        density(x,y) => generate transition density from y to x, i.e.,
            density(x,y) = p(y|x)
        posterior(x) = return density of the posterior at x
        nbsamples = nb of samples to be generated
    """
    samples = []
    samples.append(x0)
    old = x0
    acc = 0
    for ii in range(nbsamples):
        prop = proposal.sample(old) # sample from proposal
        # Use log posterior to compute that? Round-off error
        alpha = min( 1.0, posterior(prop)*proposal.density(old, prop)/\
        (posterior(old)*proposal.density(prop, old)) )  # acceptance rate
        if uniform() < alpha:
            old = prop
            acc += 1
        samples.append(old)
    return samples, acc
