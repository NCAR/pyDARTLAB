import numpy as np
from scipy.stats import norm

class InvalidVarianceError(Exception):
    """Exception raised for errors in the input variance."""
    pass

def obs_increment_eakf(ensemble, observation, obs_error_var):
    """
    Computes increments for an ensemble adjustment Kalman filter (EAKF).

    Parameters:
    - ensemble: numpy array representing the ensemble of prior state estimates.
    - observation: scalar representing the observation.
    - obs_error_var: scalar representing the observation error variance.

    Raises:
    - InvalidVarianceError: If both prior and observation error variance are <= 0.

    Returns:
    - obs_increments: numpy array representing the observation increments.
    """
    # Compute prior ensemble mean and variance
    prior_mean = ensemble.mean()
    prior_var = np.var(ensemble, ddof=1)  # ddof=1 for sample variance
    
    # If both prior and observation error variance are 0, raise an exception
    if prior_var <= 0 and obs_error_var <= 0:
        raise InvalidVarianceError("Both prior variance and observation error variance are non-positive.")

    # Compute the posterior mean and variance
    if prior_var == 0:
        post_mean = prior_mean
        post_var = 0
    elif obs_error_var == 0:
        post_mean = observation
        post_var = 0
    else:
        post_var = 1 / (1 / prior_var + 1 / obs_error_var)
        post_mean = post_var * (prior_mean / prior_var + observation / obs_error_var)


    # Shift the prior ensemble to have the posterior mean
    updated_ensemble = ensemble - prior_mean + post_mean
   
    # Contract the ensemble to have the posterior variance
    if prior_var > 0:  # Avoid division by zero
        var_ratio = post_var / prior_var
        updated_ensemble = (updated_ensemble - post_mean) * np.sqrt(var_ratio) + post_mean
        
    # Compute the increments
    obs_increments = updated_ensemble - ensemble
    
    return obs_increments


def obs_increment_enkf(ensemble, observation, obs_error_var):
    """
    Computes increments for an ensemble Kalman filter with perturbed obs mean correction.

    Parameters:
    - ensemble: numpy array representing the ensemble of prior state estimates.
    - observation: scalar representing the observation.
    - obs_error_var: scalar representing the observation error variance.

    Raises:
    - InvalidVarianceError: If both prior and observation error variance <= 0.

    Returns:
    - obs_increments: numpy array representing the observation increments.
    """
    # Compute prior ensemble mean and variance
    prior_mean = np.mean(ensemble)
    prior_var = np.var(ensemble, ddof=1)  # ddof=1 for sample variance

    # If both prior and observation error variance are zero raise InvalidVarianceError
    if prior_var <= 0 and obs_error_var <= 0:
        raise InvalidVarianceError("Both prior and observation error variance are <=0.")

    # Compute the posterior mean and variance
    if prior_var == 0:
        post_mean = prior_mean # not used
        post_var = 0
    elif obs_error_var == 0:
        post_mean = observation # not used
        post_var = 0
    else:
        # Use product of gaussians
        post_var = 1 / (1 / prior_var + 1 / obs_error_var)
        post_mean = post_var * (prior_mean / prior_var + observation / obs_error_var) # not used

    # Generate the perturbed observations by adding draw from Normal(0, obs_error_var)
    temp_obs = observation + np.sqrt(obs_error_var) * np.random.randn(*ensemble.shape)

    # Adjust so that perturbed observations have mean = to observation
    temp_obs = temp_obs - np.mean(temp_obs) + observation

    # Compute new ensemble members by taking product of prior ensemble members and perturbed obs pairs
    updated_ens = post_var * (ensemble / prior_var + temp_obs / obs_error_var)

    # Increments are difference between updated and original ensemble
    obs_increments = updated_ens - ensemble

    return obs_increments

def obs_increment_rhf(ensemble, observation, obs_error_var):
    """
    Computes increments for a rank histogram filter.
    """
    
    # Get the ensemble size
    ens_size = ensemble.shape[0]
    prior_sd = np.std(ensemble)
    prior_var = prior_sd**2 # not used

    # Allocate space for likelihood density calculations
    like_dense = np.zeros(ens_size)
    mass = np.zeros(ens_size)
    height = np.zeros(ens_size)

    # Sort the ensemble members and keep the indices
    x, e_ind = np.sort(ensemble), np.argsort(ensemble)

    # Compute the likelihood of each member given the observation
    like = np.exp(-1 * (x - observation)**2 / (2 * obs_error_var))

    # Compute the mean likelihood density in each interior bin
    like_dense[1:] = (like[:-1] + like[1:]) / 2

    # For unit normal, find distance from mean to where cdf is 1/(n+1)
    dist_for_unit_sd = -1 * weighted_norm_inv(1, 0, 1, 1 / (ens_size + 1))

    # Variance of tails is just sample prior variance
    # Mean is adjusted so that 1/(ens_size + 1) is outside
    left_mean = x[0] + dist_for_unit_sd * prior_sd
    left_var = prior_var
    left_sd = prior_sd

    # Same for the right tail
    right_mean = x[-1] - dist_for_unit_sd * prior_sd
    right_var = prior_var
    right_sd = prior_sd
    
    # Flat tails for likelihood

    # left tail
    new_sd_left  = left_sd 
    new_mean_left = left_mean 
    prod_weight_left = like[0] / (ens_size + 1)
    mass[0] = like[0] / (ens_size + 1)

    # right tail
    new_sd_right = right_sd
    new_mean_right = right_mean
    prod_weight_right = like[-1]
    mass[-1] = like[-1] / (ens_size + 1)


    # The mass in each interior box is the height times the width
    # The height of the likelihood is like_dense
    # For the prior, mass is 1 / ((n+1) width), and mass = height x width so
    # The height of the prior is 1 / ((n+1) width); Multiplying by width leaves 1/(n+1)
    # In prior, have 1/(n+1) mass in each bin, multiply by mean likelihood
    # density to get approximate mass in updated bin

    for i in range(1, ens_size):
        mass[i] = like_dense[i] / (ens_size + 1)
        if x[i] == x[i-1]:
            height[i] = -1
        else:
            height[i] = 1 / ((ens_size + 1) * (x[i] - x[i-1]))

    # Normalize the mass to get a pdf
    mass_sum = np.sum(mass)
    nmass = mass / mass_sum

    # Get the weight for the final normalized tail gaussians
    # This is the same as left_amp=(ens_size + 1)*nmass(1)
    left_amp = prod_weight_left / mass_sum
    # This is the same as right_amp=(ens_size + 1)*nmass(ens_size+1)
    right_amp = prod_weight_right / mass_sum

    # Find cumulative mass at each box boundary
    cumul_mass = np.zeros(ens_size + 1)
    cumul_mass[1:] = np.cumsum(nmass)

    # Begin internal box search at bottom of lowest box
    new_ens = np.zeros(ens_size)
    obs_increments = np.zeros(ens_size)
    lowest_box = 1

    # Find each new ensemble member's location
    for i in range(ens_size):
        # Each update ensemble member has 1/(n+1) mass before it
        umass = (i + 1) / (ens_size + 1)
        
        # If it is in the inner or outer range have to use normal
        if umass < cumul_mass[1]:
            # It is in the left tail
            # Get position of x in weighted gaussian where the cdf has value umass
            new_ens[i] = weighted_norm_inv(left_amp, new_mean_left, new_sd_left, umass)
        elif umass > cumul_mass[ens_size]:
            # It's in the right tail
            # Get the position of x in weighted gaussian where the cdf has value umass
            new_ens[i] = weighted_norm_inv(right_amp, new_mean_right, new_sd_right, 1 - umass)
            # Coming in from the right, use symmetry after pretending it's on left
            new_ens[i] = new_mean_right + (new_mean_right - new_ens[i])
        else:
            # In one of the inner boxes
            for j in range(lowest_box, ens_size):
                # Find the box that this mass is in
                if cumul_mass[j + 1] <= umass <= cumul_mass[j + 2]:
                    # Linearly interpolate in mass
                    new_ens[i] = x[j] + ((umass - cumul_mass[j + 1]) / (cumul_mass[j + 2] - cumul_mass[j + 1])) * (x[j + 1] - x[j])
                    lowest_box = j
                    break

    # Convert to increments for unsorted
    for i in range(ens_size):
        obs_increments[e_ind[i]] = new_ens[i] - x[i]

    return obs_increments

def weighted_norm_inv(alpha, mean, sd, p):
    """
    Find the value of x for which the cdf of a N(mean, sd)
    multiplied times alpha has value p.
    """
    np = p / alpha
    x = norm.ppf(np)
    x = mean + x * sd
    return x

