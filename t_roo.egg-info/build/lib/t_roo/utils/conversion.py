import numpy as np

def lambdaTilde_from_lambda1_lambda2_eta(lambda_1, lambda_2, eta):
    
    lambda_tilde = 8./13. * ((1. + 7.*eta - 31.* eta*eta)*(lambda_1+lambda_2) + np.sqrt(1 - 4.*eta)*(1. + 9.*eta - 11.*eta*eta)*(lambda_1 - lambda_2) )

    return lambda_tilde 

def symmetric_mass_ratio_from_coords(coords, params_to_inds):

        if 'mass_ratio' in params_to_inds:
            idx_q = params_to_inds['mass_ratio']
            eta = symmetric_mass_ratio_from_mass_ratio(coords[:, idx_q])
        elif ('mass_1' in params_to_inds) or ('mass_2' in params_to_inds):
            idx_m1 = params_to_inds['mass_1']
            idx_m2 = params_to_inds['mass_2']
            q = coords[:,idx_m2] / coords[:,idx_m1]
            eta = symmetric_mass_ratio_from_mass_ratio(q)
        else:
            raise ValueError("Unrecognized mass parameters")

        return eta          

def symmetric_mass_ratio_from_mass_ratio(q):
        eta = q / ((1.+q)*(1.+q))
        return eta
