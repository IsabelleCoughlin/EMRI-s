class EMRIParams:
    def __init__(self, M_bh, mu, X=0.0, lambda_var=0.0):
        """
        M_bh, m: masses in SI units
        X: spin parameter S/M^2
        lambda_var: spin-orbit misalignment angle (radians)
        """
        from .constants import G, c

        self.M_si = M_bh
        self.mu_si = mu
        
        # Convert to geometric seconds
        self.M_seconds = G*M_bh / c**3
        self.mu_seconds = G*mu / c**3
        

        self.X = X
        self.lambda_var = lambda_var