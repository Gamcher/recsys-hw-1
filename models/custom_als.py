import numpy as np
from models.als import ExplicitALS


class CustomALS(ExplicitALS):
    """Change regression target itself, not confidence weights."""
    def __init__(self, alpha=.5, tau_days=365, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.tau_days = alpha, tau_days

    def target(self, frame):
        if not 0 <= self.alpha <= 1 or self.tau_days <= 0:
            raise ValueError('Invalid target parameters')
        # Reference time comes exclusively from the training frame.
        age_days = (frame.unixReviewTime.max()-frame.unixReviewTime.to_numpy()) / 86400
        return ((frame.overall.to_numpy()-3)/2) * (
            self.alpha+(1-self.alpha)*np.exp(-age_days/self.tau_days))
