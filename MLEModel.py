import lmfit
import operator

def _residual_mle(self, params, data, weights, **kwargs):
    """Return the an array which when squared and summed is the malimum likelihood value we want to optimize.
        """
    if weights is not None:
        raise Exception("MLE residuals does not support weights")
    y = self.eval(params, **kwargs)
    r2 = y-data
    nonzero = data
    r2[nonzero] += data[nonzero]*np.log((data/y)[nonzero])
    diff_mle = (2*r2) ** 0.5
    diff_mle[y < data] *= -1
    return np.asarray(diff_mle).ravel()

class MLEModel(lmfit.model.Model):
    _residual = _residual_mle

    def __add__(self, other):
        """+"""
        return CompositeMLEModel(self, other, operator.add)

    def __repr__(self):
        """Return representation of Model."""
        return "<MLEModel: %s>" % (self.name)

class CompositeMLEModel(lmfit.model.CompositeModel):
    _residual = _residual_mle

    def __add__(self, other):
        """+"""
        return CompositeMLEModel(self, other, operator.add)

    def __repr__(self):
        """Return representation of Model."""
        return "<CompositeMLEModel: %s>" % (self.name)

def gaussian(x, amplitude, center, sigma):
    return (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))


if __name__ == "__main__":
    m = MLEModel(gaussian,prefix="g0")
    m2 = MLEModel(gaussian,prefix="g1")
    z = m+m2
