import scipy
import numpy as np 
import matplotlib.pyplot as plt

class EoS:
    """
    Class for fitting Birch-Murnaghan equations of state
    """
    def __init__(self):
        self.V0, self.E0, self.B0, self.BP = [None]*4

    def eos(self, parameters, V):
        """
        Computes energy values for given volumes following the Birch-Murnaghan formulation
        """
        V0, E0, B0, BP = parameters
        f = ((V0/V)**(2/3) - 1)/2
        E = E0 + V0*(9*B0/2*f**2 + 27*B0*(BP-4)/6*f**3 )
        return E

    def fit(self, V, E):
        """
        Fits Birch-Murnaghan EoS to provided volumes and energies
        """
        p0 = self.gen_initial_guess(V, E)
        parameters, _ = scipy.optimize.leastsq(lambda parameters: E - self.eos(parameters, V), p0, ftol=1e-12, xtol=1e-12)
        self.V0, self.E0, self.B0, self.BP = parameters

    def predict(self, V):
        """
        Applies fitted EOS to predict energies for provided volumes
        """
        assert self.V0 is not None, "Must call .fit before predicting!"
        return self.eos([self.V0, self.E0, self.B0, self.BP], V)
        
    def gen_initial_guess(self, V, E):
        """
        Generates initial parameter guess for optimization using simple quadratic fit
        """
        a, b, c = np.polyfit(V, E, 2)
        V0 = -b/(2*a)
        E0 = a*V0**2 + b*V0 + c
        B0 = 2*a*V0
        BP = 4
        p0 = [V0, E0, B0, BP]
        return p0

    def get_params(self):
        """
        Retrieves dictionary of BM fit parameters
        """
        return {"B0":self.B0, "V0":self.V0, "E0":self.E0, "BP":self.BP}

    def plot(self, V, E, title=""):
        """
        Plot cold curve
        """
        V_plot = np.linspace(np.min(V), np.max(V), 1000)
        E_plot = self.predict(V_plot)
        plt.scatter(V, E)
        plt.plot(V_plot, E_plot)
        plt.xlabel(r"Volume ($Å^3$)")
        plt.ylabel("Energy (eV)")
        plt.title(title);