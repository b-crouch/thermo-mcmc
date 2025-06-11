import numpy as np
import scipy.interpolate
from phonopy import Phonopy
import matplotlib.pyplot as plt
from .model import LogLikelihoodModel
from phonopy.structure.atoms import PhonopyAtoms


class DOSModel(LogLikelihoodModel):
    def __init__(self, H, positions, element="Ni", interval=[-1, 11], step=0.05, smear=0.05):
        """
        Initialize log-likelihood model for DOS reconstruction
        Params:
            H (hessym.SymbolicHessian): Symbolic Hessian object representing system of interest
            positions (np.array): Atomic coordinates of system
            element (str): String name of element being modeled
            interval (list[int]): interval of frequency values over which DOS should be generated
            interval (float): Frequency interval for plotting DOS points
            smear (float): Smearing width to use in smoothing DOS curve
        """
        self.H = H
        self.hessian_parameters = H.get_parameters()
        self.n_parameters = len(self.hessian_parameters)
        self.n_atoms = len(positions)
        self.unitcell = PhonopyAtoms(symbols=[element]*self.n_atoms,
                                     cell=np.eye(3),
                                     scaled_positions=positions)
        self.calculator = Phonopy(self.unitcell)
        self.interval = interval
        self.smear = smear
        self.step = step
        self.is_multimodel = False

    def gen_quantity(self, omega, mesh=16):
        """
        Generate DOS for parameter set `omega`
        """
        # Populate the Hessian
        parameter_map = {self.hessian_parameters[i]:omega[i] for i in range(self.n_parameters)}
        reconstructed_hessian, _ = self.H.reconstruct_hessian(symbol_mapping=parameter_map)

        # Compute phonons
        self.calculator.force_constants = reconstructed_hessian.reshape(self.n_atoms, 3, self.n_atoms, 3).transpose(0, 2, 3, 1)
        self.calculator.save()
        self.calculator.run_mesh([mesh, mesh, mesh])
        self.calculator.run_total_dos(freq_min=self.interval[0], freq_max=self.interval[1], sigma=self.smear, freq_pitch=self.step)
        frequencies, dos = self.calculator.get_total_DOS()
        return frequencies, dos
    
    def gen_free_energy(self, omega, T, atomic_mass, mode="classical"):
        """
        Generate free energy curve for parameter set `omega`
        """
        parameter_map = {self.hessian_parameters[i]:omega[i] for i in range(self.n_parameters)}
        F, F_per_atom = self.H.compute_free_energy(self, atomic_mass, T, symbol_mapping=parameter_map, mode=mode, verbose=False)
        return F, F_per_atom


class CvModel(LogLikelihoodModel):
    def __init__(self, H, positions, element="Ni", interval=[10, 1000], step=10, scale_factor=None):
        """
        Initialize log-likelihood model for heat capacity curve reconstruction
        Params:
            H (hessym.SymbolicHessian): Symbolic Hessian object representing system of interest
            positions (np.array): Atomic coordinates of system
            element (str): String name of element being modeled
            interval (list[int]): Range of temperature values over which Cv should be generated
            step (float): Temperature step for plotting Cv points
        """
        self.H = H
        self.hessian_parameters = H.get_parameters()
        self.n_parameters = len(self.hessian_parameters)
        self.n_atoms = len(positions)
        self.unitcell = PhonopyAtoms(symbols=[element]*self.n_atoms,
                                     cell=np.eye(3),
                                     scaled_positions=positions)
        self.calculator = Phonopy(self.unitcell)
        self.interval = interval
        self.step = step
        self.scale_factor = scale_factor if scale_factor is not None else cv_conversion_factor
        self.is_multimodel = False
        
    def gen_quantity(self, omega):
        """
        Generate Cv curve for parameter set `omega`
        """
        # Populate the Hessian
        parameter_map = {self.hessian_parameters[i]:omega[i] for i in range(self.n_parameters)}
        reconstructed_hessian, _ = self.H.reconstruct_hessian(symbol_mapping=parameter_map)

        # Compute phonons
        self.calculator.force_constants = reconstructed_hessian.reshape(self.n_atoms, 3, self.n_atoms, 3).transpose(0, 2, 3, 1)
        self.calculator.save()
        self.calculator.run_mesh([32, 32, 32])
        self.calculator.run_thermal_properties(t_min=self.interval[0], t_max=self.interval[1], t_step=self.step)
        thermal_dict = self.calculator.get_thermal_properties_dict()
        temperatures, cv = thermal_dict["temperatures"], self.scale_factor(thermal_dict["heat_capacity"])
        return temperatures, cv
    

class PhononModel(LogLikelihoodModel):
    def __init__(self, H, positions, element="Ni", freq_interval=[-1, 11], T_interval=[10, 1000], freq_step=0.05, T_step=10, smear=0.05, cv_scale_factor=None):
        self.H = H
        self.hessian_parameters = H.get_parameters()
        self.n_parameters = len(self.hessian_parameters)
        self.n_atoms = len(positions)
        self.unitcell = PhonopyAtoms(symbols=[element]*self.n_atoms,
                                     cell=np.eye(3),
                                     scaled_positions=positions)
        self.calculator = Phonopy(self.unitcell)
        self.freq_interval = freq_interval
        self.T_interval = T_interval
        self.freq_step = freq_step
        self.T_step = T_step
        self.smear = smear
        self.cv_scale_factor = cv_scale_factor if cv_scale_factor is not None else cv_conversion_factor
        self.is_multimodel = True

    def gen_quantity(self, omega, mesh=16):
        # Populate the Hessian
        parameter_map = {self.hessian_parameters[i]:omega[i] for i in range(self.n_parameters)}
        reconstructed_hessian, _ = self.H.reconstruct_hessian(symbol_mapping=parameter_map)

        # Compute phonons
        self.calculator.force_constants = reconstructed_hessian.reshape(self.n_atoms, 3, self.n_atoms, 3).transpose(0, 2, 3, 1)
        self.calculator.save()
        self.calculator.run_mesh([mesh, mesh, mesh])
        
        self.calculator.run_total_dos(freq_min=self.freq_interval[0], freq_max=self.freq_interval[1], sigma=self.smear, freq_pitch=self.freq_step)
        frequencies, dos = self.calculator.get_total_DOS()
        
        self.calculator.run_thermal_properties(t_min=self.T_interval[0], t_max=self.T_interval[1], t_step=self.T_step)
        thermal_dict = self.calculator.get_thermal_properties_dict()
        temperatures, cv = thermal_dict["temperatures"], self.cv_scale_factor(thermal_dict["heat_capacity"])

        output = {"dos":{"x":frequencies, "y":dos}, "cv":{"x":temperatures, "y":cv}}
        return output

    def gen_prediction(self, omega, x_cv, x_dos):
        predicted_vals = self.gen_quantity(omega)
        
        # Fit spline to learned quantities so predictions can be made at input `x` values
        dos_spline_fit = scipy.interpolate.make_smoothing_spline(predicted_vals["dos"]["x"], predicted_vals["dos"]["y"])
        dos_predictions = dos_spline_fit(x_dos)

        cv_spline_fit = scipy.interpolate.make_smoothing_spline(predicted_vals["cv"]["x"], predicted_vals["cv"]["y"])
        cv_predictions = cv_spline_fit(x_cv)

        predictions = {"dos":dos_predictions, "cv":cv_predictions}

        return predictions

    def gen_log_likelihood(self, omega, cv_data, dos_data):
        sigma_cv, sigma_dos = omega[-2], omega[-1]
        hessian_params = omega[:-2]
        predictions = self.gen_prediction(hessian_params, cv_data["x"], dos_data["x"])


        log_lik = np.sum(-0.5*np.log(2*np.pi*sigma_cv**2) - 0.5*((cv_data["y"]-predictions["cv"])/sigma_cv)**2) + np.sum(-0.5*np.log(2*np.pi*sigma_dos**2) - 0.5*((dos_data["y"]-predictions["dos"])/sigma_dos)**2)
        return log_lik
    
    def plot(self, omega, ax=None, label="", xlim=None, c="tab:blue"):
        """
        Plot the predicted quantity generated by the parameter set `omega`
        """
        predicted_vals = self.gen_quantity(omega)

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(predicted_vals["cv"]["x"], predicted_vals["cv"]["y"], label=label, c=c)
        ax[0].set_xlabel("Temperature")
        ax[0].set_ylabel("Cv")

        ax[1].plot(predicted_vals["dos"]["x"], predicted_vals["dos"]["y"], label=label, c=c)
        ax[1].set_xlabel("Frequency")
        ax[1].set_ylabel("DOS")

        if xlim is not None:
            plt.xlim(*xlim);

def cv_conversion_factor(cv):
    return cv/32/58.69*1000