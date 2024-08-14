import numpy as np

class Lorenz96:
    """
    This class implements the Lorenz 96 model, a simplified atmospheric model designed by Edward Lorenz. 
    It is commonly used in the study of atmospheric dynamics and chaos theory. The model simulates the 
    behavior of a set of variables that could represent atmospheric quantities over a range of spatial 
    positions on a circular domain. The dynamics of the model are governed by a set of differential 
    equations that exhibit chaotic solutions under certain conditions, making it a useful tool for 
    research in numerical weather prediction and theoretical meteorology.

    Attributes:
        model_size (int): Specifies the number of spatial points in the model, representing the system's dimensionality.
        delta_t (float): The time step used for numerical integration.
        forcing (float): A constant term in the differential equations that acts as an external forcing, 
                        influencing the system's chaotic behavior.

    Methods:
        __init__(self, model_size=36, delta_t=0.01, forcing=8.0): Initializes a new instance of the Lorenz96 model 
                                                                with optional parameters for model size, time step, 
                                                                and external forcing.
    """
    def __init__(self, model_size=40, delta_t=0.05, forcing=8):
        """
        Initializes the Lorenz96 model with optional default values.

        Parameters:
        - model_size (int): The size of the model, default is 40.
        - delta_t (float): The time step, default is 0.05.
        - forcing (int): The forcing term, default is 8.
        """
        self.model_size = model_size
        self.delta_t = delta_t
        self.forcing = forcing

    def step(self, x):
        """
        Does a single time step advance for Lorenz 96  model using four-step Runge-Kutta time step.
        
        Parameters:
        - x: The model_size state.

        self.forcing: The forcing term
        self.delta_t: The time step.
        
        Returns:
        - x_new: The new state vector after the time step.
        """

        # Compute the four steps of the Runge-Kutta method using numpy operations
        dx = self.comp_dt(x)
        x1 = self.delta_t * dx
        inter = x + x1 / 2
        
        dx = self.comp_dt(inter)
        x2 = self.delta_t * dx
        inter = x + x2 / 2

        dx = self.comp_dt(inter)
        x3 = self.delta_t * dx
        inter = x + x3

        dx = self.comp_dt(inter)
        x4 = self.delta_t * dx

        x_new = x + x1/6 + x2/3 + x3/3 + x4/6

        return x_new


    def comp_dt(self, x):
        """
        Computes the derivative of the state vector using numpy operations for efficiency.
        
        Parameters:
        - x: The state vector.

        self.forcing: The forcing term.
        
        Returns:
        - dt: The derivative of the state vector.
        """
        dt = np.zeros(self.model_size)
        for j in range(self.model_size):
            jp1 = (j + 1) % self.model_size
            jm2 = (j - 2) % self.model_size
            jm1 = (j - 1) % self.model_size
            dt[j] = (x[jp1] - x[jm2]) * x[jm1] - x[j] + self.forcing
        return dt