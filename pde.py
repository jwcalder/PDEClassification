import numpy as np

class wave_eq_solver:

    def __init__(self):
        """Wave equation solver
        ========

        Class for solving the wave equation in batches.
        """

    def solve(self, f):
        """Solve
        ======

        Solves the wave equation
        \\[u_{tt} - u_{xx} = f\\]
        subject to a forcing term \\(f\\). Currently uses the 
        unit box \\(0 < x < 1, 0 < t < 1\\) as the domain
        with boundary conditions \\(u(0,t)=u(1,t)=0\\) and 
        initial conditions \\(u(x,0)=u_t(x,0)=0\\). Uses the 
        explicit forward Euler method with half the largest allowable
        time step \\(dt=dx/2\\).

        The code supports solving a batch of b eqautions in parallel,
        where f has shape (b,n,n).
        
        Parameters
        ----------
        f : (b,n,n) numpy array, float
            Forcing term as a function of space and time. 

        Returns
        -------
        u : (b,n,n) numpy array, float
            Solution of wave equation.
        """

        b = f.shape[0]
        n = f.shape[1]
        u = np.zeros((b,n+2,n+2))
        dx = 1/(n+1)
        dt = dx/2
        s = dt**2/dx**2

        for k in range(1,n+1):
            u[:,k+1,1:-1] = 2*(1-s)*u[:,k,1:-1] - u[:,k-1,1:-1] + s*(u[:,k,2:] + u[:,k,:-2])  + dx**2*f[:,k-1,:]

        return u[:,2:,1:-1]



class heat_eq_solver:

    def __init__(self,n):
        """Heat equation solver
        ========

        Class for solving the heat equation in batches.
        Uses implicit scheme with \\(dt = dx/2\\).

        Parameters
        ----------
        n : int
            Number of grid points
        """

        self.n = n
        dx = 1/(n+1); self.dx = dx
        dt = dx/2; self.dt = dx/2
        s = dt/dx**2; self.s = s

        A = (1+2*s)*np.diag(np.ones(n),k=0) - s*(np.diag(np.ones(n-1),k=1) + np.diag(np.ones(n-1),k=-1))

        self.A = A
        self.Ainv = np.linalg.inv(A)
        self.AinvT = (self.Ainv).T

    def solve(self, f):
        """Solve
        ======

        Solves the heat equation
        \\[u_{t} - u_{xx} = f\\]
        subject to a forcing term \\(f\\). Currently uses the 
        unit box \\(0 < x < 1, 0 < t < 1\\) as the domain
        with boundary conditions \\(u(0,t)=u(1,t)=0\\) and 
        initial condition \\(u(x,0)=0\\). Uses an
        implicit scheme with time step \\(dt=dx/2\\).

        The code supports solving a batch of b eqautions in parallel,
        where f has shape (b,n,n).
        
        Parameters
        ----------
        f : (b,n,n) numpy array, float
            Forcing term as a function of space and time. 

        Returns
        -------
        u : (b,n,n) numpy array, float
            Solution of wave equation.
        """

        b = f.shape[0]
        n = self.n
        dx = self.dx
        dt = self.dt
        s = self.s

        u = np.zeros((b,n+1,n+2))
        for k in range(n):
            rhs = u[:,k,1:-1] + dt*f[:,k,:]
            u[:,k+1,1:-1] = rhs@self.AinvT

        return u[:,1:,1:-1]





