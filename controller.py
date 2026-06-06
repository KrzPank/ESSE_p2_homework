import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class contr():
    
  def __init__(self, name=""):
      self.name = name
      self._log = []
      self.e_curr = 0.0

    
  def derivative(self, x, t):
    dx_dt = -8.0 * x + 65540.0 * self.e_curr
    return dx_dt

  # added output method
  def out(self, x):
    return float(-115400.0 * x + 1.08e+09 * self.e_curr)

        
  def plot(self, no_sp=False):
    t,sp,pv,cv = np.asarray(self._log).transpose()
        
    plt.subplot(2,1,1)
    plt.plot(t, pv, label=self.name + ' PV')
    if no_sp == False:
      plt.plot(t, sp, '--', color="r", label=self.name + ' SP')
    plt.legend()
    plt.title('Process Variable and Setpoint')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(t, cv, label=self.name + ' CV')
    plt.title('Control Variable')
    plt.legend()
    plt.grid()
    plt.tight_layout()
        
  def generator(self,dt, ic=0.0):
    cv = 0.0
    x = ic
    while True:
      t,sp,pv,cv = yield cv
      cv = self.out(x)
      self._log.append([t,sp,pv,cv]) 
      self.e_curr = sp-pv
      x = odeint(self.derivative, x, [t, t+dt])[-1]
      t += dt


class satellite():
  def __init__(self, j=10.8E8, name=""):
    self.name = name
    self.j = j
    self._log = []
    self.u = 0.0
  
  def derivative(self, x, t):
    x1, x2 = x
    dx_dt = [self.u, x1]
    return dx_dt

  # added output method
  def out(self, x):
    x1, x2 = x;
    return 1/self.j * x2

  def plot(self):
    t, u, x1, x2 = np.asarray(self._log, dtype=object).transpose()
    plt.plot(t, x1, label=self.name + " state: $x_1$")
    plt.plot(t, x2, label=self.name + " state: $x_2$")
    plt.legend()
  
  def generator(self, dt, ic=[0, 0]):
    x = ic
    while True:
      t, self.u = yield self.out(x)
      x = odeint(self.derivative, x, [t, t+dt])[-1]
      x1, x2 = x
      self._log.append([t, self.u, x1, x2])
      t += dt

def simulate_comparison(dt=0.1, T=100.0, sp_deg=10.0):
    J_nom = 10.8e8
    scenarios = {
        "Nominal J": J_nom,
        "20% Reduced J": J_nom * 0.8,
        "50% Reduced J": J_nom * 0.5
    }

    # Create two subplots: attitude response and controller output
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True
    )

    for label, J_val in scenarios.items():
        # Setup blocks
        sat = satellite(j=J_val, name=f"sat_{label}")
        sat_gen = sat.generator(dt, ic=[0.0, 0.0])
        pv = sat_gen.send(None)

        ctrl = contr(name="Controller")
        ctrl_gen = ctrl.generator(dt, ic=0.0)
        ctrl_gen.send(None)

        time_vector = np.arange(0, T, dt)

        # Step input
        u = np.zeros_like(time_vector)
        u[time_vector >= 0.0] = sp_deg

        cv = 0.0

        # Simulation loop
        for idx, t in enumerate(time_vector):
            sp = u[idx]

            cv = ctrl_gen.send((t, sp, pv, cv))
            pv = sat_gen.send((t, cv))

        # Extract logs
        t_log, sp_log, pv_log, cv_log = np.asarray(ctrl._log).transpose()

        # Attitude response plot
        ax1.plot(
            t_log,
            pv_log,
            label=f"{label} ($J={J_val:.2e}$)"
        )

        # Controller output plot
        ax2.plot(
            t_log,
            cv_log,
            label=f"{label} ($J={J_val:.2e}$)"
        )

    # Desired attitude reference
    ax1.axhline(
        y=sp_deg,
        color='k',
        linestyle='--',
        label=f'Desired Attitude ({sp_deg}°)'
    )

    # Top subplot formatting
    ax1.set_title('Satellite Attitude Control Step Response Comparison')
    ax1.set_ylabel(r'Attitude $\theta$ (degrees)')
    ax1.grid(True)
    ax1.legend()

    # Bottom subplot formatting
    ax2.set_title('Controller Output')
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Control Signal (cv)')
    ax2.grid(True)
    ax2.legend()

    ax1.set_xlim(0, T)

    plt.tight_layout()
    plt.show()


# Run the simulation
simulate_comparison(dt=0.1, T=100.0, sp_deg=10.0)
