import numpy as np
import sys
import subprocess
import time
import tclab
import threading

import matplotlib.pyplot as plt
from package_DBR import Process, SelectPath_RT, Delay_RT, FO_RT

#----------------------------------Laboratoire 2----------------------------------------------------------------------#


def LL_RT(MV, Kp, TLead, TLag, Ts, PV, PVInit=0, method='EBD'):
    """
    The function "LL_RT" DOES NOT need to be included in a "for or while loop": this block is for offline use.

    :MV: input vector (manipulated variable)
    :Kp: process gain
    :TLead: lead time constant [s]
    :TLag: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector (process variable) — the simulated response is appended to this list
    :PVInit: initial value for the process variable if PV is empty (optional, default = 0)
    :method: discretisation method (optional, default = 'EBD')
        EBD: Euler Backward Difference
        EFD: Euler Forward Difference
        TRAP: Trapezoidal (Tustin) method
        others: basic approximation without lead compensation

    :return: simulated LL output vector (appended to PV)
    """

    if TLag != 0:
        K = Ts / TLag
        if len(PV) == 0:
            PV.append(PVInit)
        else:
            if method == 'EBD':
                PV.append(
                    (1 / (1 + K)) * PV[-1]
                    + (K * Kp / (1 + K)) * ((1 + TLead / Ts) * MV[-1] - (TLead / Ts) * MV[-2])
                )
            elif method == 'EFD':
                PV.append(
                    (1 - K) * PV[-1]
                    + K * Kp * ((TLead / Ts) * MV[-1] + (1 - TLead / Ts) * MV[-2])
                )
            elif method == 'TRAP':
                a = (2 * TLag - Ts) / (2 * TLag + Ts)
                b = (2 * Kp * TLead) / (2 * TLag + Ts)
                PV.append(
                    a * PV[-1] + b * (MV[-1] + MV[-2])
                )
            else:
                PV.append((1 / (1 + K)) * PV[-1] + (K * Kp / (1 + K)) * MV[-1])
    else:
        PV.append(Kp * MV[-1])


#--------------------------------------------------------------------------------------------------------------------


def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, methodI='EBD', methodD='EBD') :
    """
    The function "PID_RT" DOES NOT need to be included in a "for or while loop": this block is for offline use.
    
    :SP: setpoint vector
    :PV: process variable vector (measured output)
    :Man: list of boolean flags indicating manual mode at each step
    :MVMan: manual mode control signal vector
    :MVFF: feedforward signal vector
    :Kc: proportional gain
    :Ti: integral time constant [s]
    :Td: derivative time constant [s]
    :alpha: derivative filter coefficient (Tfd = alpha * Td)
    :Ts: sampling period [s]
    :MVMin: minimum limit for control signal (saturation)
    :MVMax: maximum limit for control signal (saturation)
    :MV: output vector (final manipulated variable, result is appended)
    :MVP: proportional term vector
    :MVI: integral term vector
    :MVD: derivative term vector
    :E: error vector (SP - PV)
    :ManFF: (optional, default = False) enable manual feedforward compensation during integrator reset
    :PVInit: (optional, default = 0) initial process variable value if PV is empty
    :methodI: (optional, default = 'EBD') discretization method for integral part
        EBD: Euler Backward Difference
        TRAP: Trapezoidal rule
    :methodD: (optional, default = 'EBD') discretization method for derivative part
        EBD: Euler Backward Difference
        TRAP: Trapezoidal rule

    :return: simulated PID output vector (appended to MV)

    The function "PID_RT" returns the simulated PID controller output vector from the input parameters and vectors.
    """

    
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else: 
        E.append(SP[-1] - PV[-1])
    
    MVP.append(Kc*E[-1])
    
    if len(MVI) == 0:
        MVI.append((Kc*Ts/Ti)*E[-1]) 
    else:
        if methodI == 'TRAP':
            MVI.append(MVI[-1] + (Kc*Ts/Ti)*(E[-1] + E[-2])/2)
        else: 
            MVI.append(MVI[-1] + (Kc*Ts/Ti)*E[-1])
            
    Tfd = alpha*Td
    if len(MVD) == 0:
        MVD.append((Kc*Td/(Tfd + Ts))*E[-1]) # E[-2] = 0
    else:
        if methodD == 'TRAP':
            MVD.append((Tfd - Ts/2)/(Tfd + Ts/2)*MVD[-1] + (Kc*Td/(Tfd + Ts/2))*(E[-1] - E[-2]))
        else: # EBD
            MVD.append(Tfd/(Tfd + Ts)*MVD[-1] + (Kc*Td/(Tfd + Ts))*(E[-1] - E[-2]))

    # Integrator Reset
    if Man[-1] == True:
        if ManFF == True:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]
        else :
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]

    # Saturation Integrator Reset
    if MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1] > MVMax :
        MVI[-1] = MVMax - MVP[-1] - MVD[-1] - MVFF[-1]
    elif MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1] < MVMin :
        MVI[-1] = MVMin - MVP[-1] - MVD[-1] - MVFF[-1]

    # Resulting MV
    MV.append(MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1])
    
    



#----------------------------------#

def IMC_Tuning(Kp,theta,T,T2=0,gamma=0.2,order=1):
    """
    The function "IMC_Tuning" DOES NOT need to be included in a "for or while loop": this block is for offline use.

    :Kp: process gain
    :theta: process dead time [s]
    :T: main time constant [s]
    :T2: second time constant (optional, default = 0) [s]
    :gamma: closed-loop aggressiveness factor (Tclp = gamma * T), default = 0.2
    :order: process order (1 for FOPDT, 2 for SOPDT), default = 1

    :return: PID tuning parameters (Kc, Ti, Td)

    The function "IMC_Tuning" returns the simulated PID tuning parameters from the input parameters using the Internal Model Control (IMC) method.
    """
    Tclp = gamma*T
   
    if order == 1: 
        Kc = ((T+(theta/2))/(Tclp+(theta/2))) /Kp
        Ti = T+(theta/2)
        Td = T*theta/((2*T)+theta)
        print(Kc, Ti, Td)
        return Kc, Ti, Td
        
    else:
        Kc = ((T+T2)/(Tclp+theta)) /Kp
        Ti = T+T2
        Td = (T*T2)/(T+T2)
        print(Kc, Ti, Td)
        return Kc, Ti, Td
    


#--------------------------------------------------------------------



class Controller:
    
    def __init__(self, parameters):
        """
        The constructor "__init__" initializes a Controller object with given parameters.

        :parameters: dictionary containing controller parameters
            - 'Kc': proportional gain (default = 1.0)
            - 'Ti': integral time constant [s] (default = 0.0)
            - 'Td': derivative time constant [s] (default = 0.0)
            - 'Tfd': derivative filter time constant [s] (default = 0.0)

        This method sets the controller parameters, assigning default values if any are missing.
        """
        
        self.parameters = parameters
        self.parameters['Kc'] = parameters['Kc'] if 'Kc' in parameters else 1.0
        self.parameters['Ti'] = parameters['Ti'] if 'Ti' in parameters else 0.0
        self.parameters['Td'] = parameters['Td'] if 'Td' in parameters else 0.0
        self.parameters['Tfd'] = self.parameters['Tfd'] if 'Tfd' in parameters else 0.0

#----------------------------------#





def margin(P: Process, C: Controller, omega, show=True) :
    """
    The function "margin" DOES NOT need to be included in a "for or while loop": this block is for offline use.

    :P: instance of the Process class, containing process parameters
    :C: instance of the Controller class, containing controller parameters
    :omega: array of frequency values [rad/s]
    :show: (optional, default = True) if True, displays the Bode plot with gain and phase margins

    :return: if show=False, returns a tuple (Ls, GM, PM)
        - Ls: loop transfer function evaluated over omega
        - GM: gain margin [dB]
        - PM: phase margin [°]

    The function "margin" returns the gain margin, phase margin, and optionally the loop transfer function from the given process and controller parameters.
    """
    
    s = 1j * omega
    
    # Process transfer function
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    PLag1 = 1/(P.parameters['Tlag1']*s + 1)
    PLag2 = 1/(P.parameters['Tlag2']*s + 1)
    PLead1 = P.parameters['Tlead1']*s + 1
    PLead2 = P.parameters['Tlead2']*s + 1
    
    Ps = np.multiply(Ptheta,PGain)
    Ps = np.multiply(Ps,PLag1)
    Ps = np.multiply(Ps,PLag2)
    Ps = np.multiply(Ps,PLead1)
    Ps = np.multiply(Ps,PLead2)
    
    # Controller transfer function
    Cs = C.parameters['Kc'] * (1 + 1/(C.parameters['Ti']*s) + C.parameters['Td']*s/(1 + C.parameters['Tfd']*s))
    
    # Loop transfer function
    Ls = np.multiply(Ps,Cs)

    # Gain margin
    GM = 0
    ultimate_freq = 0
    phase_crossing_idx = np.argmin(np.abs(np.angle(Ls, deg=True) - -180)) # Find the nearest point with an angle of -180°
    if phase_crossing_idx > 0:
        ultimate_freq = omega[phase_crossing_idx]
        GM = 20*np.log10(1 / np.abs(Ls[phase_crossing_idx]))
        print(f"Gain margin GM = {GM:.5f} dB at {ultimate_freq:.2f} rad/s")
    else:
        print(">> Index for which arg(Ls) = -180° not found")
    
    # Phase margin
    PM = 0
    crossover_freq = 0
    gain_crossing_idx = np.argmin(np.abs(np.abs(Ls) - 1)) # Find the nearest point with a gain of 1
    if gain_crossing_idx > 0:
        crossover_freq = omega[gain_crossing_idx]
        PM = 180 + np.angle(Ls[gain_crossing_idx], deg=True)
        print(f"Phase margin PM = {PM:.5f}° at {crossover_freq:.2f} rad/s")
    else:
        print(">> Index for which |Ls| = 1 not found")
        
    
    if show == True:
        fig, (ax_gain, ax_phase) = plt.subplots(2,1)
        fig.set_figheight(10)
        fig.set_figwidth(18)
        
        # Gain Bode plot
        ax_gain.semilogx(omega,20*np.log10(np.abs(Ls)), label=r'$L(s)=P(s)C(s)$')
        ax_gain.semilogx(omega,20*np.log10(np.abs(Ps)), ':',label=r'$P(s)$')
        ax_gain.semilogx(omega,20*np.log10(np.abs(Cs)), ':',label=r'$C(s)$')
        ax_gain.axhline(0, color='black', linestyle='--')
        ax_gain.vlines(ultimate_freq, -GM, 0, color='red')
        gain_min = np.min(20*np.log10(np.abs(Ls)/5))
        gain_max = np.max(20*np.log10(np.abs(Ls)*5))
        ax_gain.set_xlim([np.min(omega), np.max(omega)])
        ax_gain.set_ylim([gain_min, gain_max])
        ax_gain.set_ylabel('Amplitude' + '\n$|L(s)|$ [dB]')
        ax_gain.set_title('Diagramme de Bode')
        ax_gain.legend(loc='best')
        ax_gain.grid()
        
        # Phase Bode plot
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ls)),label=r'$L(s)=P(s)C(s)$')
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ps)), ':',label=r'$P(s)$')
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Cs)), ':',label=r'$C(s)$')
        ax_phase.axhline(-180, color='black', linestyle='--')
        ax_phase.vlines(crossover_freq, -180, -180 + PM, color='blue')
        ax_phase.set_xlim([np.min(omega), np.max(omega)])
        ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ls))) - 10
        ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ls))) + 10
        ax_phase.set_ylim([np.max([ph_min, -200]), ph_max])
        ax_phase.set_xlabel(r'Frequency $\omega$ [rad/s]')
        ax_phase.set_ylabel('Phase' + '\n' + r'$\angle L(s)$ [°]')
        ax_phase.legend(loc='best')
        ax_phase.grid()

    else:
        return Ls, GM, PM
    
    
#---------------------------------------------------------------------------------------------------
