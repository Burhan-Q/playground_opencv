'''
Title: Thermal conduction in aluminum
Author: Burhan Qadodoumi
Date: 2023-02-26

Requirement: numpy, scipy, plotly, pandas
'''
import numpy as np
from scipy.special import kn
import plotly.graph_objects as go

def besselk(N:int,array):
    """Function for calculating modified Bessel function of the second kind of integer order `N`"""
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kn.html
    res = kn(N,array)
    return res

# defining constants
T0 = 300 # K
q_prime = 1e7 # W/m
V = np.array([0.01, 0.02, 0.05]) # m/s; shape = [3,]
density = 2700 # kg/m^3
Cp = 903 # J/(kg*K)
AA = 97.1e-6 # m^2/s
k = 142 # W/m*K

# computing r and theta
r = np.arange(0.01, 0.1 + 0.01, 0.01)
theta = np.arange(0, -np.pi - np.pi / 50, -np.pi / 50)
cos_theta = np.cos(theta)
sine_theta = np.sin(theta)
r_cos_theta = np.array([v * cos_theta for v in r])
r_sine_theta = np.array([v * sine_theta for v in r])

# computing x and y
x,y = np.meshgrid(r_cos_theta,r_sine_theta)  # shape (510,510) for each x and y
R = np.sqrt(x ** 2 + y ** 2) # shape (510,510)

# plotting
for i in range(len(V)):
    Knaught = np.array(besselk(0, (V[i] * R / 2 / AA)))
    TT = (q_prime / 2 / np.pi / k) * np.exp(-V[i] * x / 2 / AA) * Knaught

    fig = go.Figure(data=[go.Surface(z=TT, x=x, y=y)])
    fig.update_layout(
        title=f"Temperature distribution with V={V[i]}m/s q={q_prime}W/m",
        autosize=False,
        width=800,
        height=800,
        )
    # NOTE: uncomment below to generate ONLY wire-surface plot (better performance)
    # fig.update_traces(
    #     hidesurface=True,
    #     contours_y_show=True,
    #     contours_x_show=True, 
    # )
    fig.update_traces(
        cmin=0, # color scale min-limt
        cmax=6000 # color scale max-limt
    )
    fig.update_scenes(zaxis_range=[0,6000]) # truncate plot from 0 upto 6000
    fig.show()
    # fig.to_html('path/to/filename') # save HTML plot file
    _ = input("Press any key to generate next plot") # pause and wait for input before generating next plot


# Alternative method (scatter plot)
import plotly.express as px
import pandas as pd

x = r_cos_theta.reshape(r_cos_theta.size,1) # create column from all data
y = r_sine_theta.reshape(r_sine_theta.size,1) # create column from all data
R = np.sqrt(x ** 2 + y ** 2) # shape (510,1)

for i in range(len(V)):
    # qq_i = np.array((density * Cp * V[i])) #calculate qq for each V[i]
    Knaught = np.array(besselk(0, V[i] * R / 2 / AA)) #calculate K0 for each V[i]
    TT = (q_prime / 2 / np.pi / k) * np.exp(-V[i] * x / 2 / AA) * Knaught

    alldata = np.column_stack([x,y,TT]) # stacks data together, resulting shape is 510,3
    df = pd.DataFrame(alldata,columns=['x','y','TT'])
    sctr = px.scatter_3d(df,x='x',y='y',z=np.zeros((510)), color='TT')
    sctr.show()
    # sctr.to_html('path/to/filename') # save HTML plot file
    _ = input("Press any key to generate next plot")


# Reference
# http://aml.engineering.columbia.edu/ntm/level2/ch03/html/l2c03s05.html
# Example: For Aluminum
# T0=300 K, 
# q'=1E7 W/m, 
# V=0.01,0.02,0.05 m/s, 
# density=2700 kg/m^3; 
# Cp=903 J/(kg*K); 
# AA=97.1E-6 m^2/s;
# k = 142 # W/m*K

# MATLAB code
# r = 0.01:0.01:0.1; r=r'; 
# theta = 0:-pi/50:-pi;
# x=r*cos(theta);
# y=r*sin(theta);
# R=sqrt(x.^2+y.^2);
# K0=besselk(0,V*R/2/AA);
# TT=(qq/2/pi/k)*exp(-V*x/2/AA).*K0;
# mesh(x,y,TT); title('Temperature distribution with V=0.01m/s, q''=1e7W/m');
# xlabel('x (m)'); ylabel('y (m)'); zlabel('Temperature K ');

# Plots illustrative temperature distributions [shown in figure 3.22] for different scanning speeds (V=0.01, 0.02, 0.05 m/s)
# y is the depth in the material
# x is the distance from the laser source
