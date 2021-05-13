#CSC 336 Summer 2020 A3Q3 starter code

#Please ask if you aren't sure what any part of the code does.
#We will do a similar example on the Week 10 worksheet.

#some general imports
import numpy.linalg as LA
import time
from time import perf_counter
import datetime
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.dates as mdates #for plotting dates


#loads the data from the provided csv file
#the data is stored in data, which will be
#a numpy array - the first column is the number of recovered individuals
#              - the second column is the number of infected individuals
#the data starts on '2020/03/05' (March 5th) and is daily.
data = []
with open("ONdata.csv") as f:
    l = f.readline()
    #print(l.split(',')) #skip the column names from the csv
    l = f.readline()
    while '2020/03/05' not in l:
        l = f.readline()
    e = l.split(',')
    data.append([float(e[5])-float(e[12]),float(e[12])])
    for l in f:
        e = l.split(',')
        data.append([float(e[5])-float(e[12]),float(e[12])])
data = np.array(data)

#the 3 main parameters for the model, we'll use them as
#global variables, so you can refer to them anywhere
beta0 = 0.32545622
gamma = 0.09828734
w = 0.75895019

#simulation basic scenario (see the simulation function later)
base_ends = [28,35,49,70,94,132]
base_beta_factors = [1, 0.57939203, 0.46448341,
                             0.23328388, 0.30647815,
                             0.19737586]

#we'll also use beta as a global variable
beta = beta0

#We are assuming no birth or death rate, so N is a constant
N = 15e6

#assumed initial conditions
E = 0 #assumed to be zero
I = 18 #initial infected, based on data
S = N - E - I #everyone else is susceptible
R = 0
x0 = np.array([S,E,I,R])

#The right hand side function for the SEIR model ODE, x'(t) = F(x(t))
#Feel free to use this when implementing the 3 methods, but you don't have to.
def F(x):
    return np.array([-x[0]*(beta/N) * x[2],
                     (beta/N * x[0]*x[2] - w*x[1]),
                     (w*x[1] - gamma*x[2]),
                     x[2]*(gamma)
                    ])

def f2(x, prex, h):
    return x - prex - h*F(x)

def f3(x, prex, h):
    return x - prex - (h/2)*(F(x) + F(prex))

def F_jac(x):
    return np.array([[-(beta/N) * x[2], 0, -(beta/N)*x[0], 0],
                     [(beta/N) * x[2], -w, (beta/N)*x[0], 0],
                     [0, w, -gamma, 0],
                     [0, 0, gamma, 0]])

def jac2(x, h):
    return np.identity(4) - h*(F_jac(x))

def jac3(x, h):
    return np.identity(4) - (h/2)*(F_jac(x))
    

#your three numerical methods for performing a single time step
def method_I(x,h):
    return x + h*(F(x))
def method_II(x,h):
    fx = lambda x_new: f2(x_new,x,h)
    jacx = lambda x_new: jac2(x_new, h)
    xtmp,info,ier,msg = fsolve(fx,x,xtol=1e-12,fprime=jacx,
                               full_output=True)    
    return xtmp
def method_III(x,h):
    fx = lambda x_new: f3(x_new,x,h)
    jacx = lambda x_new: jac3(x_new,h)
    xtmp,info,ier,msg = fsolve(fx,x,xtol=1e-12,fprime=jacx,
                               full_output=True)
    return xtmp

METHODS = {'I' : method_I, 'II' : method_II, 'III' : method_III}

#take a step of length 1 using n smaller steps
#used in simulation (see below)
def step(x,n,method):
    #simulate step_length time unit using n small uniform length steps
    #and return the new state
    for i in range(n):
        x = method(x,1/n)
    return x

def ode_solver(x,start,end):
    from scipy.integrate import solve_ivp as ode
    fun = lambda t,x : F(x)
    sol = ode(fun,[start,end],x, t_eval=range(start,end+1),
              method='LSODA', rtol = 1e-8, atol = 1e-5)
    solution = []
    for y in sol.y.T[1:,:]:
        solution.append(y.T)
    return solution

#The main simulation code:
# The simulation starts at time 0 and goes up to time ends[-1],
# the state x(t) is returned at each time 0,1,...,ends[-1].
# Inputs:
# x = initial conditions
# n = integer number of steps of method used to advance one time step
# method = 1,2, or 3 to specify which method to use. If None,
#          the builtin ODE solver is used.
# ends = list of times to break the simulate
# beta_factors = list of factors to multiply beta0 by to
#                obtain beta. E.g. on the first segment of the simulation
#                from time t=0 up to ends[0], beta = beta0 * beta_factors[0].
def simulation(x=x0, n=1, method=None,
               ends=base_ends,
               beta_factors=base_beta_factors):
    cur_time = 0
    xs = [x]
    for i,end in enumerate(ends):
        global beta
        beta = beta0 * beta_factors[i]

        if method == None:
            xs.extend(ode_solver(xs[-1],cur_time,end))
            cur_time = end
        else:
            while cur_time < end:
                xs.append(step(xs[-1],n,METHODS[method]))
                cur_time += 1
    return np.array(xs)


#some helper code to plot simulation trajectories,
#feel free to modify as needed.
def plot_trajectories(xs=data,sty='--k',label = "data"):
    start_date = datetime.datetime.strptime("2020-03-05","%Y-%m-%d")
    dates = [start_date]
    while len(dates) < len(xs):
        dates.append(dates[-1] + datetime.timedelta(1))

    #code to get matplotlib to display dates on the x-axis
    ax = plt.gca()
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)

    plt.plot(dates,xs,sty,linewidth=1,label=label)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

#example code to plot trajectories of I and R for each method
def plot_all_methods():
    plot_trajectories() #plots data
    xs = simulation()

    plot_trajectories(xs[:,2:],'-.k','ode')

    xs = simulation(n=10,method = "II")
    plot_trajectories(xs[:,2:],'--r','II')

    xs = simulation(n=10, method='I')
    plot_trajectories(xs[:,2:],'--b','I')

    xs = simulation(n=10, method="III")
    plot_trajectories(xs[:,2:],'--g','III')

    plt.legend()

#code for plot from the handout
def make_handout_data_plot():
    plt.rcParams.update({'font.size': 16})
    plt.figure()

    plot_trajectories(xs = data[:,0],sty="-b")
    plot_trajectories(xs = data[:,1],sty='-r')

    plt.grid(True, which='both')
    plt.title("Ontario Covid-19 Data")
    plt.legend(["Recovered","Infected"])
    plt.ylabel("# of people in state")
    plt.tight_layout()
    plt.show()

    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(10000))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(2500))
    plt.ylim([0,37500])


#####
# Part (c)
# add your code for your experiment to check
# accuracy and efficiency of the three methods
# Note: you should be calling simulation and varying n and method
#####
def Q3c():
    true_sol = simulation()
    i = 0
    sty = ['*-r','*-b','*-k']
    tims = []
    errs = []
    for method in METHODS:
        errs.append([])
        tims.append([])
        for n in [1,2,4,8,16,32,64,128, 256]: #add more values
            s = time.perf_counter()
            sol = simulation(n=n,method=method)
            tims[-1].append(time.perf_counter() - s)
            err = LA.norm(true_sol[-1] - sol[-1])/LA.norm(true_sol[-1])
            errs[-1].append(err)
        plt.loglog(1/np.array([1,2,4,8,16,32,64,128, 256]),errs[-1],sty[i])
        
        print(method, "conv order", -np.log(errs[-1][-1]/errs[-1][-2])/np.log(2))
        i += 1
        
    plt.title("convergence plot (error vs h)")
    plt.show()
    
    plt.figure()
    plt.loglog(1/np.array([1,2,4,8,16,32,64,128, 256]),tims[-1],sty[-1])
    plt.loglog(1/np.array([1,2,4,8,16,32,64,128, 256]),tims[-2],sty[-2])
    plt.loglog(1/np.array([1,2,4,8,16,32,64,128, 256]),tims[-3],sty[-3])
    plt.title("runtime plot (time vs h)")
    
    plt.figure()
    
    plt.title("runtime vs error")
    plt.loglog(errs[-1],tims[-1],sty[-1])
    plt.loglog(errs[-2],tims[-2],sty[-2])
    plt.loglog(errs[-3],tims[-3],sty[-3])





#####
# Part (d)
#add your code for similar experiment, but for a fixed value of n,
#plot the error vs simulation time to see how the error changes
#over the course of the simulation
#####
def Q3d():
    sold = simulation()
    i = 0
    sty = ['*-r','*-b','*-k']
    tims = []
    errs = []
    n = 2
    runtime = np.linspace(1, 133, 133)
    for method in METHODS:
        errs.append([])
        
        
        
        for i in range(len(sold)): #add more values
            sol = simulation(n=n,method=method)
            err = LA.norm(sold[i] - sol[i])/LA.norm(sold[i])
            errs[-1].append(err)
    
    
    
    
    plt.figure()
    
    plt.title("runtime vs error")
    plt.plot(runtime, errs[-1],sty[-1])
    plt.plot(runtime, errs[-2],sty[-2])
    plt.plot(runtime, errs[-3],sty[-3])
    plt.legend(["method III","method II","method I"])


##########
# Part (e)
#setup your own experiment to investigate a scenario
#what exactly you do is up to you, but you will most likely
#want to pass in different values for ends and beta_factors
#for your calls to simulation
##########
def Q3e():
    xs = simulation(ends = base_ends + [200], beta_factors = base_beta_factors + [base_beta_factors[-1]])
    plt.subplot(2,1,1)
    plt.title("Recovered population with date")
    plot_trajectories(data[:,0],sty = "b-")
    plot_trajectories(xs[:,3])
    plt.subplot(2,1,2)
    plt.title("Inflected population with date")
    plot_trajectories(data[:,1],sty = "r-")
    plot_trajectories(xs[:,2])
    plt.legend(["data", "model"])
    plt.show()