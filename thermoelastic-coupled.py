#!/usr/bin/env python
# coding: utf-8
# https://comet-fenics.readthedocs.io/en/latest/demo/thermoelasticity/thermoelasticity_transient.html

# # Thermo-elastic evolution problem (full coupling)
import dolfinx
import ufl
import numpy as np
# from mshr import *
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from tucker_mods import *
extension_to_delete = "png"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)
extension_to_delete = "xdmf"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)
extension_to_delete = "h5"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)

import time 
start_time = time.time()

## Time parametrization
t = 0 # Start time
Tf = 5 # End time
num_steps = 1000 # Number of time steps
factor = 100
Tf = Tf/factor
num_steps = int(num_steps/factor)
dt = (Tf -t )/ num_steps # Time step size
print('running until total time = ',str(Tf))
print('with ',str(num_steps),' total time steps of ',str(dt),' seconds each')

L = 1.
R = 0.1
N = 50  # mesh density

# domain = mshr.Rectangle(dolfinx.mesh.Point(0., 0.), dolfinx.mesh.Point(L, L)) - mshr.Circle(dolfinx.mesh.Point(0., 0.), R, 100)
# mesh = mshr.generate_mesh(domain, N)
## Create the domain / mesh
# Height = 1e-4 #[m]
# Width = 1e-5 #[m]
# Length = 1e-5 #[m]
# Ln, Wn, Hn = 8, 8, 20
Length, Width, Height = 2, 2, 2
Ln, Wn, Hn = 8, 8, 8
domain = dolfinx.mesh.create_box( MPI.COMM_WORLD , np.array ([[0.0 ,0.0 ,0.0] ,[ Length , Width , \
    Height ]]) , [Ln, Wn, Hn] , cell_type = dolfinx.mesh.CellType.tetrahedron )
# domain = dolfinx.mesh.create_rectangle( MPI.COMM_WORLD , np.array ([[0.0 ,0.0 ] ,[ Length , Width , \
#      ]]) , [8, 20] , cell_type = dolfinx.mesh.CellType.triangle )

T0 = dolfinx.fem.Constant(domain , 293.)
DThole = dolfinx.fem.Constant(domain , 10.)
E = 70e3
nu = 0.3
lmda_value = E*nu/((1+nu)*(1-2*nu))
lmbda = dolfinx.fem.Constant(domain , lmda_value)
mu_value = E/2/(1+nu)
mu = dolfinx.fem.Constant(domain , mu_value)
rho = dolfinx.fem.Constant(domain , 2700.)     # density
alpha = 2.31e-5  # thermal expansion coefficient
kappa = dolfinx.fem.Constant(domain , alpha*(2*mu_value + 3*lmda_value))
print('kappa = ',str(kappa.value))
kappa = kappa / 100
cV = dolfinx.fem.Constant(domain , 910e-6)*rho  # specific heat per unit volume at constant strain
k = dolfinx.fem.Constant(domain , 237e-6)  # thermal conductivity


Vue = ufl.VectorElement('CG', domain.ufl_cell(), 2) # displacement finite element
Vte = ufl.FiniteElement('CG', domain.ufl_cell(), 1) # temperature finite element
V = dolfinx.fem.FunctionSpace(domain, ufl.MixedElement([Vue, Vte]))
MS = V

fspace_interp_u = dolfinx.fem.FunctionSpace(domain, ufl.VectorElement("CG", domain.ufl_cell(), 1)) # 1st-order version of u 
fspace_interp_t = dolfinx.fem.FunctionSpace(domain, ufl.FiniteElement("CG", domain.ufl_cell(), 1)) # just to match syntax of _u 
__u_interpolated = dolfinx.fem.Function(fspace_interp_u)
__t_interpolated = dolfinx.fem.Function(fspace_interp_t)

U = dolfinx.fem.Function(V) 
Uold = dolfinx.fem.Function(V)

# FOR UNKOWN REASONS, INIITAL CONDITIONS WORK FOR DISPLACEMENT, BUT NOT TEMPERATURE: 
temp_init = 100
def init_cond_u(x): 
    return [0 + x[0]*0 , 0 + x[1]*0 , 0 + x[2]*0]
U.sub(0).interpolate(init_cond_u)
Uold.sub(0).interpolate(init_cond_u)
def init_cond_temp(x): 
    return [temp_init + x[0]*0.01]
U.sub(1).interpolate(init_cond_temp)
Uold.sub(1).interpolate(init_cond_temp)

(uold, Thetaold) = ufl.split(Uold)


# FOR UNKOWN REASONS, BOUNDARY CONDITIONS WORK FOR DISPLACEMENT, BUT NOT TEMPERATURE: 
fdim = domain.topology.dim - 1
bottom_facet = dolfinx.mesh.locate_entities(domain, fdim, lambda x:  np.isclose(x[2], 0))
# bc1vals = dolfinx.fem.Constant(domain,[0,0,0])
bc1vals = np.array([0], dtype=dolfinx.default_scalar_type)
bc1 = dolfinx.fem.dirichletbc(bc1vals, dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, bottom_facet), V.sub(0))
bc2 = dolfinx.fem.dirichletbc(ScalarType(10), dolfinx.fem.locate_dofs_topological(V.sub(1), fdim, bottom_facet), V.sub(1))

# I tried various things to make temperature bcs work: 
# bc2vals = np.array([10], dtype=dolfinx.default_scalar_type)
# bc2vals = dolfinx.fem.Constant(domain,10)
# bc2 = dolfinx.fem.dirichletbc(DThole.value, dolfinx.fem.locate_dofs_topological(V.sub(1), fdim, bottom_facet), V.sub(1))

def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma(v, Theta):
    return (lmbda*ufl.tr(eps(v)) - kappa*Theta)*ufl.Identity(3) + 2*mu*eps(v)
    # return (lmbda*ufl.tr(eps(v)) - kappa*Theta)*len(v) + 2*mu*eps(v)


U_ = ufl.TestFunction(V)
(u_test, temp_test) = ufl.split(U_)
dU = ufl.TrialFunction(V)
(u_trial, temp_trial) = ufl.split(dU)

# adding a load: 
T = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type((0, 0, 1000)))
ds = ufl.Measure("ds", domain=domain)
# a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
# L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

mech_form = ufl.inner(sigma(u_trial, temp_trial), eps(u_test))*ufl.dx + ufl.dot(T, u_test) * ds
therm_form = (cV*(temp_trial-Thetaold)/dt*temp_test +
              kappa*T0*ufl.tr(eps(u_trial-uold))/dt*temp_test +
              ufl.dot(k*ufl.grad(temp_trial), ufl.grad(temp_test)))*ufl.dx
form = mech_form + therm_form


# U = dolfinx.fem.Function(V)
    # in "poro.py", X0 and Xn are created w/ "dolfinx.fem.function" (as opposed to ufl.function), 
    # and they are what is solved for by the NewtonSolver

#  Create an output xdmf file to store the values --------------- from clips 
# xdmf = XDMFFile( mesh.comm , "./terzaghi.xdmf", "w", encoding = dolfinx.io.XDMFFile.Encoding.ASCII)
xdmf_displacement = dolfinx.io.XDMFFile( domain.comm , "./displacement.xdmf", "w", encoding = dolfinx.io.XDMFFile.Encoding.HDF5)
xdmf_temp = dolfinx.io.XDMFFile( domain.comm , "./temperature.xdmf", "w", encoding = dolfinx.io.XDMFFile.Encoding.HDF5)
xdmf_displacement.write_mesh( domain )
xdmf_temp.write_mesh( domain )

# testing if initialization happens
__u , __t = U.split()
__u_interpolated.interpolate(__u)
__t_interpolated.interpolate(__t)
xdmf_displacement.write_function( __u_interpolated ,t)
xdmf_temp.write_function( __t_interpolated ,t)

# problem = LinearProblem(a, L, [bc0, bc1]
from dolfinx.fem.petsc import LinearProblem
# problem = LinearProblem(dolfinx.fem.form(ufl.lhs(form)) , dolfinx.fem.form(ufl.rhs(form)), [bc1, bc2])
problem = LinearProblem((ufl.lhs(form)) , (ufl.rhs(form)), [bc1, bc2])

i = 0
for n in range ( num_steps ):
    t += dt
    i +=1
    print("Increment " + str(i+1))
    
    U = problem.solve()
    
    Uold.x.array[:] = U.x.array[:]
    
    if( (t<dt*1.5) or (t>(Tf-dt))):
        print('splitting, interpolating, and writing u and t ')
        # __u , __t = ufl.split(U)
        __u , __t = U.split()
        __u_interpolated.interpolate(__u)
        __t_interpolated.interpolate(__t)
        xdmf_displacement.write_function( __u_interpolated ,t)
        xdmf_temp.write_function( __t_interpolated ,t)

xdmf_displacement.close() 
xdmf_temp.close()
end_time = time.time()
# print('end_time = ',str(end_time))
total_time = end_time - start_time
print('total time: ',str(total_time))