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
from dolfinx.fem.petsc import NonlinearProblem
import dolfinx.nls.petsc


from tucker_mods import *
extension_to_delete = "png"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)
extension_to_delete = "xdmf"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)
extension_to_delete = "h5"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)

start_time = check_clock()

L = 1.
R = 0.1
N = 50  # mesh density

# domain = mshr.Rectangle(dolfinx.mesh.Point(0., 0.), dolfinx.mesh.Point(L, L)) - mshr.Circle(dolfinx.mesh.Point(0., 0.), R, 100)
# mesh = mshr.generate_mesh(domain, N)
## Create the domain / mesh
# Length, Width, Height = 1e-5 , 1e-5 , 1e-4 #[m]
# Ln, Wn, Hn = 8, 8, 20
Length, Width, Height = 2, 2, 2
# Ln, Wn, Hn = 8, 8, 8
Ln, Wn, Hn = 10, 10, 10
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
Vu_space = dolfinx.fem.FunctionSpace(domain,Vue)
u_old = dolfinx.fem.Function(Vu_space) 
u_new = dolfinx.fem.Function(Vu_space) 

Vte = ufl.FiniteElement('CG', domain.ufl_cell(), 1) # temperature finite element
Vt_space = dolfinx.fem.FunctionSpace(domain,Vte)
temp_old = dolfinx.fem.Function(Vt_space) 
temp_new = dolfinx.fem.Function(Vt_space) 
# V = dolfinx.fem.FunctionSpace(domain, ufl.MixedElement([Vue, Vte]))
# MS = V

fspace_interp_u = dolfinx.fem.FunctionSpace(domain, ufl.VectorElement("CG", domain.ufl_cell(), 1)) # 1st-order version of u 
fspace_interp_t = dolfinx.fem.FunctionSpace(domain, ufl.FiniteElement("CG", domain.ufl_cell(), 1)) # just to match syntax of _u 
__u_interpolated = dolfinx.fem.Function(fspace_interp_u)
__t_interpolated = dolfinx.fem.Function(fspace_interp_t)

# U = dolfinx.fem.Function(V) 
# Uold = dolfinx.fem.Function(V)

# FOR UNKOWN REASONS, INIITAL CONDITIONS WORK FOR DISPLACEMENT, BUT NOT TEMPERATURE: 
temp_init = 100
def init_cond_u(x): 
    return [0 + x[0]*0 , 0 + x[1]*0 , 0 + x[2]*0]
u_old.interpolate(init_cond_u)
u_new.interpolate(init_cond_u)
def init_cond_temp(x): 
    return [temp_init + x[0]*0.01]
temp_old.interpolate(init_cond_temp)
temp_new.interpolate(init_cond_temp)


# FOR UNKOWN REASONS, BOUNDARY CONDITIONS WORK FOR DISPLACEMENT, BUT NOT TEMPERATURE: 
fdim = domain.topology.dim - 1
bottom_facet = dolfinx.mesh.locate_entities(domain, fdim, lambda x:  np.isclose(x[2], 0))
# bc1vals = dolfinx.fem.Constant(domain,[0,0,0])
bc1vals = np.array([0, 0, 0], dtype=dolfinx.default_scalar_type)
# bc1 = dolfinx.fem.dirichletbc(bc1vals, dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, bottom_facet), V.sub(0))
# bc2 = dolfinx.fem.dirichletbc(ScalarType(10), dolfinx.fem.locate_dofs_topological(V.sub(1), fdim, bottom_facet), V.sub(1))
bc1 = dolfinx.fem.dirichletbc(bc1vals, dolfinx.fem.locate_dofs_topological(Vu_space, fdim, bottom_facet), Vu_space)
bc2 = dolfinx.fem.dirichletbc(ScalarType(10), dolfinx.fem.locate_dofs_topological(Vt_space, fdim, bottom_facet), Vt_space)



# we make the thermal problem first: ###########################################################################################
temp_test = ufl.TestFunction(Vt_space)
# temp_trial = ufl.TrialFunction(Vt_space)
    # test + trial function, like Functions, are defined based on the FunctionSpace
# therm_form = (cV*(temp_new-temp_old)/dt*temp_test + \
#               ufl.dot(k*ufl.grad(temp_new), ufl.grad(temp_test)))*ufl.dx 
#             # + kappa*T0*ufl.tr(eps(u_trial-uold))/dt*temp_test # leave this out b/c it involves u_trial
therm_form = (ufl.dot(k*ufl.grad(temp_new), ufl.grad(temp_test)))*ufl.dx 

# temp_new already defined
dt_new = ufl.TrialFunction(Vt_space)
J = ufl.derivative(therm_form, temp_new, dt_new)
problem_therm = NonlinearProblem(therm_form, temp_new, bcs = [bc2], J=J)
solver_therm = dolfinx.nls.petsc.NewtonSolver( domain.comm , problem_therm )
solver_therm.convergence_criterion = "incremental"
num_its , converged = solver_therm.solve( temp_new )

      

# then we make the mechanical problem ###########################################################################################
def eps(v):
    return ufl.sym(ufl.grad(v))

def sigma(v, Theta):
    return (lmbda*ufl.tr(eps(v)) - kappa*Theta)*ufl.Identity(3) + 2*mu*eps(v)
    # return (lmbda*ufl.tr(eps(v)) - kappa*Theta)*len(v) + 2*mu*eps(v)

u_test = ufl.TestFunction(Vu_space) 
# u_trial = ufl.TrialFunction(Vu_space)

# adding a load: 
T = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type((0, 0, 1000)))
ds = ufl.Measure("ds", domain=domain)
# a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
# L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
mech_form = ufl.inner(sigma(u_new, temp_new), eps(u_test))*ufl.dx + ufl.dot(T, u_test) * ds
    # I replace temp_trial w/ temp_new... let's see if that changes anything 
    
# u_new already defined
du_new = ufl.TrialFunction(Vu_space)
J = ufl.derivative(mech_form, u_new, du_new)
problem_mech = NonlinearProblem(mech_form, u_new, bcs = [bc1], J=J)
solver_mech = dolfinx.nls.petsc.NewtonSolver( domain.comm , problem_mech )
solver_mech.convergence_criterion = "incremental"
num_its , converged = solver_mech.solve( u_new )



#  Create an output xdmf file to store the values --------------- from clips 
# xdmf = XDMFFile( mesh.comm , "./terzaghi.xdmf", "w", encoding = dolfinx.io.XDMFFile.Encoding.ASCII)
xdmf_displacement = dolfinx.io.XDMFFile( domain.comm , "./results/displacement.xdmf", "w", encoding = dolfinx.io.XDMFFile.Encoding.HDF5)
xdmf_temp = dolfinx.io.XDMFFile( domain.comm , "./results/temperature.xdmf", "w", encoding = dolfinx.io.XDMFFile.Encoding.HDF5)
xdmf_displacement.write_mesh( domain )
xdmf_temp.write_mesh( domain )

# testing if initialization happens
# __u , __t = U.split()
__u_interpolated.interpolate(u_new)
__t_interpolated.interpolate(temp_new)
xdmf_displacement.write_function( __u_interpolated) # ,t) # steady-state
xdmf_temp.write_function( __t_interpolated) # ,t)


xdmf_displacement.close() 
xdmf_temp.close()
end_time = check_clock()
# print('end_time = ',str(end_time))
total_time = end_time - start_time
print('total time: ',str(total_time))