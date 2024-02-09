# # Diffusion of a Gaussian function
from tucker_mods import *
extension_to_delete = "png"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)
extension_to_delete = "xdmf"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)
extension_to_delete = "h5"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)

import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

# import dolfinx
from dolfinx import fem, mesh, io #, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define temporal parameters
t = 0  # Start time
Tf = 1.0  # Final time
num_steps = 50
dt = Tf / num_steps  # time step size

# Define mesh
nx, ny = 50, 50
# domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])],
#                                [nx, ny], mesh.CellType.triangle)
# Height = 1e-4 #[m]
# Width = 1e-5 #[m]
# Length = 1e-5 #[m]
# Ln, Wn, Hn = 8, 8, 20
Length, Width, Height = 2, 2, 2
Ln, Wn, Hn = 8, 8, 8
domain = mesh.create_box( MPI.COMM_WORLD , np.array ([[0.0 ,0.0 ,0.0] ,[ Length , Width , \
    Height ]]) , [Ln, Wn, Hn] , cell_type = mesh.CellType.tetrahedron )

# 
# Create initial condition
def initial_condition(x, a=5):
    # return np.exp(-a * (x[0]**2 + x[1]**2))
    return 0 + x[0]*0 + x[1]*0

V = fem.FunctionSpace(domain, ("Lagrange", 1))
u_n = fem.Function(V)
u_n.name = "u_n"
print_variable(V)
print_variable(u_n)
u_n.interpolate(initial_condition)

# Create boundary condition
fdim = domain.topology.dim - 1

bottom_facet = mesh.locate_entities(domain, fdim, lambda x:  np.isclose(x[0], 0)) #-2)) # changed to 3D geometry 
bc = fem.dirichletbc(PETSc.ScalarType(10), fem.locate_dofs_topological(V, fdim, bottom_facet), V)


# ## Time-dependent output
# To visualize the solution in an external program such as Paraview, we create a an `XDMFFile` which we can store multiple solutions in. The main advantage with an XDMFFile, is that we only need to store the mesh once, and can append multiple solutions to the same grid, reducing the storage space.
# The first argument to the XDMFFile is which communicator should be used to store the data. As we would like one output, independent of the number of processors, we use the `COMM_WORLD`. The second argument is the file name of the output file, while the third argument is the state of the file,
# this could be read (`"r"`), write (`"w"`) or append (`"a"`).

# +
xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
print('writing initial solution to file...')
xdmf.write_function(uh, t)
# -

# ## Variational problem and solver
# As in the previous example, we prepare objects for time dependent problems, such that we do not have to recreate data-structures.

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

# ## Preparing linear algebra structures for time dependent problems
# We note that even if `u_n` is time dependent, we will reuse the same function for `f` and `u_n` at every time step. We therefore call `dolfinx.fem.form` to generate assembly kernels for the matrix and vector.

bilinear_form = fem.form(a)
linear_form = fem.form(L)

# We observe that the left hand side of the system, the matrix $A$ does not change from one time step to another, thus we only need to assemble it once. However, the right hand side, which is dependent on the previous time step `u_n`, we have to assemble it every time step. Therefore, we only create a vector `b` based on `L`, which we will reuse at every time step.

A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

# ## Using petsc4py to create a linear solver
# As we have already assembled `a` into the matrix `A`, we can no longer use the `dolfinx.fem.petsc.LinearProblem` class to solve the problem. Therefore, we create a linear algebra solver using PETSc, and assign the matrix `A` to the solver, and choose the solution strategy.
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# -
# ## Updating the solution and right hand side per time step
# To be able to solve the variation problem at each time step, we have to assemble the right hand side and apply the boundary condition before calling
# `solver.solve(b, uh.vector)`. We start by resetting the values in `b` as we are reusing the vector at every time step.
# The next step is to assemble the vector, calling `dolfinx.fem.petsc.assemble_vector(b, L)` which means that we are assemble the linear for `L(v)` into the vector `b`. Note that we do not supply the boundary conditions for assembly, as opposed to the left hand side.
# This is because we want to use lifting to apply the boundary condition, which preserves symmetry of the matrix $A$ if the bilinear form $a(u,v)=a(v,u)$ without Dirichlet boundary conditions.
# When we have applied the boundary condition, we can solve the linear system and update values that are potentially shared between processors.
# Finally, before moving to the next time step, we update the solution at the previous time step to the solution at this time step.

print('num of time steps = ',str(num_steps))
print('dt = ',str(dt))
for i in range(num_steps):
    t += dt
    # print('i = ',str(i),'   t = ',str(t))

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    if( (t<dt*1.5) or (i==int(num_steps/2))or (t>(Tf-dt*0.5))):
        # print('splitting, interpolating, and writing u and t ')
        print('writing solution to file...')
        xdmf.write_function(uh, t)

xdmf.close()