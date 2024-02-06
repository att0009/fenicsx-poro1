Poroelasticity based on: 
https://github.com/Th0masLavigne/Dolfinx_Porous_Media/tree/main
Thermoelasticy based on: 
https://comet-fenics.readthedocs.io/en/latest/demo/thermoelasticity/thermoelasticity_transient.html

I never got any of the “BoundingBoxTree, compute_collisions, compute_colliding_cells” stuff to work – I always got error messages.  And thus, I never got the any of the matplotlib plots to generate.  

I did get nice xdmf/h5 files to generate.  They look nice in Paraview, but I did have to interpolate the functions, ie “__p_interpolated.interpolate(__p)” before writing them via
  “xdmf_pressure.write_function( __p_interpolated ,t)”, as opposed to writing the "__p" function directly.  
  This was not necessary if I used first order elements, but first-order elements led to pretty ugly plots (not smooth transitions between elements).  
  It is possible I could have used first order elements, but with way more elements, but for the # of elements given in the code by the author, a minimum of 2nd-order elements were necessary.  

Viewing with Paraview:  
-	Select “Open” (top left) 
-	Select the desired xdmf file – ***make sure the associated h5 file is also in the same folder***.  
-	Select “Xdmf3ReaderT”
-	Apply
-	In the bottom left box, under “Coloring” select “Rescale to data range over all timesteps”.  
-	Sources > Annotation > Annotate Time 
-	Apply 
-	File > Save Animation – make sure you have the correct folder selected to save the animation 
    ^ This will save a png screenshot of the plot at each timestep written to the xdmf file.  

