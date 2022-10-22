# OpenCLGameOfLifeViewable
This was a project focused on achieving maximum concurrency and parallelization through developing and scheduling the execution of GPU kernels using OpenCL.
## Multi-Species Conway's game of life rules
The rules of the original game of life are as follows: If the cell is alive, then it stays alive if it has either 2 or 3 live neighbors. 
If the cell is dead, then it springs to life only in the case that it has 3 live neighbors.
Cells in this case are represented by pixels. The only key difference here is that instead of cells representing simply a binary value, alive or dead, 
while they are alive they are also part of a specific species of alive cells. So now, for a cell to stay alive, it needs 2 or 3 alive neighbors of the same
species, and for a dead cell to come back to life, it needs 3 live neighbors of the same species, and obviously will come alive as that species. Each of 
the different species were represented by different colors. 
## Program Structure Overview
All logic regarding the translation of a current screen state to a future one given the rules of the game were encapsulated completely within a GPU kernel.
The state of the screen was represented by a one dimensional array, meaning that the translation of the one dimensional array into a 2D screen render required
some, albeit minimal, arithmetic gymnastics. The array had accompanying it a buffer array of the same size. During every iteration one array was being
used to compute the next screen, while the other acted as a buffer storing the computed future state of the screen. The one being used to compute the future
screen was used by OpenGL to render to the screen. The two arrays would alternate acting as a buffer and as the representation of the current state of the screen.
All scheduling was done using simple logic within the main program file. 
