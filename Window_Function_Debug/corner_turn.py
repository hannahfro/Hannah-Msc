

'''adrian's way:

√ Have the bash script take in a command line parameter and place it in the frequency parameter name of the python mpi script. 
	- ./run_stuff.sh 200 300 0.5
	- Within run_stuff.sh:
	- $1 is 200
	- $2 is 300
	- $3 is 0.5
	- command line argument
	- do node arthmetic to know how many nodes you need
	- Write a script that generates the number of nodes. 

√ generate_script.sh <params of sim>
	In generate_script.sh, it takes <params of sim> and creates full job submission script with <params of sim> and appropriate # of nodes
	run the output of this (./resultant_script.sh) 

In the bash scipt: cat the script itself then you get the script you ran in the slurm file so you know exactly what you ran. 


√ have the path up until the file name as the input of the python script.
- rerun one of the talk scenarios on CC and check new parallelization gives the same result. 
	- re-mem profile the corner turn
	- time it and check how the time scales and plot it and fit the time line. add 20 mins for safety 
	

VI vs Nano
'''