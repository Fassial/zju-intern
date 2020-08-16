#PBS -l nodes=z390:ppn=1
#PBS -q workq
#PBS -N sub1
cd   $PBS_O_WORKDIR
echo $PBS_NODEFILE
module load anaconda3
source activate
python  main.py -n 1
python  main.py -n 2