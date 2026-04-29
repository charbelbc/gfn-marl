#!/bin/bash
#SBATCH --account=project_2007401
#SBATCH --job-name=testdreamer
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/wireless.out
#SBATCH --error=logs/wireless.err

module --force purge
module load pytorch/2.2
source /projappl/project_2007401/genflownet/bin/activate
cd /projappl/project_2007401/gfn-marl

srun python main.py