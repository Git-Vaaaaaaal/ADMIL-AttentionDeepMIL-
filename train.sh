#!/bin/ksh 
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N ADMIL-AttentionDeepMIL-
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/ADMIL-AttentionDeepMIL-
source /beegfs/data/work/imvia/in156281/ADMIL-AttentionDeepMIL-/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/ADMIL-AttentionDeepMIL-/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib

python /beegfs/data/work/imvia/in156281/ADMIL-AttentionDeepMIL-/train.py