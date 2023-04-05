srun -p csc401 --gres gpu \
python3 a2_run.py train \
rnn_model.pt \
--cell-type rnn \
--viz-wandb blair-yang \
--device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py train \
lstm_model.pt \
--cell-type lstm \
--viz-wandb blair-yang \
--device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py train \
rnn_model_att.pt \
--with-attention \
--cell-type rnn \
--epochs 2 \
--viz-wandb blair-yang \
--device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py train \
rnn_model_mhatt.pt \
--with-multihead-attention \
--cell-type rnn \
--viz-wandb blair-yang \
--device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py train \
lstm_model_mhatt_2epochs.pt \
--with-multihead-attention \
--cell-type lstm \
--epochs 2 \
--viz-wandb blair-yang \
--device cuda