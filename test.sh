srun -p csc401 --gres gpu \
python3 a2_run.py test rnn_model_mhatt.pt \
--with-multihead-attention --cell-type rnn --testing-dir /u/cs401/A2/data/Hansard/Training/ \
--device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py test rnn_model_att.pt \
--with-attention --cell-type rnn --device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py test rnn_model.pt \
--cell-type rnn --device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py test lstm_model_mhatt.pt \
--with-multihead-attention --cell-type lstm --device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py test lstm_model_att.pt \
--with-attention --cell-type lstm --device cuda

srun -p csc401 --gres gpu \
python3 a2_run.py test lstm_model.pt \
--cell-type lstm --device cuda

python3 a2_run.py interact rnn_model_mhatt.pt --with-multihead-attention --cell-type rnn
python3 a2_run.py interact lstm_model_mhatt.pt --with-multihead-attention --cell-type lstm
model.translate("Toronto est une ville du Canada.")
model.translate("Les professeurs devraient bien traiter les assistants d’enseignement.")
model.translate("Les etudiants de l’Universite de Toronto sont excellents.")
