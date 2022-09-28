#!/bin/bash
# iterate from 0 to 6641
for i in `seq 0 33774`; do
    echo "Processing $i"
    python -m torch.distributed.launch train.py --ckpt_path=prod_hyhy --data_path=data --n_procs=8 --data=$i --run_id=$i --model=prod-hyhy --results_file=out_prod_hyhy_si/results_$i.csv --dims=2 --epochs=5
    # python preprocess.py --graph=graph_list$i --run_id=$i --grid_dims=2
    # ./get_preprocessed_graph.py $i
done

# 584 graph does not work
