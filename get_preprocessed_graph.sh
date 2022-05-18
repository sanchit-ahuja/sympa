#!/bin/bash
# iterate from 0 to 6641
for i in `seq 0 6641`; do
    echo "Processing $i"
    python preprocess.py --graph=graph_list$i --run_id=$i --grid_dims=2
    # ./get_preprocessed_graph.py $i
done