import os
import pickle
from nodecls import load_data
from collections import defaultdict
from get_dense_list import get_dense_list


def rename_files_in_dir(dir_name, file_name_suffix):
    # get the list of files in the directory
    files = os.listdir(dir_name)
    # rename the files
    for i, file in enumerate(files):
        os.rename(dir_name + "/" + file, dir_name + "/" + file + file_name_suffix)
    

''''

{
    'hyp_embedding': hyp_embedding,
    'bert_embedding' : bert_embedding,
    'train_label': train_label,
}

'''

def generate_final_pkl(dir_name, hyp_embedding_dir):
    with open(dir_name, 'rb') as f:
        data = pickle.load(f)
    
    hyp_files = os.listdir(hyp_embedding_dir)
    idx_to_file_loc_dic = {}
    fin_data_dic = defaultdict(list)
    fin_data_list = []
    for file in hyp_files:
        idx = int(file.split('-')[0])
        idx_to_file_loc_dic[idx] = file
    for i, ds in enumerate(data):
        print("Processing dataset {}".format(i))
        if i in idx_to_file_loc_dic:
            hyp_file = idx_to_file_loc_dic[i]
            try:
                hyp_embedding = load_data(hyp_embedding_dir + "/" + hyp_file)
            except:
                print("Error in loading file {}".format(hyp_file))
                continue
            fin_data_dic['hyp_embedding'].append(hyp_embedding)
            fin_data_dic['ds'].append(ds)
    return fin_data_dic

            
def get_final(file_name, hyp_file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    hyp_embeddings = load_data(hyp_file_name) 
    fin_data_dic = defaultdict(list)
    
    cnt = 0

    for i, ds in enumerate(data):
        adj_matrix = ds.adj(scipy_fmt="csr")
        adj_matrix = adj_matrix.toarray()
        edge_list = get_dense_list(adj_matrix)
        if edge_list == []:
            fin_data_dic['hyp_embedding'].append(None)
            fin_data_dic['ds'].append(ds)
        else:
            num_nodes = len(edge_list) + 1
            hyp_emb = hyp_embeddings[cnt: cnt + num_nodes]
            cnt += num_nodes
            fin_data_dic['hyp_embedding'].append(hyp_emb)
            fin_data_dic['ds'].append(ds)
            
    return fin_data_dic

    

        
    
        
    
    
        

if __name__ == "__main__":
    file_name = 'si_digraph_dgl_time_large.pkl'
    hyp_file_name = '1-bet-prod-hyhy-5ep'
    fin_data_dic = get_final(file_name, hyp_file_name)
    with open('si_digraph_dgl_time_large_final.pkl', 'wb') as f:
        pickle.dump(fin_data_dic, f)
    

