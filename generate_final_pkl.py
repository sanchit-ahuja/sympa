import os
import pickle
from nodecls import load_data
from collections import defaultdict


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
            # fin_data_list.append({'emb': hyp_embedding, 'ds': ds})
            # ds = ds.to('cuda')
            # bert_embedding = ds.ndata['x']
            # train_mask = ds.ndata['train_mask'].tolist()
            # val_mask = ds.ndata['val_mask'].tolist()
            # test_mask = ds.ndata['test_mask'].tolist()
            # if 1 in train_mask:
            #     train_label = train_mask.index(1)
            #     fin_data_dic['train'].append({
            #         'hyp_embedding': hyp_embedding,
            #         'bert_embedding' : bert_embedding,
            #         'train_mask': train_mask,
            #     })
            # elif 1 in val_mask:
            #     val_label = val_mask.index(1)
            #     fin_data_dic['val'].append({
            #         'hyp_embedding': hyp_embedding,
            #         'bert_embedding' : bert_embedding,
            #         'val_mask': val_mask,
            #     })
            # elif 1 in test_mask:
            #     test_label = test_mask.index(1)
            #     fin_data_dic['test'].append({
            #         'hyp_embedding': hyp_embedding,
            #         'bert_embedding' : bert_embedding,
            #         'test_mask': test_mask,
            #     })
    return fin_data_dic

            

        

if __name__ == "__main__":
    data = generate_final_pkl('pheme_dgl_full_roberta.pkl', 'ckpt_prod-hyeu')
    # save data as a pickle file
    with open('pheme_dgl_full_roberta_final.pkl', 'wb') as f:
        pickle.dump(data, f)
    # print(len(data['val']),'val')
    # print(len(data['test']),'test')
    # print(len(data['train']),'train')
