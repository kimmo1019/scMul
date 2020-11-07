from __future__ import division
import tensorflow as tf
import numpy as np 
import scipy.sparse as sp 
import pandas as pd 
import sys, os
import gzip
import json
from scipy.io import mmwrite,mmread
from sklearn.decomposition import PCA,TruncatedSVD,KernelPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})


class sciCAR_Sampler(object):
    def __init__(self,name='A549',min_rna_c=10,max_rna_c=None,min_rna_g=500,max_rna_g=9100,
        min_atac_c=5,max_atac_c=None,min_atac_l=200,max_atac_l=None,save=True):
        #c:cell, g:gene, l:locus
        self.name = name
        self.min_rna_c = min_rna_c
        self.max_rna_c = max_rna_c
        self.min_rna_g = min_rna_g
        self.max_rna_g = max_rna_g
        self.min_atac_c = min_atac_c
        self.max_atac_c = max_atac_c
        self.min_atac_l = min_atac_l
        self.max_atac_l = max_atac_l
        self.data_dir = 'datasets/sci-CAR/%s'%self.name
        if False:
        #if os.path.exists('datasets/sci-CAR/%s/labels.npy'%self.name):
            self.labels = np.load('%s/labels.npy'%self.data_dir)
            self.rna_mat = sp.load_npz('%s/rna_mat.npz'%self.data_dir)
            self.atac_mat = sp.load_npz('%s/atac_mat.npz'%self.data_dir)
        else:
            self.rna_mat, self.atac_mat, self.labels = self.load_data()
        print('scRNA-seq and scATAC-seq:',self.rna_mat.shape,self.atac_mat.shape)
        print('scRNA-seq sparsity:',sp.csr_matrix.count_nonzero(self.rna_mat)/np.prod(self.rna_mat.shape))
        print('scATAC-seq sparsity:',sp.csr_matrix.count_nonzero(self.atac_mat)/np.prod(self.atac_mat.shape))
        sys.exit()
        self.rna_mat = self.rna_mat*10000/self.rna_mat.sum(1)
        #self.rna_mat = self.rna_mat.tocoo()
        self.rna_mat = np.log(1+self.rna_mat)
        self.atac_mat = self.atac_mat*10000/self.atac_mat.sum(1)
        #self.atac_mat = self.atac_mat.tocoo()
        self.atac_mat = np.log(1+self.atac_mat)
        #combine data
        self.combine_mat = np.hstack((self.rna_mat,self.atac_mat))
        
        if False:
            self.rna_mat = mmread('../scAI/data/RNA.txt').todense().T
            self.atac_mat = mmread('../scAI/data/ATAC.txt').todense().T
            print(self.rna_mat.shape,self.atac_mat.shape)
            self.combine_mat = np.hstack((self.rna_mat,self.atac_mat))
            self.labels = [item.split('\t')[-1].strip().strip('"') for item in open('../scAI/data/labels.txt').readlines()]
            assert len(self.labels) == self.rna_mat.shape[0]



    def load_data(self,save=True):
        rna_cell_info = '%s/RNA/RNA_sciCAR_%s_cell.txt.gz'%(self.data_dir,self.name)
        atac_cell_info = '%s/ATAC/ATAC_sciCAR_%s_cell.txt.gz'%(self.data_dir,self.name)
        rna_count_info = '%s/RNA/RNA_sciCAR_%s_gene_count.txt.gz'%(self.data_dir,self.name)
        atac_count_info = '%s/ATAC/ATAC_sciCAR_%s_peak_count.txt.gz'%(self.data_dir,self.name)
        gene_info = '%s/RNA/RNA_sciCAR_%s_gene.txt.gz'%(self.data_dir,self.name)
        peak_info = '%s/ATAC/ATAC_sciCAR_%s_peak.txt.gz'%(self.data_dir,self.name)
        genes = np.loadtxt(gene_info, dtype = str, skiprows = 1,delimiter=',')[:,2]
        peaks = np.array([item.split(',')[1] for item in gzip.open(peak_info).readlines()[1:]])
        rna_cell_ids, rna_labels = self.load_rna_cell(rna_cell_info)
        atac_cell_ids = self.load_atac_cell(atac_cell_info)
        #align cell based on cell id
        rna_cell_index, atac_cell_index, common_cell_ids = self.match_cell(rna_cell_ids,rna_labels,atac_cell_ids)
        rna_labels = np.array(rna_labels)[rna_cell_index]
        print('%d matched cells for %s'%(len(rna_cell_index),self.name))
        #3260 matched cells for A549
        #11296 matched cells for mosue_kidney,8837 cells after removing 'NA' label
        rna_mat = self.load_read_count(rna_count_info)
        atac_mat = self.load_read_count(atac_count_info)
        rna_mat = rna_mat[rna_cell_index,:]
        atac_mat = atac_mat[atac_cell_index,:]
        if save:
            self.save_raw(rna_mat,atac_mat,common_cell_ids,rna_labels,genes,peaks)
        #filter cells and genes
        rna_mat, rna_cell_select, genes = self.filter_rna(rna_mat,genes)
        #(3260,20452)
        #filter cells and loci
        atac_mat, atac_cell_select, peaks= self.filter_atac(atac_mat,peaks)
        #(3260, 45913)
        cell_select = rna_cell_select*atac_cell_select

        rna_mat = rna_mat[cell_select,:]
        atac_mat = atac_mat[cell_select,:]
        labels = rna_labels[cell_select]
        if True:
            self.save_10x(common_cell_ids, genes, rna_mat)
        return rna_mat, atac_mat, labels

    def load_rna_cell(self, cell_info):
        rna_cell_ids, rna_labels = [], []
        with gzip.open(cell_info) as f:
            line = f.readline()
            line = f.readline()
            while line != "":
                cell_id, label = line.split(',')[0], line.rstrip().split(',')[-1]
                if label.startswith('Medullary'):
                    label = 'Medullary collecting duct cells'
                rna_cell_ids.append(cell_id)
                rna_labels.append(label)
                line = f.readline()
        return rna_cell_ids, rna_labels

    def load_atac_cell(self, cell_info):
        atac_cell_ids = []
        with gzip.open(cell_info) as f:
            line = f.readline()
            line = f.readline()
            while line != "":
                cell_id = line.split(',')[0]
                atac_cell_ids.append(cell_id)
                line = f.readline()
        return atac_cell_ids

    def match_cell(self, rna_cell_ids, rna_labels, atac_cell_ids):
        #common_cell_ids = sorted([item for i,item in enumerate(rna_cell_ids) if item in atac_cell_ids and rna_labels[i] != 'NA'])
        common_cell_ids = [item for i,item in enumerate(rna_cell_ids) if item in atac_cell_ids and rna_labels[i] != 'NA']
        rna_cell_index = [rna_cell_ids.index(item) for item in common_cell_ids]
        atac_cell_index = [atac_cell_ids.index(item) for item in common_cell_ids]
        return rna_cell_index, atac_cell_index, common_cell_ids

    def load_read_count(self,count_info):
        mat_sp = mmread(count_info).T  #cells by feats
        return mat_sp.tocsr()

    def filter_rna(self,mat_sp,genes):
        #filter cell
        if self.name == 'A549':
            cell_select = np.array(mat_sp.sum(axis=1) > self.min_rna_g).squeeze()
        else:
            cell_select = np.array((mat_sp>0).sum(axis=1)).squeeze() > self.min_rna_g
        if self.max_rna_g is not None:
            cell_select *= np.array(mat_sp.sum(axis=1) < self.max_rna_g).squeeze()
        #filter gene
        gene_select = np.array((mat_sp>0).sum(axis=0)).squeeze() > self.min_rna_c
        if self.max_rna_c is not None:
            gene_select *= np.array((mat_sp>0).sum(axis=0)).squeeze() < self.max_rna_c
        genes = genes[gene_select]
        return mat_sp[:,gene_select], cell_select, genes

    def filter_atac(self,mat_sp,peaks):
        #filter cell
        cell_select = np.array((mat_sp>0).sum(axis=1)).squeeze() > self.min_atac_l
        if self.max_atac_l is not None:
            cell_select *= np.array((mat_sp>0).sum(axis=1)).squeeze() < self.max_atac_l
        #filter locus
        locus_select = np.array((mat_sp>0).sum(axis=0)).squeeze() > self.min_atac_c
        if self.max_atac_c is not None:
            locus_select *= np.array((mat_sp>0).sum(axis=0)).squeeze() < self.max_atac_c
        peaks = peaks[locus_select]
        return mat_sp[:,locus_select], cell_select, peaks

    #save raw count matrix (paired)
    def save_raw(self,rna_mat,atac_mat,common_cell_ids,rna_labels,genes,peaks):
        mmwrite('datasets/sci-CAR/mouse_kidney/rna_mat.mtx',rna_mat.T) #peaks by cells        
        mmwrite('datasets/sci-CAR/mouse_kidney/atac_mat.mtx',atac_mat.T)
        f=open('datasets/sci-CAR/mouse_kidney/label.txt','w')
        f.write('\n'.join([item[0]+'\t'+item[1] for item in zip(common_cell_ids,rna_labels)]))
        f.close()
        f=open('datasets/sci-CAR/mouse_kidney/peaks.txt','w')
        f.write('\n'.join(peaks))
        f.close()
        f=open('datasets/sci-CAR/mouse_kidney/genes.txt','w')
        f.write('\n'.join(genes))
        f.close()

    def save_10x(self,cell_ids,genes,rna_mat):
        save_dir = 'datasets/sci-CAR/%s/RNA/10x_v2'%self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt('%s/genes.tsv'%save_dir,genes,delimiter='\t',fmt='%s')
        f_out = open('%s/barcodes.tsv'%save_dir,'w')
        f_out.write('\n'.join(cell_ids))
        f_out.close()
        rna_mat = rna_mat.tocoo()
        mmwrite('%s/matrix.mtx'%save_dir,rna_mat.T) #peaks by cells

#load data from uniformed format 
class scPair_Sampler(object):
    def __init__(self,name='A549',min_rna_c=10,max_rna_c=None,min_rna_g=500,max_rna_g=9100,
        min_atac_c=5,max_atac_c=None,min_atac_l=200,max_atac_l=None,scale=10000,save=True):
        #c:cell, g:gene, l:locus
        self.name = name
        self.min_rna_c = min_rna_c
        self.max_rna_c = max_rna_c
        self.min_rna_g = min_rna_g
        self.max_rna_g = max_rna_g
        self.min_atac_c = min_atac_c
        self.max_atac_c = max_atac_c
        self.min_atac_l = min_atac_l
        self.max_atac_l = max_atac_l
        #self.data_dir = 'datasets/scMultiOmicsData/Paired-seq'
        self.data_dir = 'datasets/scMultiOmicsData/sci-CAR'
        
        self.rna_mat, self.atac_mat, self.labels = self.load_data()
        uniq_labels = list(np.unique(self.labels))
        self.Y = np.array([uniq_labels.index(item) for item in self.labels])
        print('scRNA-seq: ', self.rna_mat.shape, 'scATAC-seq: ', self.atac_mat.shape)
        self.rna_mat = self.rna_mat*scale/self.rna_mat.sum(1)
        #self.rna_mat = self.rna_mat.tocoo()
        self.rna_mat = np.log10(1+self.rna_mat)
        self.atac_mat = self.atac_mat*scale/self.atac_mat.sum(1)
        #self.atac_mat = self.atac_mat.tocoo()
        self.atac_mat = np.log10(1+self.atac_mat)
        #simple_analyze(self.rna_mat, self.atac_mat, self.labels, self.name, n_components=30)

        

    def load_data(self,save=False):
        # rna_meta_file = '%s/Paired_seq_Cell_Mix_matrix_rna.json'%self.data_dir
        # atac_meta_file = '%s/Paired_seq_Cell_Mix_matrix_atac.json'%self.data_dir
        # rna_count_file = '%s/Paired_seq_Cell_Mix_matrix_rna_count.mtx'%self.data_dir
        # atac_count_file = '%s/Paired_seq_Cell_Mix_matrix_atac_count.mtx'%self.data_dir
        rna_meta_file = '%s/mouse_kidney_rna.json'%self.data_dir
        atac_meta_file = '%s/mouse_kidney_atac.json'%self.data_dir
        rna_count_file = '%s/mouse_kidney_rna_count.mtx'%self.data_dir
        atac_count_file = '%s/mouse_kidney_atac_count.mtx'%self.data_dir
        rna_meta = json.load(open(rna_meta_file,'r'))
        atac_meta = json.load(open(atac_meta_file,'r'))
        assert rna_meta['cell_name'] == atac_meta['cell_name']
        labels = rna_meta['cell_name']
        genes = rna_meta['gene_name']
        peaks = atac_meta['peak_name']
        rna_mat = mmread(rna_count_file).T.tocsr() #(cells, genes)
        atac_mat = mmread(atac_count_file).T.tocsr() #(cells, peaks)
        assert rna_mat.shape[0] == atac_mat.shape[0]
        #filter cells and genes
        rna_mat, rna_cell_select = self.filter_rna(rna_mat)
        #filter cells and loci
        atac_mat, atac_cell_select= self.filter_atac(atac_mat)

        cell_select = rna_cell_select*atac_cell_select
        rna_mat = rna_mat[cell_select,:]
        atac_mat = atac_mat[cell_select,:]

        labels = np.array(labels)[cell_select]
        return rna_mat, atac_mat, labels

    def filter_rna(self,mat_sp):
        #filter cell
        cell_select = np.array((mat_sp>0).sum(axis=1)).squeeze() > self.min_rna_g
        if self.max_rna_g is not None:
            cell_select *= np.array(mat_sp.sum(axis=1) < self.max_rna_g).squeeze()
        #filter gene
        gene_select = np.array((mat_sp>0).sum(axis=0)).squeeze() > self.min_rna_c
        if self.max_rna_c is not None:
            gene_select *= np.array((mat_sp>0).sum(axis=0)).squeeze() < self.max_rna_c
        return mat_sp[:,gene_select], cell_select

    def filter_atac(self,mat_sp):
        #filter cell
        cell_select = np.array((mat_sp>0).sum(axis=1)).squeeze() > self.min_atac_l
        if self.max_atac_c is not None:
            cell_select *= np.array((mat_sp>0).sum(axis=1)).squeeze() < self.max_atac_l
        #filter locus
        locus_select = np.array((mat_sp>0).sum(axis=0)).squeeze() > self.min_atac_c
        if self.max_atac_l is not None:
            locus_select *= np.array((mat_sp>0).sum(axis=0)).squeeze() < self.max_atac_c
        return mat_sp[:,locus_select], cell_select

    def get_batch(self,batch_size,label=False):
        idx = np.random.randint(low = 0, high = self.atac_mat.shape[0], size = batch_size)
        if label:
            return self.rna_mat[idx,:], self.atac_mat[idx,:], self.Y[idx]
        else:
            return self.rna_mat[idx,:], self.atac_mat[idx,:]

#load data from scAI preprocessed 
class scAI_Sampler(object):
    def __init__(self,name='kidney',n_components=30,random_seed=0,mode=1):
        #c:cell, g:gene, l:locus
        #mode: 1 only scRNA-seq, 2 only scATAC-seq, 3 both
        self.name = name
        self.mode = mode
        self.data_dir = 'datasets/scAI'
        self.rna_mat, self.atac_mat, self.labels, self.genes, self.peaks = self.load_data()
        uniq_labels = list(np.unique(self.labels))
        self.Y = np.array([uniq_labels.index(item) for item in self.labels])
        print('scRNA-seq: ', self.rna_mat.shape, 'scATAC-seq: ', self.atac_mat.shape)
        #simple_analyze(self.rna_mat, self.atac_mat, self.labels, self.name, n_components=n_components,method='pca',vis=False)

        self.rna_mat = self.rna_mat.toarray()
        self.atac_mat = self.atac_mat.toarray()
        #slight difference in fit_transform and fit then transform
        self.rna_reducer = PCA(n_components=n_components, random_state=random_seed)
        self.rna_reducer.fit(self.rna_mat)
        self.pca_rna_mat = self.rna_reducer.transform(self.rna_mat)
        a = self.rna_reducer.fit_transform(self.rna_mat)

        self.atac_reducer = PCA(n_components=n_components, random_state=random_seed)
        self.atac_reducer.fit(self.atac_mat)
        self.pca_atac_mat = self.atac_reducer.transform(self.atac_mat)
        b=self.atac_reducer.fit_transform(self.atac_mat)

        correlation(self.pca_rna_mat,self.Y)
        correlation(self.pca_atac_mat,self.Y)
        correlation(np.hstack((self.pca_rna_mat,self.pca_atac_mat)),self.Y)

        correlation(a,self.Y)
        correlation(b,self.Y)
        correlation(np.hstack((a,b)),self.Y)
        print(random_seed,n_components)
        #sys.exit()
        #inverse
        #print(self.rna_reducer.mean_.shape,self.rna_reducer.components_.shape)
        #data_reduced = np.dot(self.rna_mat - self.rna_reducer.mean_, self.rna_reducer.components_.T)

        #simple_analyze(self.rna_mat, self.atac_mat, self.labels, self.name, n_components=n_components,method='pca',vis=False)
        #simple_analyze(self.rna_mat, self.atac_mat, self.labels, self.name, n_components=n_components,method='linear')
        #simple_analyze(self.rna_mat, self.atac_mat, self.labels, self.name, n_components=n_components,method='rbf')
        #simple_analyze(self.rna_mat, self.atac_mat, self.labels, self.name, n_components=n_components,method='poly')
        #simple_analyze(self.rna_mat, self.atac_mat, self.labels, self.name, n_components=n_components,method='svd')
        

    def load_data(self,use_org=False):
        rna_count_file = '%s/%s_rna_mat.mtx'%(self.data_dir,self.name)
        atac_count_file = '%s/%s_atac_mat.mtx'%(self.data_dir,self.name)
        label_file = '%s/%s_labels.txt'%(self.data_dir,self.name)
        peak_file = '%s/%s_peaks.txt'%(self.data_dir,self.name)
        gene_file = '%s/%s_genes.txt'%(self.data_dir,self.name)
        rna_mat = mmread(rna_count_file).T.tocsr() #(cells, genes)
        atac_mat = mmread(atac_count_file).T.tocsr() #(cells, peaks)
        assert rna_mat.shape[0] == atac_mat.shape[0]
        #load cell labels
        cell_ids = [item.split('\t')[0].strip('"') for item in open(label_file).readlines()[1:]]
        labels = [item.split('\t')[-1].strip().strip('"') for item in open(label_file).readlines()[1:]]
        assert len(labels) == rna_mat.shape[0]
        #load cell peaks
        peaks = [item.strip().strip('"') for item in open(peak_file).readlines()[1:]]
        #load cell genes
        genes = [item.strip().strip('"') for item in open(gene_file).readlines()[1:]]
        assert len(peaks) == atac_mat.shape[1]
        assert len(genes) == rna_mat.shape[1]

        if use_org:
            data_dir = 'datasets/sci-CAR/mouse_kidney'
            rna_count_file = '%s/rna_mat.mtx'%data_dir
            atac_count_file = '%s/atac_mat.mtx'%data_dir
            peak_file = '%s/peaks.txt'%data_dir
            gene_file = '%s/genes.txt'%data_dir
            label_file = '%s/label.txt'%data_dir
            rna_count_mat = mmread(rna_count_file).T.tocsr() #(cells, genes)
            atac_count_mat = mmread(atac_count_file).T.tocsr() #(cells, peaks)
            org_peaks = [item.strip() for item in open(peak_file).readlines()]
            org_genes = [item.strip() for item in open(gene_file).readlines()]
            org_cell_ids = [item.split('\t')[0].replace('-','_').replace('.','_') for item in open(label_file).readlines()]
            cell_idx = [org_cell_ids.index(item) for item in cell_ids]
            peak_idx = [org_peaks.index(item) for item in peaks]
            gene_idx = [org_genes.index(item) for item in genes]
            rna_mat = rna_count_mat[cell_idx,:][:,gene_idx]
            atac_mat = atac_count_mat[cell_idx,:][:,peak_idx]
            #IT-IDF 
            rna_mat = TfidfTransformer().fit_transform(rna_mat)
            atac_mat = TfidfTransformer().fit_transform(atac_mat)
            rna_mat = MinMaxScaler().fit_transform(rna_mat)
            atac_mat = MinMaxScaler().fit_transform(atac_mat)
        return rna_mat, atac_mat, labels, genes, peaks


    def get_batch(self,batch_size):
        idx = np.random.randint(low = 0, high = self.atac_mat.shape[0], size = batch_size)
        if self.mode == 1:
            return self.pca_rna_mat[idx,:]
        elif self.mode == 2:
            return self.pca_atac_mat[idx,:]
        elif self.mode == 3:
            return np.hstack((self.pca_rna_mat, self.pca_atac_mat))[idx,:]
        else:
            print('Wrong mode!')
            sys.exit()
    
    def load_all(self):
        if self.mode == 1:
            #print('1')
            #correlation(self.pca_rna_mat,self.Y,False)
            #sys.exit()
            return self.pca_rna_mat, self.Y
        elif self.mode == 2:
            #print('2')
            #correlation(self.pca_atac_mat,self.Y,False)
            #sys.exit()
            return self.pca_atac_mat, self.Y
        elif self.mode == 3:
            #print('3')
            #correlation(np.hstack((self.pca_rna_mat, self.pca_atac_mat)),self.Y,False)
            #sys,exit()
            return np.hstack((self.pca_rna_mat, self.pca_atac_mat)), self.Y
        else:
            print('Wrong mode!')
            sys.exit()
        



def plot_embedding(X, labels, classes=None, method='tSNE', cmap='tab20', figsize=(8, 8), markersize=15, dpi=300,marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=True, show_axis_label=True, **legend_params):
    if marker is not None:
        X = np.concatenate([X, marker], axis=0)
    N = len(labels)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    matplotlib.rcParams.update({'font.size': 22})
    if X.shape[1] != 2:
        if method == 'tSNE':
            from sklearn.manifold import TSNE
            X = TSNE(n_components=2, random_state=124).fit_transform(X)
        if method == 'PCA':
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=124).fit_transform(X)
        if method == 'UMAP':
            from umap import UMAP
            X = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(X)
    labels = np.array(labels)
    plt.figure(figsize=figsize)
    if classes is None:
        classes = np.unique(labels)
    #tab10, tab20, husl, hls
    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.husl_palette(len(classes), s=.8)
    #markersize = 80
    for i, c in enumerate(classes):
        plt.scatter(X[:N][labels==c, 0], X[:N][labels==c, 1], s=markersize, color=colors[i], label=c)
    if marker is not None:
        plt.scatter(X[N:, 0], X[N:, 1], s=10*markersize, color='black', marker='*')
    
    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 20,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method+' dim 1', fontsize=12)
        plt.ylabel(method+' dim 2', fontsize=12)

    if save:
        plt.savefig(save, format='png', bbox_inches='tight',dpi=dpi)

def correlation(X,Y,heatmap=True):
    nb_classes = len(set(Y))
    km = KMeans(n_clusters=nb_classes,random_state=0,n_init=100).fit(X)
    label_kmeans = km.labels_
    nmi = normalized_mutual_info_score(Y, label_kmeans)
    ari = adjusted_rand_score(Y, label_kmeans)
    homogeneity = homogeneity_score(Y, label_kmeans)
    print('NMI = {}, ARI = {}, Homogeneity = {}'.format(nmi,ari,homogeneity))
    if heatmap:
        x_ticks = ['']*len(Y)
        y_ticks = ['']*len(Y)
        idx = []
        for i in range(nb_classes):
            sub_idx = [j for j,item in enumerate(Y) if item==i]
            idx += [j for j,item in enumerate(Y) if item==i]
            x_ticks[len(idx)-1] = str(i)
        assert len(idx)==len(Y)
        X = X[idx,:]
        Y = Y[idx]
        #similarity_mat = pairwise_distances(X,metric='cosine')
        similarity_mat = cosine_similarity(X)
        #sns.heatmap(similarity_mat,cmap='Blues')
        fig, ax = plt.subplots()
        #ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels(y_ticks)
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks)
        im = ax.imshow(similarity_mat,cmap='Blues')
        plt.colorbar(im)
        plt.savefig('heatmap.png',dpi=400)

def analyze_atac(atac_mat,labels,save):
    X = atac_mat.toarray().T
    ind = np.sum(X>0,axis=1) > X.shape[1]*0.03
    X = X[ind,:]
    print(X.shape)
    #tf-idf tranformation
    nfreqs = 1.0 * X / np.tile(np.sum(X,axis=0), (X.shape[0],1))
    X  = nfreqs * np.tile(np.log(1 + 1.0 * X.shape[1] / np.sum(X,axis=1)).reshape(-1,1), (1,X.shape[1]))
    X = X.T
    X = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components=30, random_state=3456).fit_transform(X)
    vis_tsne = TSNE(n_components=2, random_state=124).fit_transform(pca)
    plot_embedding(vis_tsne,labels,save=save)
    uniq_labels = list(np.unique(labels))
    Y = np.array([uniq_labels.index(item) for item in labels])
    correlation(X,Y)

def simple_analyze(rna_mat, atac_mat, labels, name, n_components=30,random_state=124, method='pca',vis=False):
    assert rna_mat.shape[0] == atac_mat.shape[0]
    uniq_labels = list(np.unique(labels))
    Y = np.array([uniq_labels.index(item) for item in labels])
    
    if method == 'pca':
        rna_mat = rna_mat.toarray()
        atac_mat = atac_mat.toarray()
        combine_mat = np.hstack((rna_mat, atac_mat))
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'linear':
        combine_mat = sp.hstack((rna_mat, atac_mat))
        reducer = KernelPCA(n_components=n_components, kernel='linear',random_state=random_state)
    elif method == 'rbf':
        combine_mat = sp.hstack((rna_mat, atac_mat))
        reducer = KernelPCA(n_components=n_components, kernel='rbf',random_state=random_state)
    elif method == 'poly':
        combine_mat = sp.hstack((rna_mat, atac_mat))
        reducer = KernelPCA(n_components=n_components, kernel='poly',random_state=random_state)
    elif method == 'svd':
        combine_mat = sp.hstack((rna_mat, atac_mat))
        reducer = TruncatedSVD(n_components=n_components,random_state=random_state)
    print('using %s'%method)
    #scRNA
    print('scRNA data:', rna_mat.shape)
    rna_pca = reducer.fit_transform(rna_mat)
    #print(rna_pca[0,:5])
    #sys.exit()
    if vis:
        vis_tsne = TSNE(n_components=2, random_state=124).fit_transform(rna_pca)
        plot_embedding(vis_tsne,labels,save='%s_scRNA_tsne.png'%name)
    correlation(rna_pca,Y,False)

    #scATAC
    print('scATAC data:', atac_mat.shape)
    atac_pca = reducer.fit_transform(atac_mat)
    if vis:
        vis_tsne = TSNE(n_components=2, random_state=124).fit_transform(atac_pca)
        plot_embedding(vis_tsne,labels,save='%s_scATAC_tsne.png'%name)
    correlation(atac_pca,Y,False)

    #combine after pca
    print('Late integration:')
    concat_pca = np.hstack((rna_pca, atac_pca))
    if vis:
        vis_tsne = TSNE(n_components=2, random_state=124).fit_transform(concat_pca)
        plot_embedding(vis_tsne,labels,save='%s_late_integration_tsne.png'%name)
    correlation(concat_pca,Y,False)

    #combined before pca
    print('Early integration:', combine_mat.shape)
    concat_pca = reducer.fit_transform(combine_mat)
    if vis:
        vis_tsne = TSNE(n_components=2, random_state=124).fit_transform(concat_pca)
        plot_embedding(vis_tsne,labels,save='%s_early_integration_tsne.png'%name)
    correlation(concat_pca,Y,False)

def find_RE_index(genes, peaks, thred = 100000):
    def find_nearby_RE(gene):
        RE_idx = []
        tag = 0
        g_chrom, g_start, g_end = geneid2loc[gene2id[gene]]
        for i in range(len(peaks)):
            if peaks[i].startswith(g_chrom+'-'):
                tag = 1
                RE_loc = int(peaks[i].split('-')[1])
                if abs(RE_loc-g_start)<thred:
                    RE_idx.append(i)
            else:
               if tag == 1:
                   break
        return RE_idx

    annot_mm10_file = '/home/liuqiao/mm10/gencode.vM25.annotation.gtf'
    orig_gene_file = 'datasets/sci-CAR/mouse_kidney/RNA/RNA_sciCAR_mouse_kidney_gene.txt.gz'
    gene2id = {item.split(',')[-1].strip():item.split(',')[0].split('.')[0] for  \
        item in gzip.open(orig_gene_file).readlines()[1:]}
    gene2type = {item.split(',')[-1].strip():item.split(',')[1] for  \
        item in gzip.open(orig_gene_file).readlines()[1:]}
    geneid2loc = {item.split('\t')[-1].split(';')[0].strip().split(' ')[1].strip('"').split('.')[0]: \
        [item.split('\t')[0],int(item.split('\t')[3]), int(item.split('\t')[4])]  \
        for item in open(annot_mm10_file).readlines()[5:] if item.split('\t')[2]=="gene"}
    print(genes[0],geneid2loc[gene2id[genes[0]]])
    print(genes[1],geneid2loc[gene2id[genes[1]]])
    print(genes[2],geneid2loc[gene2id[genes[2]]])
    print(genes[3],geneid2loc[gene2id[genes[3]]])
    print(genes[4],geneid2loc[gene2id[genes[4]]])
    #manually annotation
    geneid2loc['ENSMUSG00000102049'] = ['chr1',133655879,133660885] #Zbed6
    geneid2loc['ENSMUSG00000097045'] = ['chr1',171423185,171437322] #Gm26641
    geneid2loc['ENSMUSG00000097458'] = ['chr2',126521201,126545941] #Gm26697
    geneid2loc['ENSMUSG00000078484'] = ['chr4',156229044,156234857] #Klhl17
    geneid2loc['ENSMUSG00000030771'] = ['chr7',112368308,112413104] #Micalcl
    geneid2loc['ENSMUSG00000108569'] = ['chr7',134498191,134501160] #Gm4593
    geneid2loc['ENSMUSG00000023577'] = ['chr9',106504665,106561646] #Iqcf3
    geneid2loc['ENSMUSG00000097457'] = ['chr11',46440362,46448185] #2310031A07Rik
    geneid2loc['ENSMUSG00000097658'] = ['chr11',120486339,120488582] #Gm16755
    geneid2loc['ENSMUSG00000051107'] = ['chr13',51685534,51686045] #Gm15440
    geneid2loc['ENSMUSG00000097242'] = ['chr13',63289154,63297147] #Gm16907
    geneid2loc['ENSMUSG00000079489'] = ['chr14',56733162,56734954] #C030013D06Rik
    geneid2loc['ENSMUSG00000097103'] = ['chr17',6798383,6828016] #Gm2885
    geneid2loc['ENSMUSG00000090673'] = ['chr19',41482637,41586536] #Gm340
    geneid2loc['ENSMUSG00000097632'] = ['chr19',55640589,55702204] #4930552P12Rik
    geneid2loc['ENSMUSG00000091562'] = ['chr16',59492088,59601047] #Crybg3
    genes_len = []
    for each in genes:
        #46 genes have no location info -->30 after manually annoted
        if gene2id[each] not in geneid2loc.keys():
            pass
            #print(each,gene2id[each],gene2type[each])
        else:
            chrom, start, end = geneid2loc[gene2id[each]]
            genes_len.append(end-start)
    print(np.min(genes_len),np.max(genes_len),np.median(genes_len),np.mean(genes_len))
    selected_genes = [item for item in genes if gene2id[item] in geneid2loc.keys()]
    Gene_idx = [genes.index(item) for item in selected_genes]
    RE_idx_list = map(find_nearby_RE,selected_genes)
    #filter genes that have no nearby REs
    Gene_idx_filtered = [item[0] for item in zip(Gene_idx,RE_idx_list) if len(item[1])>0]
    RE_idx_list_filtered = [item[1] for item in zip(Gene_idx,RE_idx_list) if len(item[1])>0]

    # print(Gene_idx_filtered[:3],Gene_idx_filtered[-3:],len(Gene_idx_filtered))
    # print(len(RE_idx_list_filtered),RE_idx_list_filtered[0][-5:],RE_idx_list_filtered[-1][-5:])
    # print(np.min([len(item) for item in RE_idx_list_filtered]),np.max([len(item) for item in RE_idx_list_filtered]))
    # print(len([len(item) for item in RE_idx_list_filtered if len(item)==0]))
    return Gene_idx_filtered, RE_idx_list_filtered

#sample continuous (Gaussian) and discrete (Catagory) latent variables together
class Mixture_sampler(object):
    def __init__(self, nb_classes, N, dim, sd, scale=1):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        self.sd = sd 
        self.scale = scale
        np.random.seed(1024)
        self.X_c = self.scale*np.random.normal(0, self.sd**2, (self.total_size,self.dim))
        #self.X_c = self.scale*np.random.uniform(-1, 1, (self.total_size,self.dim))
        self.label_idx = np.random.randint(low = 0 , high = self.nb_classes, size = self.total_size)
        self.X_d = np.eye(self.nb_classes)[self.label_idx]
        self.X = np.hstack((self.X_c,self.X_d))
    
    def train(self,batch_size,weights=None):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim))
        #X_batch_c = self.scale*np.random.uniform(-1, 1, (batch_size,self.dim))
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        X_batch_d = np.eye(self.nb_classes)[label_batch_idx]
        return X_batch_c, X_batch_d

    def load_all(self):
        return self.X_c, self.X_d

#get a batch of data from previous 50 batches, add stochastic
class DataPool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.nb_batch = 0
        self.pool = []

    def __call__(self, data):
        if self.nb_batch < self.maxsize:
            self.pool.append(data)
            self.nb_batch += 1
            return data
        if np.random.rand() > 0.5:
            results=[]
            for i in range(len(data)):
                idx = int(np.random.rand()*self.maxsize)
                results.append(copy.copy(self.pool[idx])[i])
                self.pool[idx][i] = data[i]
            return results
        else:
            return data
            
def get_cosine_similarity(A, B):
    numerator = tf.reduce_sum(A*B,axis=1)
    l2_norm_a = tf.reduce_sum(A*A,axis=1)
    l2_norm_b = tf.reduce_sum(B*B,axis=1)
    denominator = tf.sqrt(l2_norm_a*l2_norm_b)+1e-5
    return tf.math.divide(numerator,denominator)


if __name__ == "__main__":
    s = scAI_Sampler(n_components=10,random_seed=121)
    data = np.load('results/kidney/20201023_103448_x_dim=6_y_dim=20_alpha=10.0_beta=10.0_ratio=0.7/data_at_24800.npz')
    data_x_,data_x_onehot_,label_y = data['arr_0'],data['arr_1'],data['arr_2']
    vis_tsne = TSNE(n_components=2, random_state=124).fit_transform(data_x_)
    plot_embedding(vis_tsne,s.labels,save='scDEC_combine_tsne.png')
    sys.exit()
    #s = scPair_Sampler(name='pair_seq',min_rna_c=10,max_rna_c=None,min_rna_g=10,max_rna_g=None,
    #  min_atac_c=5,max_atac_c=None,min_atac_l=10,max_atac_l=None,save=False)

    #s = sciCAR_Sampler(name='A549',min_rna_c=10,max_rna_c=None,min_rna_g=500,max_rna_g=9100,
    #    min_atac_c=5,max_atac_c=None,min_atac_l=200,max_atac_l=None,save=False)

    #s = sciCAR_Sampler(name='mouse_kidney',min_rna_c=5,max_rna_c=None,min_rna_g=100,max_rna_g=None,
    #   min_atac_c=40,max_atac_c=None,min_atac_l=200,max_atac_l=None,save=True)

    #s = scPair_Sampler(name='sci-CAR',min_rna_c=200,max_rna_c=None,min_rna_g=100,max_rna_g=None,
    #  min_atac_c=50,max_atac_c=None,min_atac_l=100,max_atac_l=None,save=False)

    #rna_mat, atac_mat, labels = s.rna_mat, s.atac_mat, s.labels
    #analyze_atac(atac_mat,labels,save='kidney_tsne.png')
    #s = scAI_Sampler(random_seed=124)
    s = scAI_Sampler(n_components=10,random_seed=121)
    s = scAI_Sampler(n_components=20,random_seed=121)
    s = scAI_Sampler(n_components=30,random_seed=121)
    s = scAI_Sampler(n_components=40,random_seed=121)
    s = scAI_Sampler(n_components=50,random_seed=121)

    sys.exit()
    find_RE_index(s.genes,s.peaks)





