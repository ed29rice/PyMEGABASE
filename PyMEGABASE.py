import pyBigWig
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time, glob, random, requests, shutil
from tqdm import tqdm
from joblib import Parallel, delayed
from pydca.plmdca import plmdca


class PyMEGABASE:
    def __init__(self, cell_line='GM12878', assembly='hg19',signal_type='signal p-value',
                 ref_cell_line_path='tmp_meta',cell_line_path=None,types_path='PyMEGABSE/types'):
        self.printHeader()
        self.cell_line=cell_line
        self.assembly=assembly
        self.signal_type=signal_type
        if cell_line_path==None:
            self.cell_line_path=cell_line+'_'+assembly
        else:
            self.cell_line_path=cell_line_path
        self.ref_cell_line='GM12878'
        self.ref_assembly='hg19'
        self.ref_cell_line_path=ref_cell_line_path
        self.types_path=types_path
        #import necessary libaries

        #Define tranlation dictinaries between aminoacids, intensity of Chip-seq signal and 
        self.RES_TO_INT = {
                'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
                'S': 16, 'T': 17, 'V': 18, 'W':19, 'Y':20,
                '-':21, '.':21, '~':21,
        }
        self.INT_TO_RES = {self.RES_TO_INT[k]:k for k in self.RES_TO_INT.keys()}

        self.TYPE_TO_INT = {'A1':0,'A2':1,'B1':2,'B2':3,'B3':4,'B4':5,'NA':6}

        self.INT_TO_TYPE = {self.TYPE_TO_INT[k]:k for k in self.TYPE_TO_INT.keys()}
        
        if assembly=='GRCh38':
            self.chrm_size = np.array([4980,4844,3966,3805,3631,3417,3187,2903,2768,2676,2702,2666,2288,2141,2040,1807,1666,1608,1173,1289,935,1017])
        else:
            self.chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028])
        
        self.histones=['H2AFZ-human', 'H3K27me3-human', 'H3K27ac-human' , 'H3K36me3-human', 'H3K4me1-human',
       'H3K4me2-human', 'H3K4me3-human', 'H3K79me2-human', 'H3K9ac-human',
       'H3K9me3-human', 'H4K20me1-human']

        
    def process_replica(self,line,cell_line_path,chrm_size):
        text=line.split()[0]
        exp=line.split()[1]
        count=line.split()[2]

        #Experiment directory 
        exp_path=cell_line_path+'/'+exp+'_'+str(count)

        if exp in self.histones:
            try:
                os.mkdir(exp_path)
            except:
                print('Directory ',exp_path,' already exist')

            with open(exp_path+'/exp_name.txt', 'w') as f:
                f.write(text+' '+exp+'\n')

            #Load data from server
            try:
                bw = pyBigWig.open("https://www.encodeproject.org/files/"+text+"/@@download/"+text+".bigWig")
                for chr in range(1,23):
                    signal = bw.stats("chr"+str(chr), type="mean", nBins=chrm_size[chr-1])
    
                    #Process signal and binning 
                    signal=np.array(signal)
                    per=np.percentile(signal[signal!=None],95)
                    signal[signal==None]=0.0
                    signal[signal>per]=per
                    signal=signal*19/per
                    signal=np.round(signal.astype(float)).astype(int)
    
                    #Save data
                    with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:
    
                        f.write("#chromosome file number of beads\n"+str(chrm_size[chr-1]))
                        f.write("#\n")
                        f.write("#bead, signal, discrete signal\n")
                        for i in range(len(signal)):
                            f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
                return exp_path

            except:
                print('This experiment was incomplete:',text,'\nit will not be used.')
                        
    
    def download_and_process_cell_line_data(self,nproc=10):
        
        try:
            os.mkdir(self.cell_line_path)
        except:
            print('Directory ',self.cell_line_path,' already exist')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.cell_line_path)
            os.mkdir(self.cell_line_path)
                    
        r = requests.get('https://www.encodeproject.org/metadata/?type=Experiment&assay_title=Histone+ChIP-seq&biosample_ontology.term_name='
                         +self.cell_line+'&files.file_type=bigWig&assay_title=Histone+ChIP-seq')
        content=str(r.content)
        self.cell_line_meta=content
        with open(self.cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.assembly and l[4]==self.signal_type:
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+'\n')
        
        count=0
        self.exp_found={}
        exp_name=''
        list_names=[]

        with open(self.cell_line_path+'/meta.txt') as fp:
            Lines = fp.readlines()
            for line in Lines:
                count += 1
                text=line.split()[0]
                exp=line.split()[1]

                #Register if experiment is new
                if exp!=exp_name:
                    try:
                        count=self.exp_found[exp]+1
                    except:
                        count=1
                    exp_name=exp
                self.exp_found[exp]=count
                list_names.append(text+' '+exp+' '+str(count))

        print('Number of replicas:', len(list_names))
        results = Parallel(n_jobs=nproc)(delayed(self.process_replica)(list_names[i],self.cell_line_path,self.chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))

        print('Experiments found in ENCODE:')
        print(self.exp_found)
        
        self.unique=[]
        print('Prediction will use:')
        with open(self.cell_line_path+'/11_marks.txt', 'w') as f:
            for e in self.histones:
                if e in self.exp_found.keys():
                    f.write(e.split('-hum')[0]+'\n')
                    print(e.split('-hum')[0])
                    self.unique.append(e)
                    
    def download_and_process_ref_data(self,nproc):
        
        try:
            os.mkdir(self.ref_cell_line_path)
        except:
            print('Directory ',self.ref_cell_line_path,' already exist')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.ref_cell_line_path)
            os.mkdir(self.ref_cell_line_path)
        
        r = requests.get('https://www.encodeproject.org/metadata/?type=Experiment&assay_title=Histone+ChIP-seq&biosample_ontology.term_name='
                         +self.ref_cell_line+'&files.file_type=bigWig&assay_title=Histone+ChIP-seq')
        content=str(r.content)
        content.split('\\n')
        self.ref_meta=content
        with open(self.ref_cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.ref_assembly and l[4]==self.signal_type:
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+'\n')
          
        ref_chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028])

        count=0
        exp_found={}
        exp_name=''
        list_names=[]

        with open(self.ref_cell_line_path+'/meta.txt') as fp:
            Lines = fp.readlines()
            for line in Lines:
                count += 1
                text=line.split()[0]
                exp=line.split()[1]
                #Register if experiment is new
                if exp!=exp_name:
                    try:
                        count=exp_found[exp]+1
                    except:
                        count=1
                    exp_name=exp
                exp_found[exp]=count
                list_names.append(text+' '+exp+' '+str(count))

        print('Number of replicas:', len(list_names))

        results = Parallel(n_jobs=nproc)(delayed(self.process_replica)(list_names[i],self.ref_cell_line_path,ref_chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        print('Experiments found in ENCODE:')
        print(exp_found)

        print('Prediction will use:')
        with open(self.ref_cell_line_path+'/11_marks.txt', 'w') as f:
            for e in self.unique:
                if e in exp_found.keys():
                    f.write(e.split('-hum')[0]+'\n')
                    print(e.split('-hum')[0])
                    
    def build_state_vector(self,int_types,all_averages):
        #Aggregate tracks by with data from other loci l-2, l-1, l, l+1, l+2
        #l+1
        shift1=np.copy(all_averages)
        shift1[:,:-1]=all_averages[:,1:]
        shift1[:,-1]=np.zeros(len(shift1[:,-1]))
        #l+2
        shift2=np.copy(all_averages)
        shift2[:,:-1]=shift1[:,1:]
        shift2[:,-1]=np.zeros(len(shift1[:,-1]))
        #l-1
        shift_1=np.copy(all_averages)
        shift_1[:,1:]=all_averages[:,:-1]
        shift_1[:,0]=np.zeros(len(shift_1[:,-1]))
        #l-2
        shift_2=np.copy(all_averages)
        shift_2[:,1:]=shift_1[:,:-1]
        shift_2[:,0]=np.zeros(len(shift1[:,-1]))

        #Stack shifted tracks and subtypes labels
        all_averages=np.vstack((int_types,shift_2,shift_1,all_averages,shift1,shift2))

        #To train, we exclude the centromers and B4 subcompartments
        ndx=(all_averages[0,:]!=6) * (all_averages[0,:]!=5)
        all_averages=all_averages[:,ndx]
        all_averages=all_averages+1

        return all_averages

                    
    def training_set_up(self):
        # We are training in odd chromosomes data
        if self.cell_line=='GM12878' and self.assembly=='hg19':
            chrms=[1,3,5,7,9,11,13,15,17,19,21]
        else:
            chrms=[i for i in range(1,23)]

        #Load types from Rao et al 2014 paper
        types=[]
        for chr in chrms:
            types.append(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str)[:,1])
        types=np.concatenate(types)
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))

        #Check which experiments are available to train 
        unique=np.loadtxt(self.cell_line_path+'/11_marks.txt',dtype=str)
        print('To train the following experiments are used:')
        print(unique)

        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            print(u)
            for i in glob.glob(self.ref_cell_line_path+'/'+u+'*'):
                tmp=[]
                try:
                    for chr in chrms:
                        _tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                        tmp.append(_tmp)
                    tmp=np.concatenate(tmp)
                    reps.append(tmp)
                except:
                    print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.round(np.mean(reps,axis=0))
            all_averages.append(ave_reps)

        all_averages=np.array(all_averages)
        all_averages=self.build_state_vector(int_types,all_averages)

        # Translate Potts states to sequences
        sequences=np.array(list(map(self.INT_TO_RES.get, all_averages.flatten()))).reshape(all_averages.shape)

        #Generate sequence file 
        with open(self.cell_line_path+"/sequences.fa",'w',encoding = 'utf-8') as f:
            for i in range(len(sequences.T)):
                f.write('>'+str(i).zfill(4)+'\n')
                f.write(''.join(sequences[:,i])+'\n')
    
    
    def training(self,nproc=10):
        # Compute DCA scores using Pseudolikelihood maximization algorithm
        plmdca_inst = plmdca.PlmDCA(
            self.cell_line_path+"/sequences.fa",
            'protein',
            seqid = 0.99,
            lambda_h = 100,
            lambda_J = 100,
            num_threads = nproc,
            max_iterations = 1000)

        # Train an get coupling and fields as lists
        fields_and_couplings = plmdca_inst.get_fields_and_couplings_from_backend()
        couplings = plmdca_inst.get_couplings_no_gap_state(fields_and_couplings)
        fields = plmdca_inst.get_fields_no_gap_state(fields_and_couplings)

        #Reshape couplings and fields to a working format 
        # J should be shaped (56,56,20,20)
        # h should be shaped (56,20)
        L = plmdca_inst._get_num_and_len_of_seqs()[1]
        q = 21
        self.L=L
        self.q=q
        qm1 = q - 1
        J=np.zeros((L,L,qm1,qm1))
        fields_all = fields_and_couplings[:L * q]
        h = list()
        for i in range(L):
            for j in range(i + 1, L):
                start_indx = int(((L *  (L - 1)/2) - (L - i) * ((L-i)-1)/2  + j  - i - 1) * qm1 * qm1)
                end_indx = start_indx + qm1 * qm1
                couplings_ij = couplings[start_indx:end_indx]
                couplings_ij = np.reshape(couplings_ij, (qm1,qm1))
                J[i,j]=couplings_ij
                J[j,i]=couplings_ij
            h.append(fields_all[i * q:(i+1)*q])
        h=np.array(h)
        print('Training finished')
        print('J and H produced')
        self.h=h
        self.J=J
        
    def prediction(self,chr=1):
        
        types=["A1" for i in range(self.chrm_size[chr-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        
        unique=np.loadtxt(self.cell_line_path+'/11_marks.txt',dtype=str)
        print('To predict the following experiments are used:')
        print(unique)
        
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+u+'*'):
                tmp=[]
                try:
                    tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                    reps.append(tmp)
                except:
                    print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.round(np.mean(reps,axis=0))
            all_averages.append(ave_reps)

        all_averages=np.array(all_averages)
        self.chr_averages=self.build_state_vector(int_types,all_averages)-1
        
        #Prediction 
        predict_type=np.zeros(self.chr_averages.shape[1])
        fails=0;r=0
        for loci in range(self.chr_averages.shape[1]):
            energy_val=[]
            energy=0
            #Check energy for all possible 5 states (A1,A2,B1,B2,B3)
            for state in range(5):
                tmp_energy=-self.h[0,state]
                for j in range(1,self.L):
                    s2=int(self.chr_averages[j,loci])
                    tmp_energy=tmp_energy-self.J[0,j,state,s2]
                energy_val.append(energy+tmp_energy)
            energy_val=np.array(energy_val)
            #Select the state with the lowest energy
            predict_type[loci]=np.where(energy_val==np.min(energy_val))[0][0]
        
        #Add gaps from UCSC database
        gaps=np.loadtxt('PyMEGABASE/gaps/'+self.assembly+'_gaps.txt',dtype=str)
        chr_gaps_ndx=np.where((gaps[:,0]=='chr'+str(chr)))[0]
        for gp in chr_gaps_ndx:
            init_loci=np.round(gaps[gp,1].astype(float)/50000).astype(int)
            end_loci=np.round(gaps[gp,2].astype(float)/50000).astype(int)
            predict_type[init_loci:end_loci]=6
     
        return predict_type

    def printHeader(self):
        print('{:^96s}'.format("****************************************************************************************"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("****************************************************************************************"))
        print('')
        print('{:^96s}'.format("The PyMEGABASE class performs the prediction of subcompartment annotations"))
        print('{:^96s}'.format("based on Chip-Seq data tracks of Histone modifications. The input data is "))
        print('{:^96s}'.format("obtained from ENCODE data base. PyMEGABASE is the implementation of MEGABASE"))
        print('{:^96s}'.format("method of prediction with BigWig Chip-Seq files."))
        print('')
        print('{:^96s}'.format("PyMEGABASE description is described in: TBD,"))
        print('')
        print('{:^96s}'.format("This package is the product of contributions from a number of people, including:"))
        print('{:^96s}'.format("Esteban Dodero-Rojas, Antonio Oliveira, Vinícius Contessoto,"))
        print('{:^96s}'.format("Ryan Cheng, and, Jose Onuchic"))
        print('{:^96s}'.format("Rice University"))
        print('')
        print('{:^96s}'.format("****************************************************************************************"))



class PyMEGABASE_extended:
    def __init__(self, cell_line='GM12878', assembly='hg19',signal_type='signal p-value',
                 ref_cell_line_path='tmp_meta',cell_line_path=None,types_path='PyMEGABSE/types',
                 histones=True,tf=False,atac=False,small_rna=False,total_rna=False):
        self.printHeader()
        self.cell_line=cell_line
        self.assembly=assembly
        self.signal_type=signal_type
        if cell_line_path==None:
            self.cell_line_path=cell_line+'_'+assembly
        else:
            self.cell_line_path=cell_line_path
        self.ref_cell_line='GM12878'
        self.ref_assembly='hg19'
        self.ref_cell_line_path=ref_cell_line_path
        self.types_path=types_path
        self.hist=histones
        self.tf=tf
        self.atac=atac
        self.small_rna=small_rna
        self.total_rna=total_rna

        #Define tranlation dictinaries between aminoacids, intensity of Chip-seq signal and 
        self.RES_TO_INT = {
                'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
                'S': 16, 'T': 17, 'V': 18, 'W':19, 'Y':20,
                '-':21, '.':21, '~':21,
        }
        self.INT_TO_RES = {self.RES_TO_INT[k]:k for k in self.RES_TO_INT.keys()}

        self.TYPE_TO_INT = {'A1':0,'A2':1,'B1':2,'B2':3,'B3':4,'B4':5,'NA':6}

        self.INT_TO_TYPE = {self.TYPE_TO_INT[k]:k for k in self.TYPE_TO_INT.keys()}
        
        if assembly=='GRCh38':
            self.chrm_size = np.array([4980,4844,3966,3805,3631,3417,3187,2903,2768,2676,2702,2666,2288,2141,2040,1807,1666,1608,1173,1289,935,1017])
        else:
            self.chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028])

        url='https://www.encodeproject.org/metadata/?type=Experiment&'
        if self.hist==True:
            url=url+'assay_title=Histone+ChIP-seq'
        if self.tf==True:
            url=url+'&assay_title=TF+ChIP-seq'
        if self.atac==True:
            url=url+'&assay_title=ATAC-seq'
        if self.small_rna==True:
            url=url+'&assay_title=small+RNA-seq'
        if self.total_rna==True:
            url=url+'&assay_title=total+RNA-seq'
        self.url_ref=url+'&biosample_ontology.term_name='+self.ref_cell_line+'&files.file_type=bigWig'

        r = requests.get(self.url_ref)
        content=str(r.content)
        experiments=[]
        for k in content.split('\\n')[:-1]:
            l=k.split('\\t')
            if l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                experiments.append(l[22])
            elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                experiments.append(l[7])
            elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                experiments.append(l[22])
            elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                experiments.append('small-RNA-seq')
            elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                experiments.append('total-RNA-seq')          

        self.experiments_unique=np.unique(experiments)   
        
    def process_replica(self,line,cell_line_path,chrm_size):
        text=line.split()[0]
        exp=line.split()[1]
        count=line.split()[2]

        #Experiment directory 
        exp_path=cell_line_path+'/'+exp+'_'+str(count)

        if exp in self.experiments_unique:
            try:
                os.mkdir(exp_path)
            except:
                print('Directory ',exp_path,' already exist')

            with open(exp_path+'/exp_name.txt', 'w') as f:
                f.write(text+' '+exp+'\n')

            #Load data from server

            bw = pyBigWig.open("https://www.encodeproject.org/files/"+text+"/@@download/"+text+".bigWig")
            for chr in range(1,23):
                signal = bw.stats("chr"+str(chr), type="mean", nBins=chrm_size[chr-1])

                #Process signal and binning 
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                signal[signal==None]=0.0
                signal[signal>per]=per
                signal=signal*19/per
                signal=np.round(signal.astype(float)).astype(int)

                #Save data
                with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:

                    f.write("#chromosome file number of beads\n"+str(chrm_size[chr-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
            #except:
            #    print('This experiment is unavailable:',exp)
            return exp_path
    
    def download_and_process_cell_line_data(self,nproc=10):
        
        try:
            os.mkdir(self.cell_line_path)
        except:
            print('Directory ',self.cell_line_path,' already exist')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.cell_line_path)
            os.mkdir(self.cell_line_path)
        
        url='https://www.encodeproject.org/metadata/?type=Experiment&'
        if self.hist==True:
            url=url+'assay_title=Histone+ChIP-seq'
        if self.tf==True:
            url=url+'&assay_title=TF+ChIP-seq'
        if self.atac==True:
            url=url+'&assay_title=ATAC-seq'
        if self.small_rna==True:
            url=url+'&assay_title=small+RNA-seq'
        if self.total_rna==True:
            url=url+'&assay_title=total+RNA-seq'
        self.url_cell_line=url+'&biosample_ontology.term_name='+self.cell_line+'&files.file_type=bigWig'

        r = requests.get(self.url_cell_line)
        content=str(r.content)
        experiments=[]
        with open(self.cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                    f.write(l[0]+' '+l[7]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'small-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'total-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
        
        count=0
        self.exp_found={}
        exp_name=''
        list_names=[]

        with open(self.cell_line_path+'/meta.txt') as fp:
            Lines = fp.readlines()
            for line in Lines:
                count += 1
                text=line.split()[0]
                exp=line.split()[1]

                #Register if experiment is new
                if exp!=exp_name:
                    try:
                        count=self.exp_found[exp]+1
                    except:
                        count=1
                    exp_name=exp
                self.exp_found[exp]=count
                list_names.append(text+' '+exp+' '+str(count))

        print('Number of replicas:', len(list_names))
        results = Parallel(n_jobs=nproc)(delayed(self.process_replica)(list_names[i],self.cell_line_path,self.chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))

        print('Experiments found in ENCODE:')
        print(self.exp_found)
        
        self.unique=[]
        print('Prediction will use:')
        with open(self.cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.experiments_unique:
                if e in self.exp_found.keys():
                    f.write(e.split('-hum')[0]+'\n')
                    print(e.split('-hum')[0])
                    self.unique.append(e)
                    
    def download_and_process_ref_data(self,nproc):
        
        try:
            os.mkdir(self.ref_cell_line_path)
        except:
            print('Directory ',self.ref_cell_line_path,' already exist')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.ref_cell_line_path)
            os.mkdir(self.ref_cell_line_path)
        
        url='https://www.encodeproject.org/metadata/?type=Experiment&'
        if self.hist==True:
            url=url+'assay_title=Histone+ChIP-seq'
        if self.tf==True:
            url=url+'&assay_title=TF+ChIP-seq'
        if self.atac==True:
            url=url+'&assay_title=ATAC-seq'
        if self.small_rna==True:
            url=url+'&assay_title=small+RNA-seq'
        if self.total_rna==True:
            url=url+'&assay_title=total+RNA-seq'
        self.url_ref=url+'&biosample_ontology.term_name='+self.ref_cell_line+'&files.file_type=bigWig'

        r = requests.get(self.url_ref)
        content=str(r.content)
        experiments=[]
        with open(self.ref_cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                    f.write(l[0]+' '+l[7]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'small-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'total-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
          
        ref_chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028])

        count=0
        exp_found={}
        exp_name=''
        list_names=[]

        with open(self.ref_cell_line_path+'/meta.txt') as fp:
            Lines = fp.readlines()
            for line in Lines:
                count += 1
                text=line.split()[0]
                exp=line.split()[1]
                #Register if experiment is new
                if exp!=exp_name:
                    try:
                        count=exp_found[exp]+1
                    except:
                        count=1
                    exp_name=exp
                exp_found[exp]=count
                list_names.append(text+' '+exp+' '+str(count))

        print('Number of replicas:', len(list_names))

        results = Parallel(n_jobs=nproc)(delayed(self.process_replica)(list_names[i],self.ref_cell_line_path,ref_chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        print('Experiments found in ENCODE:')
        print(exp_found)

        print('Prediction will use:')
        with open(self.ref_cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.unique:
                if e in exp_found.keys():
                    f.write(e.split('-hum')[0]+'\n')
                    print(e.split('-hum')[0])
                    
    def build_state_vector(self,int_types,all_averages):
        #Aggregate tracks by with data from other loci l-2, l-1, l, l+1, l+2
        #l+1
        shift1=np.copy(all_averages)
        shift1[:,:-1]=all_averages[:,1:]
        shift1[:,-1]=np.zeros(len(shift1[:,-1]))
        #l+2
        shift2=np.copy(all_averages)
        shift2[:,:-1]=shift1[:,1:]
        shift2[:,-1]=np.zeros(len(shift1[:,-1]))
        #l-1
        shift_1=np.copy(all_averages)
        shift_1[:,1:]=all_averages[:,:-1]
        shift_1[:,0]=np.zeros(len(shift_1[:,-1]))
        #l-2
        shift_2=np.copy(all_averages)
        shift_2[:,1:]=shift_1[:,:-1]
        shift_2[:,0]=np.zeros(len(shift1[:,-1]))

        #Stack shifted tracks and subtypes labels
        all_averages=np.vstack((int_types,shift_2,shift_1,all_averages,shift1,shift2))

        #To train, we exclude the centromers and B4 subcompartments
        ndx=(all_averages[0,:]!=6) * (all_averages[0,:]!=5)
        all_averages=all_averages[:,ndx]
        all_averages=all_averages+1

        return all_averages

                    
    def training_set_up(self):
        # We are training in odd chromosomes data
        if self.cell_line=='GM12878' and self.assembly=='hg19':
            chrms=[1,3,5,7,9,11,13,15,17,19,21]
        else:
            chrms=[i for i in range(1,23)]

        #Load types from Rao et al 2014 paper
        types=[]
        for chr in chrms:
            types.append(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str)[:,1])
        types=np.concatenate(types)
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))

        #Check which experiments are available to train 
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)
        print('To train the following experiments are used:')
        print(unique)

        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            print(u)
            for i in glob.glob(self.ref_cell_line_path+'/'+u+'*'):
                tmp=[]
                try:
                    for chr in chrms:
                        _tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                        tmp.append(_tmp)
                    tmp=np.concatenate(tmp)
                    reps.append(tmp)
                except:
                    print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.round(np.mean(reps,axis=0))
            all_averages.append(ave_reps)

        all_averages=np.array(all_averages)
        all_averages=self.build_state_vector(int_types,all_averages)

        # Translate Potts states to sequences
        sequences=np.array(list(map(self.INT_TO_RES.get, all_averages.flatten()))).reshape(all_averages.shape)

        #Generate sequence file 
        with open(self.cell_line_path+"/sequences.fa",'w',encoding = 'utf-8') as f:
            for i in range(len(sequences.T)):
                f.write('>'+str(i).zfill(4)+'\n')
                f.write(''.join(sequences[:,i])+'\n')
    
    
    def training(self,nproc=10):
        # Compute DCA scores using Pseudolikelihood maximization algorithm
        plmdca_inst = plmdca.PlmDCA(
            self.cell_line_path+"/sequences.fa",
            'protein',
            seqid = 0.99,
            lambda_h = 100,
            lambda_J = 100,
            num_threads = nproc,
            max_iterations = 1000)

        # Train an get coupling and fields as lists
        fields_and_couplings = plmdca_inst.get_fields_and_couplings_from_backend()
        couplings = plmdca_inst.get_couplings_no_gap_state(fields_and_couplings)
        fields = plmdca_inst.get_fields_no_gap_state(fields_and_couplings)

        #Reshape couplings and fields to a working format 
        # J should be shaped (56,56,20,20)
        # h should be shaped (56,20)
        L = plmdca_inst._get_num_and_len_of_seqs()[1]
        q = 21
        self.L=L
        self.q=q
        qm1 = q - 1
        J=np.zeros((L,L,qm1,qm1))
        fields_all = fields_and_couplings[:L * q]
        h = list()
        for i in range(L):
            for j in range(i + 1, L):
                start_indx = int(((L *  (L - 1)/2) - (L - i) * ((L-i)-1)/2  + j  - i - 1) * qm1 * qm1)
                end_indx = start_indx + qm1 * qm1
                couplings_ij = couplings[start_indx:end_indx]
                couplings_ij = np.reshape(couplings_ij, (qm1,qm1))
                J[i,j]=couplings_ij
                J[j,i]=couplings_ij
            h.append(fields_all[i * q:(i+1)*q])
        h=np.array(h)
        print('Training finished')
        print('J and H produced')
        self.h=h
        self.J=J
        
    def prediction(self,chr=1):
        
        types=["A1" for i in range(self.chrm_size[chr-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)
        print('To predict the following experiments are used:')
        print(unique)
        
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+u+'*'):
                tmp=[]
                try:
                    tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                    reps.append(tmp)
                except:
                    print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.round(np.mean(reps,axis=0))
            all_averages.append(ave_reps)

        all_averages=np.array(all_averages)
        self.chr_averages=self.build_state_vector(int_types,all_averages)-1
        
        #Prediction 
        predict_type=np.zeros(self.chr_averages.shape[1])
        fails=0;r=0
        for loci in range(self.chr_averages.shape[1]):
            energy_val=[]
            energy=0
            #Check energy for all possible 5 states (A1,A2,B1,B2,B3)
            for state in range(5):
                tmp_energy=-self.h[0,state]
                for j in range(1,self.L):
                    s2=int(self.chr_averages[j,loci])
                    tmp_energy=tmp_energy-self.J[0,j,state,s2]
                energy_val.append(energy+tmp_energy)
            energy_val=np.array(energy_val)
            #Select the state with the lowest energy
            predict_type[loci]=np.where(energy_val==np.min(energy_val))[0][0]

        #Add gaps from UCSC database
        gaps=np.loadtxt('PyMEGABASE/gaps/'+self.assembly+'_gaps.txt',dtype=str)
        chr_gaps_ndx=np.where((gaps[:,0]=='chr'+str(chr)))[0]
        for gp in chr_gaps_ndx:
            init_loci=np.round(gaps[gp,1].astype(float)/50000).astype(int)
            end_loci=np.round(gaps[gp,2].astype(float)/50000).astype(int)
            predict_type[init_loci:end_loci]=6
               
        return predict_type

    def printHeader(self):
        print('{:^96s}'.format("****************************************************************************************"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("****************************************************************************************"))
        print('')
        print('{:^96s}'.format("The PyMEGABASE class performs the prediction of subcompartment annotations"))
        print('{:^96s}'.format("based on Chip-Seq data tracks of Histone modifications. The input data is "))
        print('{:^96s}'.format("obtained from ENCODE data base. PyMEGABASE is the implementation of MEGABASE"))
        print('{:^96s}'.format("method of prediction with BigWig Chip-Seq files."))
        print('')
        print('{:^96s}'.format("PyMEGABASE description is described in: TBD,"))
        print('')
        print('{:^96s}'.format("This package is the product of contributions from a number of people, including:"))
        print('{:^96s}'.format("Esteban Dodero-Rojas, Antonio Oliveira, Vinícius Contessoto,"))
        print('{:^96s}'.format("Ryan Cheng, and, Jose Onuchic"))
        print('{:^96s}'.format("Rice University"))
        print('')
        print('{:^96s}'.format("****************************************************************************************"))
