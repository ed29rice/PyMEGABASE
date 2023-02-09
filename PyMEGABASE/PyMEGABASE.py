import os, glob, requests, shutil, pyBigWig, urllib, gzip
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from urllib.parse import urlparse
try:
    from pydca.plmdca import plmdca
except ImportError as e:
    print('Could not find pydca, it will download the package now')
    os.system('pip install -q pydca --no-deps')
    from pydca.plmdca import plmdca
    pass  # module doesn't exist, deal with it.

class PyMEGABASE_legacy:
    R"""
    The :class:`~.PyMEGABASE` class performs genomic annotations .
    
    The :class:`~.PyMEGABASE` sets the environment to generate prediction of genomic annotations.
    
    Args:
        cell_line (str, required): 
            Name of target cell type
        assembly (str, required): 
            Reference assembly of target cell line ('hg19','GRCh38','mm10')
        organism (str, required): 
            Target cell type organism (str, required):
        signal_type (str, required): 
            Input files signal type ('signal p-value', 'fold change over control', ...)
        ref_cell_line_path (str, optional): 
            Folder/Path to place reference/training data ('tmp_meta')
        cell_line_path (str, optional): 
            Folder/Path to place target cell data 
        types_path (str, optional): 
            Folder/Path where the genomic annotations are located 
        histones (bool, required): 
            Whether to use Histone Modification data from the ENCODE databank for prediction
        tf (bool, required): 
            Whether to use Transcription Factor data from the ENCODE databank for prediction
        small_rna (bool, required): 
            Whether to use Small RNA-Seq data from the ENCODE databank for prediction
        total_rna (bool, required): 
            Whether to use Total RNA-seq data from the ENCODE databank for prediction
        n_states (int, optional): 
            Number of states for the D-nodes 
        extra_filter (str, optional):
            Add filter to the fetching data url to download cell type data
        res (int, optional):
            Resolution for genomic annotations calling in kilobasepairs (5, 50, 100)
        chromosome_sizes (list, optional):
            Chromosome sizes based on the reference genome assembly - required for non-human assemblies
    """
    def __init__(self, cell_line='GM12878', assembly='hg19',organism='human',signal_type='signal p-value',
                 ref_cell_line_path='tmp_meta',cell_line_path=None,types_path='PyMEGABASE/types',
                 histones=True,tf=False,atac=False,small_rna=False,total_rna=False,n_states=19,
                 extra_filter='',res=50,chromosome_sizes=None,AB=False):
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
        self.n_states=n_states
        self.extra_filter=extra_filter
        self.res=res
        self.organism=organism.lower()

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
            self.chrm_size = np.array([4980,4844,3966,3805,3631,3417,3187,2903,2768,2676,2702,2666,2288,2141,2040,1807,1666,1608,1173,1289,935,1017,3121])*50/self.res
        elif assembly=='hg19':
            self.chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028,3105])*50/self.res
        else:
            if chromosome_sizes == None: 
                raise ValueError("Need to specify chromosome sizes for assembly: {}".format(assembly))
            self.chrm_size = np.array(chromosome_sizes)/(self.res*1000)
        self.chrm_size=np.round(self.chrm_size+0.1).astype(int)
        self.ref_chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028,1028])*50/self.res
        self.ref_chrm_size=np.round(self.ref_chrm_size+0.1).astype(int)

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
                experiments.append('plus-small-RNA-seq')
            elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                experiments.append('plus-total-RNA-seq')          
            elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                experiments.append('minus-small-RNA-seq')
            elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                experiments.append('minus-total-RNA-seq')          

        self.experiments_unique=np.unique(experiments)
        self.es_unique=[]   
        for e in self.experiments_unique:
            self.es_unique.append(e.split('-human')[0])

        print('Selected cell line to predict: '+self.cell_line)
        print('Selected assembly: '+self.assembly)
        print('Selected signal type: '+self.signal_type)
        print('Selected organism: '+self.organism)
        
    def process_replica(self,line,cell_line_path,chrm_size):
        R"""
        Preprocess function for each replica 

        Args: 
            line (lsit, required):
                Information about the replica: name, ENCODE id and replica id
            cell_line_path (str, required):
                Path to target cell type data
            chrm_size (list, required):
                Chromosome sizes based on the assembly
        """
        text=line.split()[0]
        exp=line.split()[1]
        count=line.split()[2]

        #Experiment directory 
        exp_path=cell_line_path+'/'+exp+'_'+str(count)

        if 'human' in exp.split('-'): ext='human'
        else: ext=self.organism
        if exp.split('-'+ext)[0] in self.es_unique:
            try:
                os.mkdir(exp_path)
            except:
                print('Directory ',exp_path,' already exist')

            with open(exp_path+'/exp_name.txt', 'w') as f:
                f.write(text+' '+exp+'\n')
                
            #Load data from server
            try:
                bw = pyBigWig.open("https://www.encodeproject.org/files/"+text+"/@@download/"+text+".bigWig")
                for chr in range(1,len(chrm_size)):
                    signal = bw.stats("chr"+str(chr), type="mean", nBins=chrm_size[chr-1])

                    #Process signal and binning 
                    signal=np.array(signal)
                    per=np.percentile(signal[signal!=None],95)
                    per_min=np.percentile(signal[signal!=None],5)
                    signal[signal==None]=per_min
                    signal[signal<per_min]=per_min
                    signal[signal>per]=per
                    signal=signal-per_min
                    signal=signal*self.n_states/(per-per_min)
                    signal=np.round(signal.astype(float)).astype(int)

                    #Save data
                    with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:

                        f.write("#chromosome file number of beads\n"+str(chrm_size[chr-1]))
                        f.write("#\n")
                        f.write("#bead, signal, discrete signal\n")
                        for i in range(len(signal)):
                            f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
                chr='X'
                signal = bw.stats("chr"+chr, type="mean", nBins=chrm_size[-1])
                #Process signal and binning
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)

                #Save data
                with open(exp_path+'/chr'+chr+'.track', 'w') as f:

                    f.write("#chromosome file number of beads\n"+str(chrm_size[-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
                return exp

            except:
                print('This experiment was incomplete:',text,'\nit will not be used.')
 
    def download_and_process_cell_line_data(self,nproc=10,all_exp=True):
        R"""
        Download and preprocess target cell data for the D-nodes

        Args: 
            nproc (int, required):
                Number of processors dedicated to download and process data
            all_exp (bool, optional):
                Download and process all replicas for each experiment
        """
        try:
            os.mkdir(self.cell_line_path)
        except:
            print('Directory ',self.cell_line_path,' already exist')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.cell_line_path)
            os.mkdir(self.cell_line_path)
        
        url='https://www.encodeproject.org/metadata/?type=Experiment&'+self.extra_filter
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
                    f.write(l[0]+' '+'plus-small-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'plus-total-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'minus-small-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'minus-total-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
       
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
                if all_exp==True:
                    list_names.append(text+' '+exp+' '+str(count))
                else:
                    if count==1:
                        list_names.append(text+' '+exp+' '+str(count))

        print('Number of replicas:', len(list_names))
        self.successful_exp = Parallel(n_jobs=nproc)(delayed(self.process_replica)(list_names[i],self.cell_line_path,self.chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        self.successful_exp= [i for i in self.successful_exp if i]
        self.successful_unique_exp=np.unique(self.successful_exp)
        self.su_unique=[]   
        for e in self.successful_unique_exp:
            self.su_unique.append(e.split('-'+self.organism)[0])
        print('Experiments found in ENCODE for the selected cell line:')
        self.unique=[]

        with open(self.cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.experiments_unique:
                if e.split('-human')[0] in self.su_unique:
                    f.write(e.split('-human')[0]+'\n')
                    print(e.split('-human')[0])
                    self.unique.append(e)
        if len(self.unique) > 4:
            print('Predictions would use: ',len(self.unique),' experiments')
        else:
            print('This sample only has ',len(self.unique),' experiments. We do not recommend prediction on samples with less than 5 different experiments.')
                    
    def download_and_process_ref_data(self,nproc,all_exp=True):
        R"""
        Download and preprocess reference data for the D-nodes

        Args: 
            nproc (int, required):
                Number of processors dedicated to download and process data
            all_exp (bool, optional):
                Download and process all replicas for each experiment
        """
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
                    f.write(l[0]+' '+'plus-small-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'plus-total-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'minus-small-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'minus-total-RNA-seq'+' '+l[5]+' '+l[4]+'\n')
         
        ref_chrm_size = self.ref_chrm_size
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
                if (exp.split('-human')[0] in self.su_unique) or (text.split('-human')[0] in self.su_unique):
                    if exp!=exp_name:
                        try:
                            count=exp_found[exp]+1
                        except:
                            count=1
                        exp_name=exp
                    exp_found[exp]=count
                    if all_exp==True:
                        list_names.append(text+' '+exp+' '+str(count))
                    else:
                        if count==1:
                            list_names.append(text+' '+exp+' '+str(count))

        print('Number of replicas:', len(list_names))

        results = Parallel(n_jobs=nproc)(delayed(self.process_replica)(list_names[i],self.ref_cell_line_path,ref_chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))

        print('Prediction will use:')
        with open(self.ref_cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.unique:
                if e in exp_found.keys():
                    f.write(e.split('-'+self.organism)[0]+'\n')
                    print(e.split('-'+self.organism)[0])

    def extra_track(self,experiment,bw_file):
        R"""
        Function to introduce custom tracks

        Args: 
            experiment (str, required):
                Name of the experiment
            bw_file (str, required):
                Path to the custom track
        """
        if not self.organism in experiment: experiment=experiment+'-'+self.organism
        if not experiment.split('-'+self.organism)[0] in self.es_unique: 
            print('This experiment is not found in the training set, then cannot be used.')
            return 0
        if experiment in self.exp_found.keys():
            print('This target has replicas already')
            print('The new track will be addded as a different replica of the same target')
            #Experiment directory
            count=self.exp_found[experiment]+1
            
        else:
            print('This target has no replicas')
            print('The new track will be added a the first replica of the target')
            #Experiment directory
            count=1
        exp_path=self.cell_line_path+'/'+experiment+'_'+str(count)
        print(exp_path)
        
        try:
            os.mkdir(exp_path)
        except:
            print('Directory ',exp_path,' already exist')

        with open(exp_path+'/exp_name.txt', 'w') as f:
            f.write(experiment+' '+experiment+'\n')

        #Load data from track
        
        try:

            bw = pyBigWig.open(bw_file)
            for chr in range(1,len(self.chrm_size)):
                signal = bw.stats("chr"+str(chr), type="mean", nBins=self.chrm_size[chr-1])

                #Process signal and binning
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)

                #Save data
                with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:

                    f.write("#chromosome file number of beads\n"+str(self.chrm_size[chr-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
            chr='X'
            signal = bw.stats("chr"+chr, type="mean", nBins=self.chrm_size[-1])

            #Process signal and binning
            signal=np.array(signal)
            per=np.percentile(signal[signal!=None],95)
            per_min=np.percentile(signal[signal!=None],5)
            signal[signal==None]=per_min
            signal[signal<per_min]=per_min
            signal[signal>per]=per
            signal=signal-per_min
            signal=signal*self.n_states/(per-per_min)
            signal=np.round(signal.astype(float)).astype(int)

            #Save data
            with open(exp_path+'/chr'+chr+'.track', 'w') as f:

                f.write("#chromosome file number of beads\n"+str(self.chrm_size[-1]))
                f.write("#\n")
                f.write("#bead, signal, discrete signal\n")
                for i in range(len(signal)):
                    f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")

            if experiment in self.exp_found.keys():
                self.exp_found[experiment]=self.exp_found[experiment]+1

            else:
                self.exp_found[experiment]=1
                self.successful_unique_exp=np.append(self.successful_unique_exp,experiment)
                self.su_unique.append(experiment.split('-'+self.organism)[0])
                with open(self.cell_line_path+'/unique_exp.txt', 'a') as f:
                    f.write(experiment.split('-'+self.organism)[0]+'\n')
                    self.unique.append(experiment)

            return experiment
        
        except:
            print('This experiment was incomplete:',experiment,'\nit will not be used.')

    def build_state_vector(self,int_types,all_averages):
        R"""
        Builds the set of state vectors used on the training process

        Args: 
            int_types (list, required):
                Genomic annotations
            all_averages (list, required):
                D-node data 
        """
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

    def get_tmatrix(self,chrms,silent=False):
        R"""
        Extract the training data

        Args: 
            chrms (list, optional):
                Set of chromosomes from the reference data used as the training set
            silent (bool, optional):
                Silence outputs
        """
        #Load types from Rao et al 2014 paper
        types=[]
        for chr in chrms:
            if self.res==50:
                types.append(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str)[:,1])
            elif self.res==100:
                types.append(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str)[::int(self.res/50),1])
            else:
                tmp=list(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str))
                if len(tmp) < self.ref_chrm_size[chr-1]:
                    diff=self.ref_chrm_size[chr-1] - len(tmp)
                    for i in range( diff ):
                        tmp.append('NA')
                types.append(tmp)

        types=np.concatenate(types)
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))

        #Check which experiments are available to train 
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)
        if unique.shape==(): unique=[unique]
        if silent==False:print('To train the following experiments are used:')

        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            if silent==False:print(u)
            for i in glob.glob(self.ref_cell_line_path+'/'+str(u)+'*'):
                tmp=[]
                try:
                    for chr in chrms:
                        _tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                        tmp.append(_tmp)
                    tmp=np.concatenate(tmp)
                    reps.append(tmp)
                except:
                    if silent==False:print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.round(np.mean(reps,axis=0))
            all_averages.append(ave_reps)

        all_averages=np.array(all_averages)
        all_averages=self.build_state_vector(int_types,all_averages)
        return all_averages

    def filter_exp(self):
        R"""
        Performs baseline assestment on experiment baselines
        """
        a=[]
        for i in range(1,3):
            a.append(self.test_set(chr=i,silent=True))
        a=np.concatenate(a,axis=1)

        locus=2
        good_exp=0
        gexp=[]
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)

        for exper in range(len(unique)):
            i=exper+len(unique)*locus
            if (np.abs(np.mean(a[i])-np.mean(self.tmatrix[i+1]))<1) and (np.std(a[i])-np.std(self.tmatrix[i+1])<2):
                good_exp=good_exp+1
                gexp.append(unique[exper]+'\n')
            else:
                print('Not using '+unique[exper],' to predict')

        #gexp=sample(gexp,8)
        with open(self.cell_line_path+'/unique_exp_filtered.txt','w') as f:    
            for i in gexp:
                f.write(i)
            print('Number of suitable experiments for prediction:',good_exp)
        if good_exp>0:
            os.system('mv '+self.cell_line_path+'/unique_exp.txt '+self.cell_line_path+'/unique_exp_bu.txt')
            os.system('mv '+self.cell_line_path+'/unique_exp_filtered.txt '+self.cell_line_path+'/unique_exp.txt')
        else:
            print('There are no experiment suitable for the prediction')

    def training_set_up(self,chrms=None,filter=True):
        R"""
        Formats data to allow the training

        Args: 
            chrms (list, optional):
                Set of chromosomes from the reference data to use as the training set
            filter (bool, optional):
                Filter experiments based on the baseline
        """
        if chrms==None:
            # We are training in odd chromosomes data
            if self.cell_line=='GM12878' and self.assembly=='hg19':
                chrms=[1,3,5,7,9,11,13,15,17,19,21]
            else:
                chrms=[i for i in range(1,23)]
        
        if filter==True: 
            all_averages=self.get_tmatrix(chrms,silent=True)
            self.tmatrix=np.copy(all_averages)
            self.filter_exp()

        all_averages=self.get_tmatrix(chrms,silent=False)
        self.tmatrix=np.copy(all_averages)
        
        # Translate Potts states to sequences
        sequences=np.array(list(map(self.INT_TO_RES.get, all_averages.flatten()))).reshape(all_averages.shape)

        #Generate sequence file 
        with open(self.cell_line_path+"/sequences.fa",'w',encoding = 'utf-8') as f:
            for i in range(len(sequences.T)):
                f.write('>'+str(i).zfill(4)+'\n')
                f.write(''.join(sequences[:,i])+'\n')
     
    def training(self,nproc=10,lambda_h=100,lambda_J=100):
        R"""
        Performs the training of the Potts model based on the reference data

        Args: 
            nproc (int, required):
                Number of processors used to train
            lambda_h (bool, optional):
                Value for the intensity of the regularization value for the h energy term
            lambda_J (float, optional):
                Value for the intensity of the regularization value for the J energy term
        Returns:
            array (size of chromosome,5*number of unique experiments)
                D-node input data

        """
        # Compute DCA scores using Pseudolikelihood maximization algorithm
        plmdca_inst = plmdca.PlmDCA(
            self.cell_line_path+"/sequences.fa",
            'protein',
            seqid = 0.99,
            lambda_h = lambda_h,
            lambda_J = lambda_J,
            num_threads = nproc,
            max_iterations = 1000)
        print('Training started')
        # Train an get coupling and fields as lists
        self.fields_and_couplings = plmdca_inst.get_fields_and_couplings_from_backend()
        self.plmdca_inst=plmdca_inst
        fields_and_couplings = self.fields_and_couplings
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
        h_and_J={}
        h_and_J['h']=h
        h_and_J['J']=J
        h_and_J['J_flat']=couplings
        #Save fields and couplings 
        with open(self.cell_line_path+'/h_and_J.npy', 'wb') as f:
            np.save(f, h_and_J)

    def test_set(self,chr=1,silent=False):
        R"""
        Predicts and outputs the genomic annotations for chromosome X

        Args: 
            chr (int, required):
                Chromosome to extract input data fro the D-nodes
            silent (bool, optional):
                Avoid printing information 
        Returns:
            array (size of chromosome,5*number of unique experiments)
                D-node input data

        """
        if silent==False:print('Test set for chromosome: ',chr)        
        if chr!='X':
            types=["A1" for i in range(self.chrm_size[chr-1])]
        else:
            types=["A1" for i in range(self.chrm_size[-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str) 
        if unique.shape==(): unique=[unique]
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+str(u)+'*'):
                tmp=[]
                try:
                    tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                    reps.append(tmp)
                except:
                    if silent==False:print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.round(np.mean(reps,axis=0))
            all_averages.append(ave_reps)

        all_averages=np.array(all_averages)
        chr_averages=self.build_state_vector(int_types,all_averages)-1
        return chr_averages[1:]+1
  
    def prediction_single_chrom(self,chr=1,h_and_J_file=None):
        R"""
        Predicts and outputs the genomic annotations for chromosome X

        Args: 
            chr (int, optional):
                Chromosome to predict
            h_and_J_file (str, optional):
                Model energy term file path
        
        Returns:
            array (size of chromosome)
                Predicted annotations

        """
        print('Predicting subcompartments for chromosome: ',chr)       
        if h_and_J_file!=None:
            with open(h_and_J_file, 'rb') as f:
                h_and_J = np.load(f, allow_pickle=True)
                h_and_J = h_and_J.item()
            self.h=h_and_J['h']
            self.J=h_and_J['J']
 
        types=["A1" for i in range(self.chrm_size[chr-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str) 
        if unique.shape==(): unique=[unique]
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+str(u)+'*'):
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
        fails=0;r=0;
        self.L=len(self.h)
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
        try: 
            gaps=np.loadtxt('PyMEGABASE/gaps/'+self.assembly+'_gaps.txt',dtype=str)
            chr_gaps_ndx=np.where((gaps[:,0]=='chr'+str(chr)))[0]
            for gp in chr_gaps_ndx:
                init_loci=np.round(gaps[gp,1].astype(float)/(self.res*1000)).astype(int)
                end_loci=np.round(gaps[gp,2].astype(float)/(self.res*1000)).astype(int)
                predict_type[init_loci:end_loci]=6
        except:
            print('Gaps not found, not included in predictions')
               
        return predict_type

    def prediction_X(self,chr='X',h_and_J_file=None):
        R"""
        Predicts and outputs the genomic annotations for chromosome X

        Args: 
            chr (int, optional):
                Chromosome to predict
            h_and_J_file (str, optional):
                Model energy term file path
        
        Returns:
            array (size of chromosome)
                Predicted annotations

        """


        print('Predicting subcompartments for chromosome: ',chr)       
        if h_and_J_file!=None:
            with open(h_and_J_file, 'rb') as f:
                h_and_J = np.load(f, allow_pickle=True)
                h_and_J = h_and_J.item()
            self.h=h_and_J['h']
            self.J=h_and_J['J']
 
        types=["A1" for i in range(self.chrm_size[-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str) 
        if unique.shape==(): unique=[unique]
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+str(u)+'*'):
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
        self.L=len(self.h)
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
        try:
            print('Resolution:', self.res)
            gaps=np.loadtxt('PyMEGABASE/gaps/'+self.assembly+'_gaps.txt',dtype=str)
            chr_gaps_ndx=np.where((gaps[:,0]=='chr'+str(chr)))[0]
            for gp in chr_gaps_ndx:
                init_loci=np.round(gaps[gp,1].astype(float)/self.res*1000).astype(int)
                end_loci=np.round(gaps[gp,2].astype(float)/self.res*1000).astype(int)
                predict_type[init_loci:end_loci]=6
        except:
            print('Gaps not found, not included in predictions')
               
        return predict_type

    def write_bed(self,out_file='predictions', compartments=True,subcompartments=True):
        R"""
        Formats and saves predictions on BED format

        Args: 
            out_file (str, optional):
                Folder/Path to save the prediction results
            save_subcompartments (bool, optional):
                Whether generate files with subcompartment annotations 
            save_compartments (bool, optional):
                Whether generate files with compartment annotations
        
        Returns:
            predictions_subcompartments (dict), predictions_compartments (dict)
                Predicted subcompartment annotations and compartment annotations on dictionaries organized by chromosomes

        """
        def get_color(s_id):
            return info[s_id][1]

        def get_num(s_id):
            return str(info[s_id][0])

        def get_bed_file_line(chromosome, position, c_id):
            return "chr" + str(chromosome) + "\t" + str((position - 1) * resolution) + "\t" + str(
                position * resolution) + "\t" + c_id + "\t" + get_num(c_id) + "\t.\t" + str(
                (position - 1) * resolution) + "\t" + str(
                position * resolution) + "\t " + get_color(c_id)
        
        def save_bed(type):
            folder=self.cell_line_path+'/predictions'
            if type=='c':
                ext='_compartments'
            else:
                ext='_subcompartments'
            all_data = {}
            for chrom_index in range(1,len(self.chrm_size)):
                try:
                    filename = folder + '/chr' + str(chrom_index) + ext + '.txt'
                    data = []
                    with open(filename) as file:
                        for line in file:
                            items = line.split()
                            if len(items) == 2:
                                data.append((int(items[0]), items[1]))
                            else:
                                print(line)
                    all_data[chrom_index] = data
                except:
                    print('Didnt found chrom:',chrom_index)
            
            with open(out_file+ext+'.bed', 'w') as f:
                header='# Experiments used for this prediction: '
                for exp in exps:
                    header=header+exp+' '
                f.write(header+'\n')
                for chrom_index in range(1,len(self.chrm_size)):
                    try:
                        for (pos, sID) in all_data[chrom_index]:
                            f.write(get_bed_file_line(chrom_index, pos, sID) + '\n')
                    except:
                        pass            

        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)
        for u in unique:
            os.system('cat '+self.cell_line_path+'/'+u+'*/exp_name.txt | awk \'{print $1}\' >> '+self.cell_line_path+'/exps_used.dat')
        exps=np.loadtxt(self.cell_line_path+'/exps_used.dat',dtype=str)

        resolution=self.res*1000
        info = {
            "NA": (0, "255,255,255"),
            "A1": (2, "245,47,47"),
            "A2": (1, "145,47,47"),
            "B1": (-1, "47,187,224"),
            "B2": (-2, "47,47,224"),
            "B3": (-3, "47,47,139"),
            "B4": (-4, "75,0,130"),
            "A": (1, "245,47,47"),
            "B": (-1, "47,187,224"),
        }
        if compartments==True:
            save_bed('c')

        if subcompartments==True:
           save_bed('s')

    def prediction_all_chrm(self,path=None,save_subcompartments=True,save_compartments=True):
        R"""
        Predicts and outputs the genomic annotations for all the chromosomes

        Args: 
            path (str, optional):
                Folder/Path to save the prediction results
            save_subcompartments (bool, optional):
                Whether generate files with subcompartment annotations for each chromosomes
            save_compartments (bool, optional):
                Whether generate files with compartment annotations for each chromosomes
        
        Returns:
            predictions_subcompartments (dict), predictions_compartments (dict)
                Predicted subcompartment annotations and compartment annotations on dictionaries organized by chromosomes

        """
        if path==None: path=self.cell_line_path+'/predictions'
        print('Saving prediction in:',path)
        #Define translation dictionaries between states and subcompartments
        TYPE_TO_INT = {'A1':0,'A2':1,'B1':2,'B2':3,'B3':4,'B4':5,'NA':6}
        INT_TO_TYPE = {TYPE_TO_INT[k]:k for k in TYPE_TO_INT.keys()}
        INT_TO_TYPE_AB = {0:'A', 1:'A', 2:'B', 3:'B', 4:'B', 5:'B', 6:'NA'}

        os.system('mkdir '+path)
        #Predict and save data for chromosomes
        predictions_subcompartments={}
        predictions_compartments={}
        for chr in range(1,len(self.chrm_size)):
            pred=self.prediction_single_chrom(chr,h_and_J_file=self.cell_line_path+'/h_and_J.npy')
            types_pyME_sub=np.array(list(map(INT_TO_TYPE.get, pred)))
            types_pyME_AB=np.array(list(map(INT_TO_TYPE_AB.get, pred)))
            #Save data
            if save_subcompartments==True:
                with open(path+'/chr'+str(chr)+'_subcompartments.txt','w',encoding = 'utf-8') as f:
                    for i in range(len(types_pyME_sub)):
                        f.write("{} {}\n".format(i+1,types_pyME_sub[i]))
            if save_compartments==True:
                with open(path+'/chr'+str(chr)+'_compartments.txt','w',encoding = 'utf-8') as f:
                    for i in range(len(types_pyME_AB)):
                        f.write("{} {}\n".format(i+1,types_pyME_AB[i]))
            predictions_subcompartments[chr]=types_pyME_sub
            predictions_compartments[chr]=types_pyME_AB

        #Chromosome X
        chr='X'
        pred=self.prediction_X(h_and_J_file=self.cell_line_path+'/h_and_J.npy')
        types_pyME_sub=np.array(list(map(INT_TO_TYPE.get, pred)))
        types_pyME_AB=np.array(list(map(INT_TO_TYPE_AB.get, pred)))
        #Save data
        if save_subcompartments==True:
            with open(path+'/chr'+str(chr)+'_subcompartments.txt','w',encoding = 'utf-8') as f:
                for i in range(len(types_pyME_sub)):
                    f.write("{} {}\n".format(i+1,types_pyME_sub[i]))
                    
        if save_compartments==True:
            with open(path+'/chr'+str(chr)+'_compartments.txt','w',encoding = 'utf-8') as f:
                for i in range(len(types_pyME_AB)):
                    f.write("{} {}\n".format(i+1,types_pyME_AB[i]))     

        self.write_bed(out_file=path+'/predictions', compartments=save_compartments,subcompartments=save_subcompartments)

        return predictions_subcompartments, predictions_compartments

    def printHeader(self):
        print('{:^96s}'.format("****************************************************************************************"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("****************************************************************************************"))
        print('')
        print('{:^96s}'.format("The PyMEGABASE class performs the prediction of genomic annotations"))
        print('{:^96s}'.format("based on 1D data tracks of Chip-Seq and RNA-Seq. The input data is "))
        print('{:^96s}'.format("obtained from ENCODE data base."))
        print('{:^96s}'.format("PyMEGABASE description is described in: TBD"))
        print('')
        print('{:^96s}'.format("This package is the product of contributions from a number of people, including:"))
        print('{:^96s}'.format("Esteban Dodero-Rojas, Antonio Oliveira, Vinícius Contessoto,"))
        print('{:^96s}'.format("Ryan Cheng, and, Jose Onuchic"))
        print('{:^96s}'.format("Rice University"))
        print('')
        print('{:^96s}'.format("****************************************************************************************"))

class PyMEGABASE:
    R"""
    The :class:`~.PyMEGABASE` class performs genomic annotations .
    
    The :class:`~.PyMEGABASE` sets the environment to generate prediction of genomic annotations.
    
    Args:
        cell_line (str, required): 
            Name of target cell type
        assembly (str, required): 
            Reference assembly of target cell line ('hg19','GRCh38','mm10')
        organism (str, required): 
            Target cell type organism (str, required):
        signal_type (str, required): 
            Input files signal type ('signal p-value', 'fold change over control', ...)
        ref_cell_line_path (str, optional): 
            Folder/Path to place reference/training data ('tmp_meta')
        cell_line_path (str, optional): 
            Folder/Path to place target cell data 
        types_path (str, optional): 
            Folder/Path where the genomic annotations are located 
        histones (bool, required): 
            Whether to use Histone Modification data from the ENCODE databank for prediction
        tf (bool, required): 
            Whether to use Transcription Factor data from the ENCODE databank for prediction
        small_rna (bool, required): 
            Whether to use Small RNA-Seq data from the ENCODE databank for prediction
        total_rna (bool, required): 
            Whether to use Total RNA-seq data from the ENCODE databank for prediction
        n_states (int, optional): 
            Number of states for the D-nodes 
        extra_filter (str, optional):
            Add filter to the fetching data url to download cell type data
        res (int, optional):
            Resolution for genomic annotations calling in kilobasepairs (5, 50, 100)
        chromosome_sizes (list, optional):
            Chromosome sizes based on the reference genome assembly - required for non-human assemblies
        file_format (str, optional):
            File format for the input data
    """
    def __init__(self, cell_line='GM12878', assembly='hg19',organism='human',signal_type='signal p-value',file_format='bigWig',
                ref_cell_line_path='tmp_meta',cell_line_path=None,types_path=None,
                histones=True,tf=False,atac=False,small_rna=False,total_rna=False,n_states=19,
                extra_filter='',res=50,chromosome_sizes=None,AB=False):
        self.printHeader()
        pt = os.path.dirname(os.path.realpath(__file__))
        self.path_to_share = os.path.join(pt,'share/')
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
        if types_path!=None:
            self.types_path=types_path
        else:
            self.types_path=self.path_to_share+'types'
        self.hist=histones
        self.tf=tf
        self.atac=atac
        self.small_rna=small_rna
        self.total_rna=total_rna
        self.n_states=n_states
        self.extra_filter=extra_filter
        self.res=res
        self.organism=organism.lower()
        if file_format.lower()=='bigwig':self.file_format='bigWig'
        elif file_format.lower()=='bed': self.file_format='bed+narrowPeak'

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
            self.chrm_size = np.array([4980,4844,3966,3805,3631,3417,3187,2903,2768,2676,2702,2666,2288,2141,2040,1807,1666,1608,1173,1289,935,1017,3121])*50/self.res
            self.chrom_l={'chr1':248956422,'chr2':242193529,'chr3':198295559,'chr4':190214555,'chr5':181538259,'chr6':170805979,'chr7':159345973,'chrX':156040895,'chr8':145138636,'chr9':138394717,'chr11':135086622,'chr10':133797422,'chr12':133275309,'chr13':114364328,'chr14':107043718,'chr15':101991189,'chr16':90338345,'chr17':83257441,'chr18':80373285,'chr20':64444167,'chr19':58617616,'chrY':	57227415,'chr22':50818468,'chr21':46709983}
        elif assembly=='hg19':
            self.chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028,3105])*50/self.res
            self.chrom_l={'chr1':249250621,'chr2':243199373,'chr3':198022430,'chr4':191154276,'chr5':180915260,'chr6':171115067,'chr7':159138663,'chrX':155270560,'chr8':146364022,'chr9':141213431,'chr10':135534747,'chr11':135006516,'chr12':133851895,'chr13':115169878,'chr14':107349540,'chr15':102531392,'chr16':90354753,'chr17':81195210,'chr18':78077248,'chr20':63025520,'chrY':59373566,'chr19':59128983,'chr22':51304566,'chr21':48129895}
        else:
            if chromosome_sizes == None: 
                raise ValueError("Need to specify chromosome sizes for assembly: {}".format(assembly))
            self.chrm_size = np.array(chromosome_sizes)/(self.res*1000)
            self.chrom_l = np.array(chromosome_sizes)
        self.chrm_size=np.round(self.chrm_size+0.1).astype(int)
        self.ref_chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028,1028])*50/self.res
        self.ref_chrm_size=np.round(self.ref_chrm_size+0.1).astype(int)

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
        self.url_ref=url+'&biosample_ontology.term_name='+self.ref_cell_line+'&files.file_type='+self.file_format

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
                experiments.append('plus-small-RNA-seq')
            elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                experiments.append('plus-total-RNA-seq')          
            elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                experiments.append('minus-small-RNA-seq')
            elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                experiments.append('minus-total-RNA-seq')          

        self.experiments_unique=np.unique(experiments)
        self.es_unique=[]   
        for e in self.experiments_unique:
            self.es_unique.append(e.split('-human')[0])

        print('Selected cell line to predict: '+self.cell_line)
        print('Selected assembly: '+self.assembly)
        print('Selected signal type: '+self.signal_type)
        print('Selected organism: '+self.organism)

    def process_replica_bw(self,line,cell_line_path,chrm_size):
        R"""
        Preprocess function for each replica formated in bigwig files

        Args: 
            line (lsit, required):
                Information about the replica: name, ENCODE id and replica id
            cell_line_path (str, required):
                Path to target cell type data
            chrm_size (list, required):
                Chromosome sizes based on the assembly
        """
        text=line.split()[0]
        exp=line.split()[1]
        count=line.split()[2]
        sr_number=line.split()[3]

        #Experiment directory 
        exp_path=cell_line_path+'/'+exp+'_'+str(count)

        if 'human' in exp.split('-'): ext='human'
        else: ext=self.organism
        if exp.split('-'+ext)[0] in self.es_unique:
            try:
                os.mkdir(exp_path)
            except:
                print('Directory ',exp_path,' already exist')

            with open(exp_path+'/exp_name.txt', 'w') as f:
                f.write(text+' '+exp+'\n')
            with open(exp_path+'/exp_accession.txt', 'w') as f:
                f.write(sr_number+' '+exp+'\n')
                
            #Load data from server
            try:
                bw = pyBigWig.open("https://www.encodeproject.org/files/"+text+"/@@download/"+text+".bigWig")
                for chr in range(1,len(chrm_size)):
                    signal = bw.stats("chr"+str(chr), type="mean", nBins=chrm_size[chr-1])

                    #Process signal and binning 
                    signal=np.array(signal)
                    per=np.percentile(signal[signal!=None],95)
                    per_min=np.percentile(signal[signal!=None],5)
                    signal[signal==None]=per_min
                    signal[signal<per_min]=per_min
                    signal[signal>per]=per
                    signal=signal-per_min
                    signal=signal*self.n_states/(per-per_min)
                    signal=np.round(signal.astype(float)).astype(int)

                    #Save data
                    with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:

                        f.write("#chromosome file number of beads\n"+str(chrm_size[chr-1]))
                        f.write("#\n")
                        f.write("#bead, signal, discrete signal\n")
                        for i in range(len(signal)):
                            f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
                chr='X'
                signal = bw.stats("chr"+chr, type="mean", nBins=chrm_size[-1])
                #Process signal and binning
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)

                #Save data
                with open(exp_path+'/chr'+chr+'.track', 'w') as f:

                    f.write("#chromosome file number of beads\n"+str(chrm_size[-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
                return exp

            except:
                print('This experiment was incomplete:',text,'\nit will not be used.')

    def process_replica_bed(self,line,cell_line_path,chrm_size):
        R"""
        Preprocess function for each replica formated in bed files

        Args: 
            line (lsit, required):
                Information about the replica: name, ENCODE id and replica id
            cell_line_path (str, required):
                Path to target cell type data
            chrm_size (list, required):
                Chromosome sizes based on the assembly
        """
        text=line.split()[0]
        exp=line.split()[1]
        count=line.split()[2]
        sr_number=line.split()[3]

        #Experiment directory 
        exp_path=cell_line_path+'/'+exp+'_'+str(count)

        if 'human' in exp.split('-'): ext='human'
        else: ext=self.organism
        if exp.split('-'+ext)[0] in self.es_unique:
            try:
                os.mkdir(exp_path)
            except:
                print('Directory ',exp_path,' already exist')

            with open(exp_path+'/exp_name.txt', 'w') as f:
                f.write(text+' '+exp+'\n')
            with open(exp_path+'/exp_accession.txt', 'w') as f:
                f.write(sr_number+' '+exp+'\n')
            #Load data from server
            try:
                exp_url="https://www.encodeproject.org/files/"+text+"/@@download/"+text+".bed.gz"
                response = urllib.request.urlopen(exp_url)
                gunzip_response = gzip.GzipFile(fileobj=response)
                content = gunzip_response.read()
                data=np.array([i.split('\t') for i in content.decode().split('\n')[:-1]])

                for chr in range(1,len(chrm_size)):
                    chrm_data=data[data[:,0]=='chr'+str(chr)][:,[1,2,6]].astype(float)
                    signal=np.zeros(chrm_size[chr-1])
                    ndx_small=np.floor(chrm_data[:,1]/(self.res*1000)) == np.floor(chrm_data[:,0]/(self.res*1000))
                    # Aggregate peak intensity
                    for ll in chrm_data[ndx_small]:
                        ndx=int(ll[0]/(self.res*1000))
                        if ndx<len(signal):
                            signal[ndx]+=ll[2]
                    for ll in chrm_data[~ndx_small]:
                        ndx1=int(ll[0]/(self.res*1000))
                        ndx2=int(ll[1]/(self.res*1000))
                        if ndx1<len(signal) and ndx2<len(signal):
                            p=(ndx2-ll[0]/(self.res*1000))/((ll[1]-ll[0])/(self.res*1000))
                            signal[ndx1]+=ll[2]*p
                            signal[ndx2]+=ll[2]*(1-p)

                    per=np.percentile(signal[signal!=None],95)
                    per_min=np.percentile(signal[signal!=None],5)
                    signal[signal==None]=per_min
                    signal[signal<per_min]=per_min
                    signal[signal>per]=per
                    signal=signal-per_min
                    signal=signal*self.n_states/(per-per_min)
                    signal=np.round(signal.astype(float)).astype(int)
                    #Save data
                    with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:

                        f.write("#chromosome file number of beads\n"+str(chrm_size[chr-1]))
                        f.write("#\n")
                        f.write("#bead, signal, discrete signal\n")
                        for i in range(len(signal)):
                            f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")

                chr='X'
                chrm_data=data[data[:,0]=='chr'+chr][:,[1,2,6]].astype(float)
                signal=np.zeros(chrm_size[-1])
                ndx_small=np.floor(chrm_data[:,1]/(self.res*1000)) == np.floor(chrm_data[:,0]/(self.res*1000))
                # Aggregate peak intensity
                for ll in chrm_data[ndx_small]:
                    ndx=int(ll[0]/(self.res*1000))
                    if ndx<len(signal):
                        signal[ndx]+=ll[2]
                for ll in chrm_data[~ndx_small]:
                    ndx1=int(ll[0]/(self.res*1000))
                    ndx2=int(ll[1]/(self.res*1000))
                    if ndx1<len(signal) and ndx2<len(signal):
                        p=(ndx2-ll[0]/(self.res*1000))/((ll[1]-ll[0])/(self.res*1000))
                        signal[ndx1]+=ll[2]*p
                        signal[ndx2]+=ll[2]*(1-p)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)
                #Save data
                with open(exp_path+'/chr'+chr+'.track', 'w') as f:

                    f.write("#chromosome file number of beads\n"+str(chrm_size[-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
                return exp

            except:
                print('This experiment was incomplete:',text,'\nit will not be used.')

    def download_and_process_cell_line_data(self,nproc=10,all_exp=True):
        R"""
        Download and preprocess target cell data for the D-nodes

        Args: 
            nproc (int, required):
                Number of processors dedicated to download and process data
            all_exp (bool, optional):
                Download and process all replicas for each experiment
        """
        try:
            os.mkdir(self.cell_line_path)
        except:
            print('Directory ',self.cell_line_path,' already exist')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.cell_line_path)
            os.mkdir(self.cell_line_path)
        
        url='https://www.encodeproject.org/metadata/?type=Experiment&'+self.extra_filter
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
        self.url_cell_line=url+'&biosample_ontology.term_name='+self.cell_line+'&files.file_type='+self.file_format

        r = requests.get(self.url_cell_line)
        content=str(r.content)
        experiments=[]
        with open(self.cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                    f.write(l[0]+' '+l[7]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'plus-small-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'plus-total-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'minus-small-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'minus-total-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
       
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
                sr_number=line.split()[-1]

                #Register if experiment is new
                if exp!=exp_name:
                    try:
                        count=self.exp_found[exp]+1
                    except:
                        count=1
                    exp_name=exp
                self.exp_found[exp]=count
                if all_exp==True:
                    list_names.append(text+' '+exp+' '+str(count)+' '+sr_number)
                else:
                    if count==1:
                        list_names.append(text+' '+exp+' '+str(count)+' '+sr_number)

        print('Number of replicas:', len(list_names))
        if self.file_format=='bigWig':
            self.successful_exp = Parallel(n_jobs=nproc)(delayed(self.process_replica_bw)(list_names[i],self.cell_line_path,self.chrm_size) 
                                        for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        else:
            self.successful_exp = Parallel(n_jobs=nproc)(delayed(self.process_replica_bed)(list_names[i],self.cell_line_path,self.chrm_size) 
                                        for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        self.successful_exp= [i for i in self.successful_exp if i]
        self.successful_unique_exp=np.unique(self.successful_exp)
        self.su_unique=[]   
        for e in self.successful_unique_exp:
            self.su_unique.append(e.split('-'+self.organism)[0])
        print('Experiments found in ENCODE for the selected cell line:')
        self.unique=[]

        with open(self.cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.experiments_unique:
                if e.split('-human')[0] in self.su_unique:
                    f.write(e.split('-human')[0]+'\n')
                    print(e.split('-human')[0])
                    self.unique.append(e)
        if len(self.unique) > 4:
            print('Predictions would use: ',len(self.unique),' experiments')
        else:
            print('This sample only has ',len(self.unique),' experiments. We do not recommend prediction on samples with less than 5 different experiments.')

    def download_and_process_ref_data(self,nproc,all_exp=True):
        R"""
        Download and preprocess reference data for the D-nodes

        Args: 
            nproc (int, required):
                Number of processors dedicated to download and process data
            all_exp (bool, optional):
                Download and process all replicas for each experiment
        """
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
        self.url_ref=url+'&biosample_ontology.term_name='+self.ref_cell_line+'&files.file_type='+self.file_format

        r = requests.get(self.url_ref)
        content=str(r.content)
        experiments=[]
        with open(self.ref_cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                    f.write(l[0]+' '+l[7]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'plus-small-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'plus-total-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'minus-small-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'minus-total-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
        
        ref_chrm_size = self.ref_chrm_size
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
                sr_number=line.split()[-1]

                #Register if experiment is new
                if (exp.split('-human')[0] in self.su_unique) or (text.split('-human')[0] in self.su_unique):
                    if exp!=exp_name:
                        try:
                            count=exp_found[exp]+1
                        except:
                            count=1
                        exp_name=exp
                    exp_found[exp]=count
                    if all_exp==True:
                        list_names.append(text+' '+exp+' '+str(count)+' '+sr_number)
                    else:
                        if count==1:
                            list_names.append(text+' '+exp+' '+str(count)+' '+sr_number)

        print('Number of replicas:', len(list_names))

        if self.file_format=='bigWig':
            results = Parallel(n_jobs=nproc)(delayed(self.process_replica_bw)(list_names[i],self.ref_cell_line_path,ref_chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        else:
            results = Parallel(n_jobs=nproc)(delayed(self.process_replica_bed)(list_names[i],self.ref_cell_line_path,ref_chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        print('Prediction will use:')
        with open(self.ref_cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.unique:
                if e in exp_found.keys():
                    f.write(e.split('-'+self.organism)[0]+'\n')
                    print(e.split('-'+self.organism)[0])

    def custom_bw_track(self,experiment,bw_file):
        R"""
        Function to introduce custom bigwig tracks

        Args: 
            experiment (str, required):
                Name of the experiment
            bw_file (str, required):
                Path to the custom track
        """
        if not self.organism in experiment: experiment=experiment+'-'+self.organism
        if not experiment.split('-'+self.organism)[0] in self.es_unique: 
            print('This experiment is not found in the training set, then cannot be used.')
            return 0
        if experiment in self.exp_found.keys():
            print('This target has replicas already')
            print('The new track will be addded as a different replica of the same target')
            #Experiment directory
            count=self.exp_found[experiment]+1           
        else:
            print('This target has no replicas')
            print('The new track will be added a the first replica of the target')
            #Experiment directory
            count=1
        exp_path=self.cell_line_path+'/'+experiment+'_'+str(count)
        print(exp_path)
        
        try:
            os.mkdir(exp_path)
        except:
            print('Directory ',exp_path,' already exist')

        with open(exp_path+'/exp_name.txt', 'w') as f:
            f.write(experiment+' '+experiment+'\n')

        #Load data from track
        try:
            bw = pyBigWig.open(bw_file)
            for chr in range(1,len(self.chrm_size)):
                signal = bw.stats("chr"+str(chr), type="mean", nBins=self.chrm_size[chr-1])

                #Process signal and binning
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)

                #Save data
                with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:

                    f.write("#chromosome file number of beads\n"+str(self.chrm_size[chr-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
            chr='X'
            signal = bw.stats("chr"+chr, type="mean", nBins=self.chrm_size[-1])

            #Process signal and binning
            signal=np.array(signal)
            per=np.percentile(signal[signal!=None],95)
            per_min=np.percentile(signal[signal!=None],5)
            signal[signal==None]=per_min
            signal[signal<per_min]=per_min
            signal[signal>per]=per
            signal=signal-per_min
            signal=signal*self.n_states/(per-per_min)
            signal=np.round(signal.astype(float)).astype(int)

            #Save data
            with open(exp_path+'/chr'+chr+'.track', 'w') as f:

                f.write("#chromosome file number of beads\n"+str(self.chrm_size[-1]))
                f.write("#\n")
                f.write("#bead, signal, discrete signal\n")
                for i in range(len(signal)):
                    f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")

            if experiment in self.exp_found.keys():
                self.exp_found[experiment]=self.exp_found[experiment]+1

            else:
                self.exp_found[experiment]=1
                self.successful_unique_exp=np.append(self.successful_unique_exp,experiment)
                self.su_unique.append(experiment.split('-'+self.organism)[0])
                with open(self.cell_line_path+'/unique_exp.txt', 'a') as f:
                    f.write(experiment.split('-'+self.organism)[0]+'\n')
                    self.unique.append(experiment)

            return experiment
        
        except:
            print('This experiment was incomplete:',experiment,'\nit will not be used.')

    def custom_bed_track(self,experiment,bed_file):
        R"""
        Function to introduce custom bed tracks

        Args: 
            experiment (str, required):
                Name of the experiment
            bed_file (str, required):
                Path to the custom track
        """
        if not self.organism in experiment: experiment=experiment+'-'+self.organism
        if not experiment.split('-'+self.organism)[0] in self.es_unique: 
            print('This experiment is not found in the training set, then cannot be used.')
            return 0
        if experiment in self.exp_found.keys():
            print('This target has replicas already')
            print('The new track will be addded as a different replica of the same target')
            #Experiment directory
            count=self.exp_found[experiment]+1           
        else:
            print('This target has no replicas')
            print('The new track will be added a the first replica of the target')
            #Experiment directory
            count=1
        exp_path=self.cell_line_path+'/'+experiment+'_'+str(count)
        print(exp_path)
        
        try:
            os.mkdir(exp_path)
        except:
            print('Directory ',exp_path,' already exist')

        with open(exp_path+'/exp_name.txt', 'w') as f:
            f.write(experiment+' '+experiment+'\n')

        def get_records(bed_file):
            try:
                response = urllib.request.urlopen(bed_file)
                gunzip_response = gzip.GzipFile(fileobj=response)
                content = gunzip_response.read()
                data=np.array([i.split('\t') for i in content.decode().split('\n')[:-1]])
            except:
                try:
                    gunzip_response = gzip.GzipFile(bed_file)
                    content = gunzip_response.read()
                    data=np.array([i.split('\t') for i in content.decode().split('\n')[:-1]])
                except:
                    data=np.loadtxt(bed_file,dtype=str)
            return data
                

        #Load data from track
        try:
            data=get_records(bed_file)
            get_records(bed_file)
            for chr in range(1,len(self.chrm_size)):
                chrm_data=data[data[:,0]=='chr'+str(chr)][:,[1,2,6]].astype(float)
                signal=np.zeros(self.chrm_size[chr-1])
                ndx_small=np.floor(chrm_data[:,1]/(self.res*1000)) == np.floor(chrm_data[:,0]/(self.res*1000))
                # Aggregate peak intensity
                for ll in chrm_data[ndx_small]:
                    ndx=int(ll[0]/(self.res*1000))
                    if ndx<len(signal):
                        signal[ndx]+=ll[2]
                for ll in chrm_data[~ndx_small]:
                    ndx1=int(ll[0]/(self.res*1000))
                    ndx2=int(ll[1]/(self.res*1000))
                    if ndx1<len(signal) and ndx2<len(signal):
                        p=(ndx2-ll[0]/(self.res*1000))/((ll[1]-ll[0])/(self.res*1000))
                        signal[ndx1]+=ll[2]*p
                        signal[ndx2]+=ll[2]*(1-p)

                #Process signal and binning
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)

                #Save data
                with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:

                    f.write("#chromosome file number of beads\n"+str(self.chrm_size[chr-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
            chr='X'
            chrm_data=data[data[:,0]=='chr'+chr][:,[1,2,6]].astype(float)
            signal=np.zeros(self.chrm_size[-1])
            ndx_small=np.floor(chrm_data[:,1]/(self.res*1000)) == np.floor(chrm_data[:,0]/(self.res*1000))
            # Aggregate peak intensity
            for ll in chrm_data[ndx_small]:
                ndx=int(ll[0]/(self.res*1000))
                if ndx<len(signal):
                    signal[ndx]+=ll[2]
            for ll in chrm_data[~ndx_small]:
                ndx1=int(ll[0]/(self.res*1000))
                ndx2=int(ll[1]/(self.res*1000))
                if ndx1<len(signal) and ndx2<len(signal):
                    p=(ndx2-ll[0]/(self.res*1000))/((ll[1]-ll[0])/(self.res*1000))
                    signal[ndx1]+=ll[2]*p
                    signal[ndx2]+=ll[2]*(1-p)

            #Process signal and binning
            signal=np.array(signal)
            per=np.percentile(signal[signal!=None],95)
            per_min=np.percentile(signal[signal!=None],5)
            signal[signal==None]=per_min
            signal[signal<per_min]=per_min
            signal[signal>per]=per
            signal=signal-per_min
            signal=signal*self.n_states/(per-per_min)
            signal=np.round(signal.astype(float)).astype(int)

            #Save data
            with open(exp_path+'/chr'+chr+'.track', 'w') as f:

                f.write("#chromosome file number of beads\n"+str(self.chrm_size[-1]))
                f.write("#\n")
                f.write("#bead, signal, discrete signal\n")
                for i in range(len(signal)):
                    f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")

            if experiment in self.exp_found.keys():
                self.exp_found[experiment]=self.exp_found[experiment]+1

            else:
                self.exp_found[experiment]=1
                self.successful_unique_exp=np.append(self.successful_unique_exp,experiment)
                self.su_unique.append(experiment.split('-'+self.organism)[0])
                with open(self.cell_line_path+'/unique_exp.txt', 'a') as f:
                    f.write(experiment.split('-'+self.organism)[0]+'\n')
                    self.unique.append(experiment)

            return experiment
        
        except:
            print('This experiment was incomplete:',experiment,'\nit will not be used.')

    def build_state_vector(self,int_types,all_averages):
        R"""
        Builds the set of state vectors used on the training process

        Args: 
            int_types (list, required):
                Genomic annotations
            all_averages (list, required):
                D-node data 
        """
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

    def get_tmatrix(self,chrms,silent=False):
        R"""
        Extract the training data

        Args: 
            chrms (list, optional):
                Set of chromosomes from the reference data used as the training set
            silent (bool, optional):
                Silence outputs
        """
        #Load types from Rao et al 2014 paper
        types=[]
        for chr in chrms:
            if self.res==50:
                types.append(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str)[:,1])
            elif self.res==100:
                types.append(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str)[::int(self.res/50),1])
            else:
                tmp=list(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str))
                if len(tmp) < self.ref_chrm_size[chr-1]:
                    diff=self.ref_chrm_size[chr-1] - len(tmp)
                    for i in range( diff ):
                        tmp.append('NA')
                types.append(tmp)

        types=np.concatenate(types)
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))

        #Check which experiments are available to train 
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)
        if unique.shape==(): unique=[unique]
        if silent==False:print('To train the following experiments are used:')

        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            if silent==False:print(u)
            for i in glob.glob(self.ref_cell_line_path+'/'+str(u)+'*'):
                tmp=[]
                try:
                    for chr in chrms:
                        _tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                        tmp.append(_tmp)
                    tmp=np.concatenate(tmp)
                    reps.append(tmp)
                except:
                    if silent==False:print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.round(np.mean(reps,axis=0))
            all_averages.append(ave_reps)

        all_averages=np.array(all_averages)
        all_averages=self.build_state_vector(int_types,all_averages)
        return all_averages

    def filter_exp(self):
        R"""
        Performs baseline assestment on experiment baselines
        """
        a=[]
        for i in range(1,3):
            a.append(self.test_set(chr=i,silent=True))
        a=np.concatenate(a,axis=1)

        locus=2
        good_exp=0
        gexp=[]
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)

        for exper in range(len(unique)):
            i=exper+len(unique)*locus
            if (np.abs(np.mean(a[i])-np.mean(self.tmatrix[i+1]))<1) and (np.std(a[i])-np.std(self.tmatrix[i+1])<2):
                good_exp=good_exp+1
                gexp.append(unique[exper]+'\n')
            else:
                print('Not using '+unique[exper],' to predict')

        #gexp=sample(gexp,8)
        with open(self.cell_line_path+'/unique_exp_filtered.txt','w') as f:    
            for i in gexp:
                f.write(i)
            print('Number of suitable experiments for prediction:',good_exp)
        if good_exp>0:
            os.system('mv '+self.cell_line_path+'/unique_exp.txt '+self.cell_line_path+'/unique_exp_bu.txt')
            os.system('mv '+self.cell_line_path+'/unique_exp_filtered.txt '+self.cell_line_path+'/unique_exp.txt')
        else:
            print('There are no experiment suitable for the prediction')

    def training_set_up(self,chrms=None,filter=True):
        R"""
        Formats data to allow the training

        Args: 
            chrms (list, optional):
                Set of chromosomes from the reference data to use as the training set
            filter (bool, optional):
                Filter experiments based on the baseline
        """
        if chrms==None:
            # We are training in odd chromosomes data
            if self.cell_line=='GM12878' and self.assembly=='hg19':
                chrms=[1,3,5,7,9,11,13,15,17,19,21]
            else:
                chrms=[i for i in range(1,23)]
        
        if filter==True: 
            all_averages=self.get_tmatrix(chrms,silent=True)
            self.tmatrix=np.copy(all_averages)
            self.filter_exp()

        all_averages=self.get_tmatrix(chrms,silent=False)
        self.tmatrix=np.copy(all_averages)
        
        # Translate Potts states to sequences
        sequences=np.array(list(map(self.INT_TO_RES.get, all_averages.flatten()))).reshape(all_averages.shape)

        #Generate sequence file 
        with open(self.cell_line_path+"/sequences.fa",'w',encoding = 'utf-8') as f:
            for i in range(len(sequences.T)):
                f.write('>'+str(i).zfill(4)+'\n')
                f.write(''.join(sequences[:,i])+'\n')

    def training(self,nproc=10,lambda_h=100,lambda_J=100):
        R"""
        Performs the training of the Potts model based on the reference data

        Args: 
            nproc (int, required):
                Number of processors used to train
            lambda_h (bool, optional):
                Value for the intensity of the regularization value for the h energy term
            lambda_J (float, optional):
                Value for the intensity of the regularization value for the J energy term
        Returns:
            array (size of chromosome,5*number of unique experiments)
                D-node input data

        """
        # Compute DCA scores using Pseudolikelihood maximization algorithm
        plmdca_inst = plmdca.PlmDCA(
            self.cell_line_path+"/sequences.fa",
            'protein',
            seqid = 0.99,
            lambda_h = lambda_h,
            lambda_J = lambda_J,
            num_threads = nproc,
            max_iterations = 1000)
        print('Training started')
        # Train an get coupling and fields as lists
        self.fields_and_couplings = plmdca_inst.get_fields_and_couplings_from_backend()
        self.plmdca_inst=plmdca_inst
        fields_and_couplings = self.fields_and_couplings
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
        h_and_J={}
        h_and_J['h']=h
        h_and_J['J']=J
        h_and_J['J_flat']=couplings
        #Save fields and couplings 
        with open(self.cell_line_path+'/h_and_J.npy', 'wb') as f:
            np.save(f, h_and_J)

    def test_set(self,chr=1,silent=False):
        R"""
        Predicts and outputs the genomic annotations for chromosome X

        Args: 
            chr (int, required):
                Chromosome to extract input data fro the D-nodes
            silent (bool, optional):
                Avoid printing information 
        Returns:
            array (size of chromosome,5*number of unique experiments)
                D-node input data

        """
        if silent==False:print('Test set for chromosome: ',chr)        
        if chr!='X':
            types=["A1" for i in range(self.chrm_size[chr-1])]
        else:
            types=["A1" for i in range(self.chrm_size[-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str) 
        if unique.shape==(): unique=[unique]
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+str(u)+'*'):
                tmp=[]
                try:
                    tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                    reps.append(tmp)
                except:
                    if silent==False:print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.round(np.mean(reps,axis=0))
            all_averages.append(ave_reps)

        all_averages=np.array(all_averages)
        chr_averages=self.build_state_vector(int_types,all_averages)-1
        return chr_averages[1:]+1

    def prediction_single_chrom(self,chr=1,h_and_J_file=None,energies=False,probabilities=False):
        R"""
        Predicts and outputs the genomic annotations for chromosome X

        Args: 
            chr (int, optional):
                Chromosome to predict
            h_and_J_file (str, optional):
                Model energy term file path
        
        Returns:
            array (size of chromosome)
                Predicted annotations

        """
        print('Predicting subcompartments for chromosome: ',chr)       
        if h_and_J_file!=None:
            with open(h_and_J_file, 'rb') as f:
                h_and_J = np.load(f, allow_pickle=True)
                h_and_J = h_and_J.item()
            self.h=h_and_J['h']
            self.J=h_and_J['J']
 
        types=["A1" for i in range(self.chrm_size[chr-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str) 
        if unique.shape==(): unique=[unique]
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+str(u)+'*'):
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
        fails=0;r=0;
        self.L=len(self.h)
        enes=[]
        probs=[]
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
            enes.append(energy_val)
            probs.append(np.exp(-energy_val)/np.sum(np.exp(-energy_val)))
            #Select the state with the lowest energy
            predict_type[loci]=np.where(energy_val==np.min(energy_val))[0][0]
        enes=np.array(enes)
        probs=np.array(probs)
        #Add gaps from UCSC database
        try: 
            gaps=np.loadtxt(self.path_to_share+'/gaps/'+self.assembly+'_gaps.txt',dtype=str)
            chr_gaps_ndx=np.where((gaps[:,0]=='chr'+str(chr)))[0]
            for gp in chr_gaps_ndx:
                init_loci=np.round(gaps[gp,1].astype(float)/(self.res*1000)).astype(int)
                end_loci=np.round(gaps[gp,2].astype(float)/(self.res*1000)).astype(int)
                predict_type[init_loci:end_loci]=6
                enes[init_loci:end_loci]=0
                probs[init_loci:end_loci]=0
        except:
            print('Gaps not found, not included in predictions')
        if energies==True:
            if probabilities==True:
                return predict_type, enes, probs
            else:
                return predict_type, enes
        else:
            if probabilities==True:
                return predict_type, probs
            else:
                return predict_type

    def prediction_X(self,chr='X',h_and_J_file=None,energies=False,probabilities=False):
        R"""
        Predicts and outputs the genomic annotations for chromosome X

        Args: 
            chr (int, optional):
                Chromosome to predict
            h_and_J_file (str, optional):
                Model energy term file path
        
        Returns:
            array (size of chromosome)
                Predicted annotations

        """


        print('Predicting subcompartments for chromosome: ',chr)       
        if h_and_J_file!=None:
            with open(h_and_J_file, 'rb') as f:
                h_and_J = np.load(f, allow_pickle=True)
                h_and_J = h_and_J.item()
            self.h=h_and_J['h']
            self.J=h_and_J['J']
 
        types=["A1" for i in range(self.chrm_size[-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str) 
        if unique.shape==(): unique=[unique]
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+str(u)+'*'):
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
        self.L=len(self.h)
        enes=[]
        probs=[]
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
            enes.append(energy_val)
            probs.append(np.exp(-energy_val)/np.sum(np.exp(-energy_val)))
            #Select the state with the lowest energy
            predict_type[loci]=np.where(energy_val==np.min(energy_val))[0][0]
        enes=np.array(enes)
        probs=np.array(probs)
        #Add gaps from UCSC database
        try:
            print('Resolution:', self.res)
            gaps=np.loadtxt(self.path_to_share+'/gaps/'+self.assembly+'_gaps.txt',dtype=str)
            chr_gaps_ndx=np.where((gaps[:,0]=='chr'+str(chr)))[0]
            for gp in chr_gaps_ndx:
                init_loci=np.round(gaps[gp,1].astype(float)/self.res*1000).astype(int)
                end_loci=np.round(gaps[gp,2].astype(float)/self.res*1000).astype(int)
                predict_type[init_loci:end_loci]=6
                enes[init_loci:end_loci]=0
                probs[init_loci:end_loci]=0
        except:
            print('Gaps not found, not included in predictions')
        
        if energies==True:
            if probabilities==True:
                return predict_type, enes, probs
            else:
                return predict_type, enes
        else:
            if probabilities==True:
                return predict_type, probs
            else:
                return predict_type

    def write_bed(self,out_file='predictions', compartments=True,subcompartments=True):
        R"""
        Formats and saves predictions on BED format

        Args: 
            out_file (str, optional):
                Folder/Path to save the prediction results
            save_subcompartments (bool, optional):
                Whether generate files with subcompartment annotations 
            save_compartments (bool, optional):
                Whether generate files with compartment annotations
        
        Returns:
            predictions_subcompartments (dict), predictions_compartments (dict)
                Predicted subcompartment annotations and compartment annotations on dictionaries organized by chromosomes

        """
        def get_color(s_id):
            return info[s_id][1]

        def get_num(s_id):
            return str(info[s_id][0])

        def get_bed_file_line(chromosome, position, c_id):
            if self.chrom_l['chr'+str(chromosome)]<position*resolution:
                return "chr" + str(chromosome) + "\t" + str((position - 1) * resolution) + "\t" + str(
                    position * resolution) + "\t" + c_id + "\t" + get_num(c_id) + "\t.\t" + str(
                    (position - 1) * resolution) + "\t" + str(
                    position * resolution) + "\t " + get_color(c_id)
            else:
                if self.chrom_l['chr'+str(chromosome)]>(position-1)*resolution:
                    return "chr" + str(chromosome) + "\t" + str((position - 1) * resolution) + "\t" + str(
                        self.chrom_l['chr'+str(chromosome)]) + "\t" + c_id + "\t" + get_num(c_id) + "\t.\t" + str(
                        (position - 1) * resolution) + "\t" + str(
                        self.chrom_l['chr'+str(chromosome)]) + "\t " + get_color(c_id)
                else:
                    return ''
        def save_bed(type):
            folder=self.cell_line_path+'/predictions'
            if type=='c':
                ext='_compartments'
            else:
                ext='_subcompartments'
            all_data = {}
            for chrom_index in range(1,len(self.chrm_size)):
                try:
                    filename = folder + '/chr' + str(chrom_index) + ext + '.txt'
                    data = []
                    with open(filename) as file:
                        for line in file:
                            items = line.split()
                            if len(items) == 2:
                                data.append((int(items[0]), items[1]))
                            else:
                                print(line)
                    all_data[chrom_index] = data
                except:
                    print('Didnt found chrom:',chrom_index)
            
            with open(out_file+ext+'.bed', 'w') as f:
                header='# Experiments used for this prediction: '
                for exp in exps:
                    header=header+exp+' '
                f.write(header+'\n')
                for chrom_index in range(1,len(self.chrm_size)):
                    try:
                        for (pos, sID) in all_data[chrom_index]:
                            f.write(get_bed_file_line(chrom_index, pos, sID) + '\n')
                    except:
                        pass            

        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)
        for u in unique:
            os.system('cat '+self.cell_line_path+'/'+u+'*/exp_accession.txt | awk \'{print $1}\' >> '+self.cell_line_path+'/exps_used.dat')
        exps=np.loadtxt(self.cell_line_path+'/exps_used.dat',dtype=str)

        resolution=self.res*1000
        info = {
            "NA": (0, "255,255,255"),
            "A1": (2, "245,47,47"),
            "A2": (1, "145,47,47"),
            "B1": (-1, "47,187,224"),
            "B2": (-2, "47,47,224"),
            "B3": (-3, "47,47,139"),
            "B4": (-4, "75,0,130"),
            "A": (1, "245,47,47"),
            "B": (-1, "47,187,224"),
        }
        if compartments==True:
            save_bed('c')

        if subcompartments==True:
           save_bed('s')

    def prediction_all_chrm(self,path=None,save_subcompartments=True,save_compartments=True,energies=False,probabilities=False):
        R"""
        Predicts and outputs the genomic annotations for all the chromosomes

        Args: 
            path (str, optional):
                Folder/Path to save the prediction results
            save_subcompartments (bool, optional):
                Whether generate files with subcompartment annotations for each chromosomes
            save_compartments (bool, optional):
                Whether generate files with compartment annotations for each chromosomes
        
        Returns:
            predictions_subcompartments (dict), predictions_compartments (dict)
                Predicted subcompartment annotations and compartment annotations on dictionaries organized by chromosomes

        """
        if path==None: path=self.cell_line_path+'/predictions'
        print('Saving prediction in:',path)
        #Define translation dictionaries between states and subcompartments
        TYPE_TO_INT = {'A1':0,'A2':1,'B1':2,'B2':3,'B3':4,'B4':5,'NA':6}
        INT_TO_TYPE = {TYPE_TO_INT[k]:k for k in TYPE_TO_INT.keys()}
        INT_TO_TYPE_AB = {0:'A', 1:'A', 2:'B', 3:'B', 4:'B', 5:'B', 6:'NA'}

        os.system('mkdir '+path)
        #Predict and save data for chromosomes
        predictions_subcompartments={}
        predictions_compartments={}
        energies_chr={}
        probabilities_chr={}
        for chr in range(1,len(self.chrm_size)):
            if energies==True:
                if probabilities==True:
                    pred, enes, probs=self.prediction_single_chrom(chr,h_and_J_file=self.cell_line_path+'/h_and_J.npy',energies=True,probabilities=True)
                    energies_chr[chr]=enes
                    probabilities_chr[chr]=probs
                else:
                    pred, enes=self.prediction_single_chrom(chr,h_and_J_file=self.cell_line_path+'/h_and_J.npy',energies=True,probabilities=False)
                    energies_chr[chr]=enes
            else:
                if probabilities==True:
                    pred, probs=self.prediction_single_chrom(chr,h_and_J_file=self.cell_line_path+'/h_and_J.npy',energies=False,probabilities=True)
                    probabilities_chr[chr]=probs
                else:
                    pred=self.prediction_single_chrom(chr,h_and_J_file=self.cell_line_path+'/h_and_J.npy',energies=False,probabilities=False)
            types_pyME_sub=np.array(list(map(INT_TO_TYPE.get, pred)))
            types_pyME_AB=np.array(list(map(INT_TO_TYPE_AB.get, pred)))
            #Save data
            if save_subcompartments==True:
                with open(path+'/chr'+str(chr)+'_subcompartments.txt','w',encoding = 'utf-8') as f:
                    for i in range(len(types_pyME_sub)):
                        f.write("{} {}\n".format(i+1,types_pyME_sub[i]))
            if save_compartments==True:
                with open(path+'/chr'+str(chr)+'_compartments.txt','w',encoding = 'utf-8') as f:
                    for i in range(len(types_pyME_AB)):
                        f.write("{} {}\n".format(i+1,types_pyME_AB[i]))
            predictions_subcompartments[chr]=types_pyME_sub
            predictions_compartments[chr]=types_pyME_AB

        #Chromosome X
        chr='X'
        pred=self.prediction_X(h_and_J_file=self.cell_line_path+'/h_and_J.npy')
        types_pyME_sub=np.array(list(map(INT_TO_TYPE.get, pred)))
        types_pyME_AB=np.array(list(map(INT_TO_TYPE_AB.get, pred)))
        #Save data
        if save_subcompartments==True:
            with open(path+'/chr'+str(chr)+'_subcompartments.txt','w',encoding = 'utf-8') as f:
                for i in range(len(types_pyME_sub)):
                    f.write("{} {}\n".format(i+1,types_pyME_sub[i]))
                    
        if save_compartments==True:
            with open(path+'/chr'+str(chr)+'_compartments.txt','w',encoding = 'utf-8') as f:
                for i in range(len(types_pyME_AB)):
                    f.write("{} {}\n".format(i+1,types_pyME_AB[i]))     

        self.write_bed(out_file=path+'/predictions', compartments=save_compartments,subcompartments=save_subcompartments)
        if energies==True:
            if probabilities==True:
                return predictions_subcompartments, predictions_compartments, energies_chr, probabilities_chr
            else:
                return predictions_subcompartments, predictions_compartments, energies_chr
        else:
            if probabilities==True:
                return predictions_subcompartments, predictions_compartments, probabilities_chr
            else:
                return predictions_subcompartments, predictions_compartments
        

    def printHeader(self):
        print('{:^96s}'.format("****************************************************************************************"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** PyMEGABASE-1.0.0 *** *** *** *** *** *** *** ****"))
        print('{:^96s}'.format("****************************************************************************************"))
        print('')
        print('{:^96s}'.format("The PyMEGABASE class performs the prediction of genomic annotations"))
        print('{:^96s}'.format("based on 1D data tracks of Chip-Seq and RNA-Seq. The input data is "))
        print('{:^96s}'.format("obtained from ENCODE data base."))
        print('{:^96s}'.format("PyMEGABASE description is described in: TBD"))
        print('')
        print('{:^96s}'.format("This package is the product of contributions from a number of people, including:"))
        print('{:^96s}'.format("Esteban Dodero-Rojas, Antonio Oliveira, Vinícius Contessoto,"))
        print('{:^96s}'.format("Ryan Cheng, and, Jose Onuchic"))
        print('{:^96s}'.format("Rice University"))
        print('')
        print('{:^96s}'.format("****************************************************************************************"))
