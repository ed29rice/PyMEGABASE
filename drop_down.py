import numpy as np
import urllib, requests
import ipywidgets as widgets

class cell_lines:
    def __init__(self, assembly='hg19',signal_type='signal p-value',
    histones=True,tf=False,atac=False,small_rna=False,total_rna=False):
        self.assembly=assembly  
        self.signal_type=signal_type
        self.hist=histones
        self.tf=tf
        self.atac=atac
        self.small_rna=small_rna
        self.total_rna=total_rna

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

        url=url+'&files.file_type=bigWig&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens'
        

        print('Looking for all available cell lines')
        r = requests.get(url)
        content=str(r.content)
        experiments=[]
        for k in content.split('\\n')[:-1]:
            l=k.split('\\t')
        
        cell_lines=[]
        cell_lines_url=[]
        for k in content.split('\\n')[1:-1]:
            l=k.split('\\t')
            if l[5]==self.assembly and l[4]==self.signal_type:
                name=l[10]
                cell_lines.append(name)
        cell_lines=np.unique(cell_lines)
        
        self.menu=widgets.Dropdown(
            options=cell_lines,
            description='Cell line:',
            disabled=False)        
        

