#!/usr/bin/env python
# coding: utf-8

import re
import os
import sys
import random
import string
import time
import numpy as np
from PyPDF2 import PdfFileMerger    
from datetime import datetime
from functools import wraps

def etimer(func):
    """ A decorator that estimates how long a function runs    
    Args :
     func(callable): a function 
    Returns :
     the decorated function (callable)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()
        print('{} running time is {:.2f}s'.format(func.__name__, (t2 - t1)))
    return wrapper


class FileBots:
    """ 
    A class to search for files and combine pdfs hierarchically.
    ...
    Attributes
    ----------
    dir_in: str
        an input directory under which files are searched.
    reg_pattern: str
        a regular expression to locate filenames of interest.
    
    Methods
    ----------
    locate(show=True) :
       Locates and (prints) the matched path and filenames.
       
    pdf_merge_tree(dir_out=None, out_filename=None, ovewrite_all=False) :   
       Merge pdfs in the subfolder(s) that they are located separately.
       
    pdf_merge_all(dir_out=None) :
       Merge all pdfs that are found by the .locate method.
    
    clean(warning=True)
      Delete files found by the .locate method.
   
    Examples
    --------
    # When used as a module:
    # make sure file_bot_pdf.py is in your current dir or in sys.path
    
    dir_in = '/Users/smallpandas/learn_py/DS_courses'  # directory to search
    reg_pattern = r'.*_tree_merged.pdf'  # regex pattern for filenames
    bot = fb.FileBots(dir_in, reg_pattern) # instantiate a bot object
    bot.locate(show=False)  # locate files
    bot.pdf_merge_tree()  # merge pdfs found in the last leaf subdirectory
    bot.pdf_merge_all()     # merge all pdf files found

    Bug report to Dr Tiantian Yuan (www.linkedin.com/in/tiantianyuan)
    """
    
    def __init__(self, dir_in, reg_pattern):
        """
        Construct required attributes and check for errors.
        
        Parameters
        ----------
        dir_in : str
            an input directory under which files are searched.
        reg_pattern : str
        a regular expression to locate filenames of interest.        
        """
        
        if not isinstance(dir_in, str):
            raise ValueError('Directory path must be string!')
        if dir_in.endswith('/'):
            dir_in = dir_in[:-1]
        self.dir_in = dir_in
        if not isinstance(reg_pattern, str):
            raise ValueError('Search pattern should be a regex string!')
        self.pattern = re.compile(reg_pattern) 
    
    # a time decorator 
    @etimer    
    def locate(self, show=True):
        """ 
        Locates the matched path and filenames.
        
        Parameters
        ----------
        show : boolean, optional
            print out the path and filenames for checking (default is True).
        """
        
        self.show = show
        file_dict = []
        dir_dict = []        
        for rootdir, subdirs, files in os.walk(self.dir_in):            
            for file in files:               
                # match file with reg_pattern
                matches = self.pattern.findall(file)                
                if matches:   
                    file_location = ''.join(string for string in matches[0] if len(string) > 0)
                    # collect each matched file in a list
                    file_dict.append(file_location)
                    # collect the path of the matched file in a list
                    dir_dict.append(rootdir + os.sep) 

        # store all matched files and their paths as pairs
        np_path_file = np.array(list(zip(dir_dict, file_dict)))
        self.np_path_file = np_path_file
        
        # find the shared paths of matched files.         
        np_myset = np.array(list(set(dir_dict)))
        self.np_myset = np_myset

        if self.show == True:
            for i, j in np_path_file:
                print('{} (path->file): {} -> {}'.format(sys._getframe().f_code.co_name, i, j))
                
    @etimer           
    def pdf_merge_tree(self, dir_out=None, out_filename=None, ovewrite_all=False):
        """
        Merge the located pdfs according to the folder tree structure. 
            For example, in subfolder A, a1.pdf, a2.pdf, ..., a7.pdfs are found,
            then merge a1...a7.pdf as out_filename.pdf and store them in dir_out.
            Repeat this for all subfolders under dir_in.
        
        Parameters
        ----------
        dir_out : str, optional
            The directory to write the output pdf file(s). 
            Default is None. In this case, output files are written in the last
                leaf folder where they are merged.      
        out_filename : str, optional
            The output filename. 
            Default is None. In this case, filename (*_tree_merged.pdf) is automatically 
                generated with * composed of datetime and random characters. 
        ovewrite_all: boolean, optional
            Whether or not to overwrite existing files with the same name as out_filename.
            Default is False. In this case, prefix '_' will be added to new file to avoid 
                duplicate files.  
        """

        if (not 'np_myset' in self.__dict__) or (not 'np_path_file' in self.__dict__):
            raise AttributeError('Use method .locate to get files first!')
            
        if (dir_out != None) and (dir_out.endswith('/')):
            dir_out = dir_out[:-1]
            
        random.seed(30)         
        for path_uni in self.np_myset:            
            # initiate a merger for each folder where pdfs are merged
            merger = PdfFileMerger(strict=False)
            file_match = (self.np_path_file[:, 0] == path_uni)
            matched = self.np_path_file[file_match, 1]
            # sort alphabetically
            matched_sorted = np.sort(matched)
            
            # only merge when more than one matched pdf file 
            if sum(file_match) <= 1:
                if self.show == True:
                    print('Less equal than 1 pdf to merge under this path: (do nothing)', path_uni)
            else:
                for file in matched_sorted:
                    full_file = path_uni + file
                    merger.append(full_file, pages=None)

                if (out_filename != None) and (os.path.isfile(path_uni + out_filename)):
                    if ovewrite_all == True:
                        os.remove(path_uni + out_filename)
                    else:
                        write_filename = '_' + out_filename

                # take care of the output filenames and overwrites 
                if out_filename == None:
                    # generate a filename consists of current time + random charac
                    rand_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 5)) 
                    timetag = datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')[:-3]
                    write_filename = timetag + '_' + rand_name + '_tree_merged.pdf'

                if dir_out == None:
                    merger.write(path_uni + write_filename)
                else:
                    merger.write(dir_out + os.sep + write_filename) 
                merger.close()
        print(f'Done! Merged {self.np_path_file.shape[0]} pdfs in {len(self.np_myset)} subfolders')    

        
    @etimer
    def pdf_merge_all(self, dir_out=None, out_filename=None):
        """
        Merge all pdfs found in the .locate method as one pdf and store it in dir_out.
        
        Parameters
        ----------
        dir_out: str, optional
            The directory to write the output pdf.  
            Default is None. In this case, dir_out = dir_in. 
            
        out_filename: str, optional
            The name for the output pdf. 
            Default is None. In this case, out_filename = datetime+random_character+'_all_merged.pdf'.
        """
        
        if (dir_out != None) and (dir_out.endswith('/')):
            dir_out = dir_out[:-1]

        if out_filename == None:
            rand_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 3)) 
            timetag = datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')[:-3]
            write_filename = timetag + '_' + rand_name + '_all_merged.pdf'
        else:
            write_filename = out_filename

        merger = PdfFileMerger(strict=False)
        for i, j in self.np_path_file:
            full_file = i + j
            merger.append(full_file, pages=None)

        if dir_out == None:
            merger.write(self.dir_in + os.sep + write_filename)
        elif isinstance(dir_out, str):
            merger.write(dir_out + os.sep + write_filename)
        else:
            print('Nothing to wrote ...')
        merger.close()
        print(f'Done! Merged {self.np_path_file.shape[0]} pdfs!')    
           
    @etimer
    def clean(self, warning=True):
        """ 
        Delete files found by the .locate method.
            
        Parameters
        ----------
        warning: boolean, optional
            Default is True (recommended!). In this case, a caution message will be prompted showing
            files to be deleted. The user will input yes or no at the console to confirm.
        """
        if (not 'np_myset' in self.__dict__) or \
            (not 'np_path_file' in self.__dict__):
            raise AttributeError('Use method .locate to get files first!')
    
        for path_file, file in self.np_path_file:
            print(path_file, '-->', file)
        
        if warning == True:       
            warn_str = 'Are you sure you want to delete all the following files?! \
                       Type yes or no here: '
            doit = input(warn_str)
            if doit == 'yes':
                for path_file, file in self.np_path_file:
                    os.remove(path_file + file) 
            else:
                print('OK, bot will not remove any file.')
        else:
            for path_file, file in self.np_path_file:
                os.remove(path_file + file) 
                print('Files removed with your permission!')