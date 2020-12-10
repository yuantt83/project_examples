Synopsis:

This module aims to assist with file-folder related tasks. Imagine telling a robot two things:
here is a folder, and here is the kind of files you want (instantiate the bot). It will then auotmatically locate and show those
files for you (the .locate method). It can also clean up those files by deleting them (the .clean method).   

This robot currently specialises in pdf files, hence the suffix of \_pdf in its name.
If the located files are pdfs,  the bot can do two more things:
1. [method .merge_pdf_tree] It can merge pdfs in their subfolders and put them in their subfolders. An example is that you have 
30 books (folders), each folder contains 10 chapters (pdfs).  The bot will bind the chapters and produce
a merged pdf for each book and put them in their own folders. 
2. [method .merge_pdf_all]  It can merge all located pdfs, no questions asked, period.


This folder contains:
README
file_bot_pdf.ipynb: Jupyter notebook explaining how to import the module and use the class/methods
file_bot_pdf.py: python code module

Tests:
Examples of how to import and test the code can be found in the Jupyter notebook.

Note:
The robot is likely to acquire other specialities as time goes by, so watch for version update! Or seriously, Siri should be able to do 
this already (why hasn't it ?!). 

Bug report to: Dr Tiantian Yuan
www.linkedin.com/in/tiantianyuan                                                                     
