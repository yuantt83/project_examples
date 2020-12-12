Synopsis:

This module aims to assist with file-folder related chores.  I want to give minimal instructions to an assitant (e.g., a folder, and the features  of the filenames that I want to find), then the assitant will go and find those exact files, and do some simple organising, like, 1) put the files in a new folder, 2) delete the files, 3) do something convenient about the files. 

The steps of using this module thus follow the line of thought above. 
Think of it as a robot. To instantiate the bot, give it two initial settings: 
a). The folder you want it to work on.
b). An exact description of the kind of files you want to find (use regex as the descriptive words)

The bot can then do the following tasks auotmatically (when you specify the task/method):
M0. Locate and show those files for you (the .locate method). 

After the .locate method, it will remember those files and do more, like
M1. copy those files to another folder (the .copy_to_folder method).

M2. Clean up those files by deleting them (the .clean method).

This robot currently specialises in pdf files, hence the suffix of \_pdf in its name.
If the located files are pdfs,  the bot can do two more things:
M3. [method .merge_pdf_tree] It can merge pdfs in their subfolders and put them in their subfolders. An example is that you have 
30 books (folders), each folder contains 10 chapters (pdfs).  The bot will bind the chapters and produce
a merged pdf for each book and put them in their own folders. 
M4. [method .merge_pdf_all]  It can merge all located pdfs, no questions asked, period.


This folder contains:
README
file_bot_pdf.ipynb: Jupyter notebook explaining how to import the module and use the class/methods
file_bot_pdf.py: python code module

Tests:
Examples of how to import and test the code can be found in the Jupyter notebook.

Note:
The robot is likely to acquire other specialities as time goes by, so watch for version update! Or seriously, Siri should be able to do  this already (why hasn't it ?!). 

Bug report to: Dr Tiantian Yuan
www.linkedin.com/in/tiantianyuan                                                                     
