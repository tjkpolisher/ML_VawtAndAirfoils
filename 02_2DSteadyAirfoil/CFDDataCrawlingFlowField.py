#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os, sys
from os import path
import numpy as np
import pandas as pd
import shutil
from distutils.dir_util import copy_tree
import string


# ### 파일 복사하기
# ```Python
# shutil.copyfile("./test1/test1.txt", "./test2.txt")
# shutil.copy("./test1/test1.txt", "./test3.txt")
# shutil.copy2("./test1/test1.txt", "./test4.txt")
# ```
# 1. copyfile과 copy는 메타정보는 복사되지 않습니다.  
# 2. copy2는 메타정보도 복사합니다. 즉, copy2를 사용하면 파일을 작성한 날짜도 복사되지만 copyfile과 copy는 파일을 작성한 날짜가 복사한 날짜로 변경됩니다.  
# 3. 사용 방법
# ```Python
# shutil.copy("원래 파일 경로", "복사할 폴더 경로")
# ```

# ### 폴더 복사하기
# 1. 폴더를 새로 생성하면서 복사할 경우
# ```Python
# import shutil
# shutil.copytree("./test1", "./test2")
# ```
# 
# 2. 이미 존재하는 폴더에 복사를 하고 싶은 경우  
# ```Python
# from distutils.dir_util import copy_tree
# copy_tree("./test1", "./test2")
# ```

# In[ ]:


os.chdir("C:\\Users\\cfdML\\Documents")


# In[ ]:


#os.chdir()
if not path.isdir("AirfoilClCdCoordinates"):
    os.makedirs("AirfoilClCdCoordinates")

origin = "D:\\airfoilFluent"
origin_data = "D:\\airfoilFluent\\airfoilSimulations"
origin_coord = "D:\\airfoilFluent\\airfoilCoordinates"
copy = "C:\\Users\\cfdML\\Documents\\AirfoilClCdCoordinates"

folders = os.listdir(origin_data)

airfoil_list_name = []
airfoil_list_index = []


counter = 1 # Numbering of the airfoils (initial value)

for i in folders:
    os.chdir(copy)
    copy_folderAirfoilName = copy + "\\airfoil" + str(counter) # the name of the airfoil in current iteration
    
    if not path.isdir(copy_folderAirfoilName): # Making folder for each airfoils
        os.makedirs(copy_folderAirfoilName)
    
    # Making folders for each angle of attack (total 16ea.)
    for j in range(0, 16):
        os.chdir(copy_folderAirfoilName)
        if j<=10:
            alpha = 2*j
        else:
            alpha = -2*j+20
        if not path.exists(str(alpha)):
            os.makedirs(str(alpha))
        
        # Copying flow field data of each airfoils
        os.chdir(origin_data+"\\"+str(i)+"\\"+str(i)+"_files\\user_files")
        
        ## List of .out files in the current folder
        file_list = os.listdir()
        file_out_list = [file for file in file_list if file.endswith('.csv')]
        
        if not path.exists(copy_folderAirfoilName+"\\" + str(i) + "alpha" + str(alpha) + ".csv"):
            shutil.copy2(origin_data+"\\"+str(i)+"\\"+str(i)+"_files\\user_files\\" + str(i) + "alpha" + str(alpha) + ".csv",
                         copy_folderAirfoilName+"\\" + str(i) + "alpha" + str(alpha) + ".csv")
    
    # Write in airfoil list
    airfoil_list_index.append("airfoil" + str(counter)) # airfoil numbering
    airfoil_list_name.append(str(i)) # the name of the airfoil

    counter += 1 


# In[ ]:

#airfoil_index = np.array(airfoil_list_index).reshape((-1,1))
#airfoil_name = np.array(airfoil_list_name).reshape((-1,1))
'''airfoil_lists = pd.DataFrame()
airfoil_lists['Index'] = airfoil_list_index
airfoil_lists['Name'] = airfoil_list_name
os.chdir(copy)
airfoil_lists.to_excel("AirfoilIndexList.xlsx", index=False)'''