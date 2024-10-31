# -*- coding: utf-8 -*-

"""
##################################
# 
# Author: Dr. Sofia Gil-Clavel
# 
# Last update: October 31st, 2024.
# 
# Description: Main functions used in the articles:
#   - Gil-Clavel, Sofia, and Tatiana Filatova. “Using Natural Language Processing
#       and Networks to Automate Structured Literature Reviews: An Application to 
#       Farmers Climate Change Adaptation.” arXiv, July 3, 2024. 
#       https://arxiv.org/abs/2306.09737v2.
#   - Gil-Clavel, S., Wagenblast, T., Akkerman, J., & Filatova, T. (2024, April 26). 
#       Patterns in Reported Adaptation Constraints: Insights from Peer-Reviewed 
#       Literature on Flood and Sea-Level Rise. https://doi.org/10.31235/osf.io/3cqvn
#   - Gil-Clavel, S., Wagenblast, T., & Filatova, T. (2023, November 24). Incremental
#       and Transformational Climate Change Adaptation Factors in Agriculture Worldwide:
#       A Natural Language Processing Comparative Analysis. 
#       https://doi.org/10.31235/osf.io/3dp5e
# 
# Computer Environment:
#   - Windows 
#   - Microsoft Windows 10 Enterprise
#   - Python 3.11
# 
# Conda Environment to run the code:
#   - @SofiaG1L/NLP4LitRev/PY_ENVIRONMENT/pytorch_textacy.yml
#
# Conda Environments that have to be installed in the computer:
#   - @SofiaG1L/NLP4LitRev/PY_ENVIRONMENTS/EXTRA/neuralcoref2.yml
#   - @SofiaG1L/NLP4LitRev/PY_ENVIRONMENTS/EXTRA/scispacy.yml
# 
##################################
"""

import os as os

# Data Handling
import pandas as pd

import numpy as np
from collections import Counter
import subprocess

# Processing text
import regex as re
# from bs4 import BeautifulSoup
import itertools as it

# Import Text Processing Libraries
# import textacy as txt
import spacy as spacy
from spacy.tokenizer import Tokenizer

# =============================================================================
# Function to extract only words
# =============================================================================
def FriendOfFriend(S1,S2):
    if S1=="+/-" or S2=="+/-":
        if S1=="+/-":
            return(S2)
        if S2=="+/-":
            return(S1)
    elif S1==S2:
        return("+")
    elif S1!=S2:
        return("-")
    else:
        return("")


def tokenize(text):
    if pd.isna(text):
        return ""
    else:
        if type(text)==list:
            text=text[0]
        text=text.replace("-\n","")
        text=re.findall(r'[\w-]*\p{L}[\w-]*', text)
        text=[te for te in text if len(te)>2]
        return ' '.join(text)

# def extract_entities(doc, include_types=None, sep='_'):
#     ents = txt.extract.entities(doc,
#              include_types=include_types,
#              exclude_types=None,
#              drop_determiners=True,
#              min_freq=1)
#     return [(sep.join([str(t) for t in e]),e.label_) for e in ents] # .lemma_ 


'''
# =============================================================================
# Find Countries Based on Entitites and Database
# =============================================================================
'''
def CheckSubCountry(W,CITIES):
    IND=np.where([len(jj)>0 for jj in 
        [re.findall(r'\b' + W + r'\b',str(jj)) for jj in list(CITIES.subcountry)]])
    return(CITIES.iloc[IND])

def CheckCities(W,CITIES):
    IND=np.where([len(jj)>0 for jj in 
        [re.findall(r'\b' + W + r'\b',str(jj)) for jj in list(CITIES.name)]])
    return(CITIES.iloc[IND])

def ExtractCountryGPE(place,CITIES,DICT_PLC={}):
    place=place.lower()
    TIT1=[jj for jj in np.unique(CITIES.country).tolist() if jj.find(place)>-1]
    if(len(TIT1)>0):
        SUB=Counter(TIT1).most_common(2)
        if len(SUB)==1:
            return(SUB[0][0])  
        else: 
            if place==SUB[0][0]:
                return(SUB[0][0])
            elif place==SUB[1][0]:
                return(SUB[1][0])
            else:
                if SUB[0][1]>SUB[1][1]:
                    return(SUB[0][0])  
    else:
        if place in list(DICT_PLC.keys()): # If the country is in the dictionary
            return(DICT_PLC[place])
        else:
            TIT1=CheckSubCountry(place,CITIES)
            if(len(TIT1)>0):
                SUB=Counter(TIT1.country).most_common(2)
                if len(SUB)==1:
                    return(SUB[0][0])  
                else: 
                    if place==SUB[0][0]:
                        return(SUB[0][0])
                    elif place==SUB[1][0]:
                        return(SUB[1][0])
                    else:
                        if SUB[0][1]>SUB[1][1]:
                            return(SUB[0][0])   
            else:
                TIT1=CheckCities(place,CITIES)
                if(len(TIT1)>0):
                    SUB=Counter(TIT1.country).most_common(2)
                    if len(SUB)==1:
                        return(SUB[0][0])  
                    else: 
                        if place==SUB[0][0]:
                            return(SUB[0][0])
                        elif place==SUB[1][0]:
                            return(SUB[1][0])
                        else:
                            if SUB[0][1]>SUB[1][1]:
                                return(SUB[0][0])  

def GPEvsNationality(x,CITIES,NATIONALITIES,DICT_PLC={}):
    if x[1] in ["GPE","LOC"]:
        return(ExtractCountryGPE(x[0],CITIES,DICT_PLC))
    else:
        try:
            return(NATIONALITIES["Official Country Name"].loc[x[0]])
        except:
            return(None)

def ExtractCountry(x,CITIES,NATIONALITIES,DICT_PLC={},TYPE=["GPE","LOC","NORP"]):
    if type(x)==list:
        STD0=[jj for jj in x if jj[1] in TYPE]
        STD0=[(re.sub('[\(\)_]+', '', jj[0]).strip(),jj[1]) for jj in STD0]
        STD0=[GPEvsNationality(jj,CITIES,NATIONALITIES,DICT_PLC) for jj in STD0]
        STD0=[x for x in STD0 if not pd.isna(x)]
        return(STD0)
    else:
        return([])

def get_continent(col):
    text=os.popen('conda run -n scispacy python ".\\ConvertCountries.py" "'+col+'"').read()
    if len(text)<=9:
        text=re.sub("\n", "", text)
        text=text.split(",")
        return([x.strip() for x in text])
    else:
        return(["Unknown","Unknown"])

### Functions to clean text ###
def ReplaceAbre(TXT,ABR): 
    for ii in range(ABR.shape[0]):
        # break
        abr=ABR.iloc[ii].ABR
        mnn=ABR.iloc[ii].DEF
        TXT=re.sub(r'\('+abr+'\)',"",TXT)
        TXT=re.sub(r'(?:^|\W)'+abr+'(?:$|\W)'," "+mnn+" ",TXT)
    return(TXT)

def CleanText(tt,DIR="C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\",
              DIR_main="C:\\Dropbox\\TU_Delft\\Projects\\ML_FindingsGrammar\\CODE\\Processing_PDFs\\"):
    new_text2=""
    with open(DIR+"Temporal.txt", "w", encoding='utf-8') as text_file:
        text_file.write(tt)
    # Replacing abreviations with meanings
    # subprocess.run('conda run -n scispacy python "'+DIR_main+"FindAbreviations.py"+'" "'+DIR+"Temporal.txt"+'"',
    #                shell=True, check=True)
    subprocess.run('conda run -n scispacy python "'+DIR_main+"FindAbreviations.py"+\
                   '" "'+DIR+"Temporal.txt"+'" "'+DIR+"TemporalABR.csv"+'"',
                   shell=True, check=True)
    ABR=pd.read_csv(DIR+"TemporalABR.csv")
    
    TXT=ReplaceAbre(tt,ABR)
    for tt2 in TXT.split("\n"):
        # break
        # Removing anything between parentheses and the parentheses
        text=re.sub("[\(\[].*?[\)\]]", "", tt2)
        
        with open(DIR+"Temporal.txt", "w", encoding='utf-8') as text_file:
            text_file.write(text)
        
        # Replacing pronouns with objects removed: neuralcoref2
        subprocess.run('conda run -n neuralcoref2 python "'+DIR_main+"MainNeuralCoref.py"+'" "'+DIR+"Temporal.txt"+'"',
                       shell=True, check=True)

        with open(DIR+"Temporal.txt", encoding='utf-8') as text_file:
            new_text2+=text_file.read()+"\n"
    
    return(new_text2)



### Change Subject & Object to CCA measures & factors labels
def LabelMeasures(text,DICT_A=None,DICT_F=None):
    ADAPT=[]
    for tt in text:
        # break
        NA=[]; NB=[]
        ### Adaptations Dictionary ###
        if type(DICT_A)!=None:
            AA_S=np.where([ len(x)>0 for x in list(map(lambda x: re.findall(x,tt[0]),
                     DICT_A.Adaptation_REGEX))])
            AA_T=np.where([ len(x)>0 for x in list(map(lambda x: re.findall(x,tt[2]),
                     DICT_A.Adaptation_REGEX))])
        # Factors Dictionary
        if type(DICT_F)!=None:
            FF_S=np.where([len(x)>0 for x in map(lambda x: re.findall(x,tt[0]),
                     DICT_F.Factor_REGEX)])
            FF_T=np.where([len(x)>0 for x in map(lambda x: re.findall(x,tt[2]),
                     DICT_F.Factor_REGEX)])
        
        ### Using the dictionaries to homogenize the nouns ###
        # Replacing Subject/Source
        if type(DICT_A)!=None and len(AA_S[0])>=1:
            NA+=list(zip(DICT_A.Adaptation_Name[AA_S[0]],["ADAPT"]*len(AA_S[0])))
        elif type(DICT_F)!=None and len(FF_S[0])>=1:
            NA+=list(zip(DICT_F.Factor_Name[FF_S[0]],["FACT"]*len(DICT_F.Factor_Name[FF_S[0]])))
        else:
            NA.append((tt[0],""))
        # Replacing Object/Target
        if type(DICT_A)!=None and len(AA_T[0])>=1:
            NB+=list(zip(DICT_A.Adaptation_Name[AA_T[0]],["ADAPT"]*len(DICT_A.Adaptation_Name[AA_T[0]])))
        elif type(DICT_F)!=None and len(FF_T[0])>=1:
            NB+=list(zip(DICT_F.Factor_Name[FF_T[0]],["FACT"]*len(DICT_F.Factor_Name[FF_T[0]])))
        else:
            NB.append((tt[2],""))
        ### Complementing in case len(NA)!=len(NB)
        N=len(NA)*len(NB) # Size
        ADAPT+=list(it.product(NA,[tt[1]]*N,NB))
    
    ADAPT=list(set(ADAPT))
    return(ADAPT)




