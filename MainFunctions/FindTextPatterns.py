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
# These functions are based on:
# - Original code from (Accessed March 2023): 
#   https://github.com/NSchrading/intro-spacy-nlp/blob/master/subject_object_extraction.py.
# - Modified by (Accessed March 2023): 
#   https://stackoverflow.com/questions/39763091/how-to-extract-subjects-in-a-sentence-and-their-respective-dependent-phrases.
# - Updated by (Accessed March 2023): 
#   https://stackoverflow.com/questions/75264790/how-to-extract-the-subject-verb-object-and-their-relationship-of-a-sentence.
#
##################################
"""

# Data Handling
import pandas as pd 
import numpy as np 

# Processing text
import string
import regex as re 

import spacy as spacy 
from spacy.matcher import Matcher

# =============================================================================
# Global Variables
# =============================================================================
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl",\
            "pobj","compound","agent","amod","nmod","cc","conj","poss"] #"prep",
OBJECTS = ["dobj", "dative", "attr", "oprd","nsubj",\
            "pobj","compound","amod","nmod","conj","poss"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm", "hmod", "infmod", "xcomp",
              "rcmod", "poss", "possessive","compound"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]


# Parameters:
# - DIR: Directory where the csv file with the verbs is stored
def SignDict(DIR="@SofiaG1L/Database_CCA/DATA/Verbs.csv"):
    VERBS=pd.read_csv(DIR,sep=",")
    VERBS=VERBS[VERBS.NONE!=1]
    VERBS=VERBS.reset_index(drop=True)

    VERBS_dict=VERBS.melt(id_vars='VERB', var_name="SIGN",
               value_vars=["POSITIVE","NEGATIVE","NEUTRAL","NONE","DEPEND"])
    VERBS_dict=VERBS_dict[VERBS_dict.value==1]
    VERBS_dict=VERBS_dict.reset_index(drop=True)

    VERBS_dict["SIGN_"]=""
    VERBS_dict["NOTE"]=""

    N=VERBS.shape[0]
    for ii in range(N):
        if VERBS_dict.iloc[ii]["SIGN"]=="POSITIVE":
            VERBS_dict["SIGN_"].iloc[ii]="+"
            
        elif VERBS_dict.iloc[ii]["SIGN"]=="NEGATIVE":
            VERBS_dict["SIGN_"].iloc[ii]="-"
            
        elif VERBS_dict.iloc[ii]["SIGN"]=="NEUTRAL":
            VERBS_dict["SIGN_"].iloc[ii]="+/-"
        
        else: # This can only be DEPEND
            v=VERBS_dict.iloc[ii]["VERB"]
            wh=np.where(VERBS.VERB==v)
            check=VERBS.iloc[wh]["notes"].item()
            check=check.split("|")
            for jj in check:
                if jj.find("pos")>-1:
                    VERBS_dict=pd.concat([VERBS_dict,
                                   pd.DataFrame({
                                   'VERB':VERBS_dict.iloc[ii]['VERB'],\
                                   'SIGN':VERBS_dict.iloc[ii]['SIGN'],\
                                   'value':VERBS_dict.iloc[ii]['value'],\
                                   'SIGN_':"+",
                                   'NOTE':jj}, index=[0])],\
                        ignore_index=True)
                elif jj.find("neg")>-1:
                    VERBS_dict=pd.concat([VERBS_dict,
                                   pd.DataFrame({
                                   'VERB':VERBS_dict.iloc[ii]['VERB'],\
                                   'SIGN':VERBS_dict.iloc[ii]['SIGN'],\
                                   'value':VERBS_dict.iloc[ii]['value'],\
                                   'SIGN_':"-",
                                   'NOTE':jj}, index=[0])],\
                        ignore_index=True)
                else: # to broad
                    VERBS_dict=pd.concat([VERBS_dict,
                                   pd.DataFrame({
                                   'VERB':VERBS_dict.iloc[ii]['VERB'],\
                                   'SIGN':VERBS_dict.iloc[ii]['SIGN'],\
                                   'value':VERBS_dict.iloc[ii]['value'],\
                                   'SIGN_':"+",
                                   'NOTE':jj}, index=[0])],\
                        ignore_index=True)
            
    VERBS_dict=VERBS_dict[VERBS_dict['SIGN_']!='']
    VERBS_dict=VERBS_dict.reset_index(drop=True)

    VERBS_dict.index=VERBS_dict.VERB
    
    return(VERBS_dict)


# =============================================================================
# Use VERB to extract NOUNs
# =============================================================================
def find_noun_right(doc, count,IsPast=False):
    # Start looking to the right
    RIGHT=[]
    for cc, right in enumerate(doc[count:]):
        # break
        if right.dep_ in OBJECTS or right.lemma_=="and":
            # break
            if len(right.lemma_)>2 or right.lemma_ in ["i","he","it"]: # preventing noise from entering
                RIGHT.append(right.lemma_)
        
        if right.dep_ in ["punct","ROOT"]:
            if not right.lemma_ in ["-",",",";"]:
                break
            else:
                RIGHT.append(right.lemma_)
    return((' '.join(RIGHT)).lower())

def find_noun_left(doc, count):
    # check if previous two words were VBZ + TO
    check='_'.join([token.tag_ for token in doc[(count-2):count]])
    if check=="VBZ_TO":
        count=count-2
    # Start looking to the right
    LEFT=[]
    REV=[ii for ii in reversed(doc[:count])]
    for cc, left in enumerate(REV):
        # break
        if left.pos_ in ["ADJ","NOUN"]:
            # break
            if len(left.lemma_)>2 or left.lemma_ in ["I","he","it"]: # preventing noise from entering
                LEFT.append(left.lemma_)
            
        elif str(left) in [";",",","and"]:
            if str(left)!="and":
                LEFT.append(left.lemma_)
            else:
                LEFT.append(",")
            
        elif left.pos_ in ["VERB","SCONJ"]:
            break
        
    LEFT=[ii for ii in reversed(LEFT)]
    return(' '.join(LEFT))


# This function looks for 
def FindAux(doc, verb_name,nlp):
    
    pattern = [[{'POS': 'AUX', 'OP': '+'},
                {'POS': 'ADV', 'OP': '*'},
                {'POS': 'PART', 'OP': '*'},
                {'POS': 'ADJ', 'OP': '*'},
                {'POS': 'VERB', 'OP': '+'},
                {'POS': 'PART', 'OP': '?'},
                {'POS': 'VERB', 'OP': '?'}],
               [{'POS': 'AUX', 'OP': '+'},
                {'POS': 'ADV', 'OP': '*'},
                {'POS': 'PART', 'OP': '*'},
                {'POS': 'ADJ', 'OP': '*'},
                {'POS': 'VERB', 'OP': '+'},
                {'POS': 'CCONJ', 'OP': '?'},
                {'POS': 'VERB', 'OP': '?'}]]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb_phrase", pattern)

    # call the matcher to find matches 
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    
    return([x for x in spans if str(x).find(str(verb_name))>-1])
        

# This function looks for 
def IsPastPart(doc, verb_name,nlp):
    check=FindAux(doc, verb_name,nlp)
    if len(check)>0:
        VERB_PHRASE=max(check, key=len)
        VERB_PHRASE=[(x,x.tag_,x.morph.get("Tense")) for x in VERB_PHRASE if str(x)==str(verb_name)]
        
        if len(VERB_PHRASE)>0:
            if VERB_PHRASE[0][1]=="VBN" and VERB_PHRASE[0][2][0]=="Past":
                return(True)
            else:
                return(False)
    else:
        return(False)

def PresentSentence(doc, verb_name,nlp):
    
    try:
        VERB_PHRASE2=max(FindAux(doc, verb_name,nlp), key=len)
        VERB_PHRASE=[(x,x.tag_,x.morph.get("Tense")) for x in VERB_PHRASE2 if x==verb_name] #
        
        PHRASE=extract_rel_verbs(doc, verb_name=VERB_PHRASE[0][0],IsPast=True)
        
        if VERB_PHRASE[0][1]=="VBN" and VERB_PHRASE[0][2][0]=="Past":
            return([(PHRASE[2],PHRASE[1],PHRASE[0])])
        else:
            return([PHRASE])
    except:
        PHRASE=extract_rel_verbs(doc, verb_name=verb_name)
        return([PHRASE])
        
def extract_rel_verbs(doc, verb_name,IsPast=False):
    # Transforming nouns into its lemmas
    RIGHT = find_noun_right(doc, verb_name.i+1,IsPast=IsPast)
    LEFT = find_noun_left(doc, verb_name.i) 
    return((LEFT,verb_name.lemma_,RIGHT))

def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs

def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs
            
def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False

def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend(
                [tok for tok in dep.rights if tok.dep_ in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs

def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None

def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
        
    if len(subs)==0:
        subs=[tok for tok in v.lefts]
        
    return subs, verbNegated

def getAllObjsWithAdjectives(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]

    if len(objs) == 0:
        objs = [tok for tok in rights if tok.dep_ in ADJECTIVES]

    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs

def GenSubj(sub,v,tokens,nlp):
    if sub.i>0:
        aux=FindAux(tokens,v,nlp)
        if len(aux)>0:
            if sub in aux[0]:
                sub=tokens[aux[0][0].i-1]
        LEFTS=[t for t in sub.lefts]
        LEFTS.append(sub)
        if len(aux)>0:
            v=aux[0]
        try:
            sub_compound = nlp(re.search(str(LEFTS[0])+"(.*?)"+str(v),str(tokens)).group(0))
        except:
            try:
                sub_compound = nlp(re.search(str(v)+"(.*?)"+str(LEFTS[0]),str(tokens)).group(0))
            except:
                return([sub.lemma_])
        sub_compound = [t.lemma_ for t in sub_compound if t.lemma_ not in ["(",")","[","]"] and 
                        (t.dep_ in SUBJECTS or t.lemma_ in ["and",","])]
        return(sub_compound)
    else:
        return([sub.lemma_])

def SplitMarker(text,nlp):
    subText=[]
    for cc,ii in enumerate(text):
        if ii.dep_=='mark':
            subText.append(cc)
    
    SENTENCES=[]
    if len(subText)>0:
        SENTENCES.append(text[0:subText[0]])
        if len(subText)==1:
            SENTENCES.append(text[(subText[0]+1):])
            SENTENCES=[nlp(ii.text) for ii in SENTENCES]
            return(SENTENCES)
        else:
            n=len(subText)
            for ii in range(1,n):
                SENTENCES.append(text[(subText[(ii-1)]+1):subText[ii]])
            SENTENCES.append(text[(subText[ii]+1):])
    SENTENCES=[nlp(ii.text) for ii in SENTENCES]
    return(SENTENCES)

def Look4Ents(obj,tokens):
    ENTS=tokens.ents
    for ii in ENTS:
        if obj.i in range(ii.start,ii.end):
            return(ii.text)
        
def findSVAOs(tokens,nlp):
    svos = []
    verbs = [tok for tok in tokens if 
             (tok.pos_ == "VERB" and not tok.dep_ in["aux","amod"]) or tok.dep_=="ROOT"]
    
    verbs=[ii for ii in verbs if ii.i not in [0,len(tokens)]]
    
    for v in verbs:
        # break
        if  not IsPastPart(tokens, v,nlp):
            # I added the following line
            if v!=verbs[-1]:
                subs, verbNegated = getAllSubs(v)
                # hopefully there are subs, if not, don't examine this verb any longer
                if len(subs) > 0:
                    # break
                    v, objs = getAllObjsWithAdjectives(v)
                    for sub in subs:
                        # print(sub)
                        for obj in objs:
                            objNegated = isNegated(obj)
                            if Look4Ents(obj,tokens)!=None:
                                obj_desc_tokens = Look4Ents(obj,tokens)
                            else: 
                                obj_desc_tokens = find_noun_right(tokens,v.i+1)
                            sub_compound = GenSubj(sub,v,tokens,nlp)
                            svos.append((" ".join(tok for tok in sub_compound),
                                         "!" + v.lemma_ if verbNegated or objNegated else v.lemma_,
                                         obj_desc_tokens))
            else:
                subs, verbNegated = getAllSubs(v)
                # hopefully there are subs, if not, don't examine this verb any longer
                if len(subs) > 0:
                    # break
                    v, objs = getAllObjsWithAdjectives(v)
                    for sub in subs:
                        # print(sub)
                        for obj in objs:
                            if obj!=objs[-1]:
                                objNegated = isNegated(obj)
                                # obj_desc_tokens = generate_left_right_adjectives3(obj)
                                if Look4Ents(obj,tokens)!=None:
                                    obj_desc_tokens = Look4Ents(obj,tokens)
                                else: 
                                    obj_desc_tokens = find_noun_right(tokens,v.i+1)
                                sub_compound = GenSubj(sub,v,tokens,nlp)
                                svos.append((" ".join(tok for tok in sub_compound),
                                             "!" + v.lemma_ if verbNegated or objNegated else v.lemma_,
                                             obj_desc_tokens))
                            else:
                                objNegated = isNegated(obj)
                                # obj_desc_tokens = generate_left_right_adjectives2(obj,tokens)
                                obj_desc_tokens = find_noun_right(tokens,v.i+1)
                                sub_compound = GenSubj(sub,v,tokens,nlp)
                                svos.append((" ".join(tok for tok in sub_compound),
                                             "!" + v.lemma_ if verbNegated or objNegated else v.lemma_,
                                             obj_desc_tokens))
        else:
            svos.append(PresentSentence(tokens, v,nlp)[0])
    
    # Keeping unique elements
    svos=list(set(svos))
    
    # Removing objects contained in other objects: climate<climate change
    RMV=[]
    N=len(svos)
    for count in range(N):
        if count>0:
            if svos[count-1][2] in svos[count][2]:
                RMV.append(count-1)
    svos=[x for c,x in enumerate(svos) if c not in RMV]
    svos=[x for x in svos if x[0]!="" and (x[0] != len(x[0]) * x[0][0])]
    svos=[x for x in svos if x[2]!="" and (x[2] != len(x[2]) * x[2][0])]
    
    return([v for v in svos if v[2].find(v[1])==-1 and v[0].find(v[1])==-1])


### Function to detect the sentences that are fidings
def Finding(text,nlp0):
    if not pd.isna(text):
        VAL=(nlp0(text)).cats["POS"]
        if VAL<0.5:
            return(0)
        else:
            return(1)
    else:
        return(0)

def NegSIGN(SIGN):
    if SIGN=="+":
        return("-")
    elif SIGN=="-":
        return("+")
    else:
        return("+/-")

def SignVerbSplit(VRB,NOUNB,VERBS_dict):
    ss=VRB.split("!")
    
    VERBS=list(np.unique(VERBS_dict.index))
    
    if len(ss)==1:
        ss=ss[0]
        neg=0
    else:
        ss=ss[1]
        neg=1
    
    if ss in VERBS:
        if len(VERBS_dict.loc[ss])>2:
            if neg:
                return(NegSIGN(VERBS_dict.loc[ss]["SIGN_"]),NOUNB)
            else:
                return(VERBS_dict.loc[ss]["SIGN_"],NOUNB)
        else:
            DIRECT=VERBS_dict.loc[ss]
            drt_any=0
            for dd in range(2):
                srh=DIRECT["NOTE"].iloc[dd]
                if re.search(srh,NOUNB)!=None:
                    if neg:
                        return(NegSIGN(DIRECT["SIGN_"].iloc[dd]),re.sub(srh+" [a-z]*(\s|)","",NOUNB))
                    else:
                        return(DIRECT["SIGN_"].iloc[dd],re.sub(srh+" [a-z]*(\s|)","",NOUNB))
                    drt_any=1
                    break
            if drt_any==0:
                return("+/-",NOUNB)
    else:
        # If there is no direction then drop
        return("",NOUNB)
        
def FindAllSents(text, nlp1, nlp2, VERBS_dict):
    SENTENCES=SplitMarker(nlp2(text),nlp1)
    
    if len(SENTENCES)==0:
        SENTENCES=[nlp1(text)]
    
    SEN_VRB=[]
    
    for text in SENTENCES:
        # break
        # Finding all (Object, Verb, Subject) in sentence 
        VAL=findSVAOs(text,nlp2)
        for vv in VAL:
            # Checkin direction of the verb
            V,NB= SignVerbSplit(vv[1],vv[2],VERBS_dict)
            if V!="":
                # Splitting each Object and Subject based
                NOUNA=re.split(',|;|\Wand\W|\Wor\W',vv[0])
                NOUNA=[x.strip() for x in NOUNA]
                NOUNA=[x for x in NOUNA if x not in ["and","or"] and x!="" and (x!= len(x) * x)]
                NOUNB=re.split(',|;|\Wand\W|\Wor\W',NB)
                NOUNB=[x.strip() for x in NOUNB]
                NOUNB=[x for x in NOUNB if x not in ["and","or"] and x!="" and (x!= len(x) * x)]
                for ii in NOUNA:
                    for jj in NOUNB:
                        if ii!="" and jj!="" and ii!=jj:
                            SEN_VRB.append((ii.strip(),V,jj.strip()))
    
    SEN_VRB=list(set(SEN_VRB))
    
    return(SEN_VRB)






