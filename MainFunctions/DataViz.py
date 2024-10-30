# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:36:27 2022

@author: sgilclavel
"""

# Network Analysis
import networkx as nx

import pandas as pd
import numpy as np
from collections import Counter
import random

# Plots 
from matplotlib import pyplot as plt
from matplotlib.patches import CirclePolygon

# Colors
import distinctipy as distinctipy

# =============================================================================
# Functions to create the Network Graph
# =============================================================================

## Yes
def CoordsCircle(CTR,R,N,INC_LEN=False):
    if N>1:
        # N=N-1
        circle = CirclePolygon(CTR, radius = R, fc = 'y',resolution=N)
        verts = circle.get_path().vertices
        trans = circle.get_patch_transform()
        points = trans.transform(verts)
        if not INC_LEN:
            return([(cc[0],cc[1]) for cc in points.tolist()[:-1]])
        else:
            return([(cc[0],cc[1],R) for cc in points.tolist()[:-1]])
        return(points)
    else:
        circle = CirclePolygon(CTR, radius = R, fc = 'y',resolution=N)
        verts = circle.get_path().vertices
        trans = circle.get_patch_transform()
        points = trans.transform(verts)
        points=points[0]
        if not INC_LEN:
            cc=points.tolist()
            return([(cc[0],cc[1])])
        else:
            cc=points.tolist()
            return([(cc[0],cc[1],R)])
        return(points)
    

## Yes
def RandChose(XY,N):
    XY_=range(len(XY))
    IND=random.sample(XY_, N)
    CH=[v for c,v in enumerate(XY) if c in IND]    
    NOCH=[v for c,v in enumerate(XY) if c not in IND]    
    return(CH,NOCH)

## Yes
def SplitCoords(XY,CTR,LN,h): 
    p0=1.0/(1.0+h)
    XY1=[]
    for cc in XY:
        x=cc[0]
        y=cc[1]
        if x==CTR[0]:
            p=p0
            x1=x
            x1-=LN*0.5
            for ii in range(h):
                XY1.append((x1+LN*p,y))
                p+=p0
        elif y==CTR[1]:
            p=p0
            y1=y
            y1-=LN*0.5
            for ii in range(h):
                XY1.append((x,y1+LN*p))
                p+=p0
            
        else:
            print("The center is wrong.")
    return(XY1)

## Yes
def CheckPTS(N,SIDES=4):
    h=int(np.floor(N/SIDES)) # At least h can be accomodated per side
    re=N-h*SIDES # There are re that need to be accomodated
    PTSperSIDE=[h for ii in range(SIDES)]
    for ii in range(SIDES):
        PTSperSIDE[ii]+=1
        re-=1
        if re==0:
            break
    return(PTSperSIDE)

## Yes
def CoordsSquarePts(CTR,LN,N,INC_LEN=False):
    # Generate Basic Coords
    XY=Coords(CTR,LN)
    # Check case
    if N==1:
        if not INC_LEN:
            return([CTR])
        else:
            return([(CTR[0],CTR[1],LN)])
    elif N<=4: # The number of square sides
        CH,NOCH=RandChose(XY,N)
        if not INC_LEN:
            return(CH)
        else:
            return([(x,y,LN) for x,y in CH])
    elif N%4==0: # More points than square sides but even
        h=int(N/4) 
        CC=SplitCoords(XY,CTR,LN,h)
        if not INC_LEN:
            return(CC)
        else:
            return([(x,y,LN) for x,y in CC])
    elif N%4!=0: # More points than square sides but odd
        PTSperSIDE=CheckPTS(N)
        PTS=[]
        for ii in range(4):
            PTS.append(SplitCoords([XY[ii]],CTR,LN,PTSperSIDE[ii]))
        
        PTS=[item for sublist in PTS for item in sublist]
        if not INC_LEN:
            return(PTS)
        else:
            return([(x,y,LN) for x,y in PTS])
    else:
        print("N="+str(N)+" is not a positive integer.")
     

# =============================================================================
# The next functions split the points in the squares
# =============================================================================

# Optimal number of points per layer
def MAXpts(L):
    if type(L)!=int or L<0:
        print("Only positive integers are accepted.")
        return(None)
    if L==0:
        return(1)
    else:
        return((2*L-1)*4)
    
# Function to distribute points in squares
## Yes
def DistPts(N,L=None):
    if L!=None: # If number of Layers is given by the user.
        SUM=sum(map(MAXpts,range(L)))
        if N>SUM:
            print(f"{N} is bigger than the possible number of dots in {L} squares.")
            return(None)
        elif N<L:
            print(f"{N} is smaller than the number of layers: {L}.")
            return(None)
        else:
            LAY=[1 for ii in range(L)]
            N0=N-L
            while N0>0:
                for ii in range(L):
                    if LAY[ii]<MAXpts(ii):
                        LAY[ii]+=1
                        N0-=1
                        if N0<=0:
                            break
            return(LAY)
    else: # If the user doesn't give number of layers
        CHECK=list(map(MAXpts,range(N)))
        L=np.min(np.where(np.cumsum(CHECK)>=N))+1
        return(DistPts(N,L=L))
        
## Yes
def CoordsDistPts(CTR,N,LN=None,dw=10,INC_LEN=False,Type="Circle"):
    ## Placing Adapt edges in Graph
    LEN=DistPts(N,LN)
    # choosing function:
    if Type=="Circle":
        XY=[]
        w=1
        for ii in range(len(LEN)):
            XY+=CoordsCircle(CTR,w,LEN[ii],INC_LEN)
            w+=dw
    elif Type=="Square":
        XY=[]
        w=1
        for ii in range(len(LEN)):
            # CoordsNestedPts(CTR,LN,N,INC_LEN=False)
            XY+=CoordsSquarePts(CTR,w,LEN[ii],INC_LEN)
            w+=dw
    else:
        print(f"Type {Type} does not exist. Only 'Circle' or 'Square'.")
    return(XY)
    

# =============================================================================
# Other useful functions
# =============================================================================

## Yes
def CoordsRotateDict(XY,T):
    XY1={}
    for k,v in XY.items():
        # break
        XY1[k]=list(CoordsRotate(v,T))
    return(XY1)

# =============================================================================
# The next functions are to Plot the graph based on the Modulrarity
# =============================================================================

## Yes
def UpdateCoords(DictCoords,xn,yn):
    NDC={}
    for k,v in DictCoords.items():
        NDC[k]=(v[0]+xn,v[1]+yn)
    return(NDC)

## Yes
def INSIDESQR(DictCoords,UCR,LCR):
    ANY=[]
    Xmax=LCR[0]; Xmin=UCR[0]
    Ymax=UCR[1]; Ymin=LCR[1]
    for k,v in DictCoords.items(): 
        ANY.append(not (v[0] < Xmin or v[0] > Xmax or v[1] < Ymin or v[1] > Ymax))
    return(any(ANY))

## Yes
def CheckMinMax(posn):
    MIN_X=np.Infinity
    MAX_X=-np.Infinity
    MIN_Y=np.Infinity
    MAX_Y=-np.Infinity
    
    for v in posn.values():
        for v2 in v.values():
            if v2[0]<MIN_X:
                MIN_X=v2[0]
            if v2[0]>MAX_X:
                MAX_X=v2[0]
            if v2[1]<MIN_Y:
                MIN_Y=v2[1]
            if v2[1]>MAX_Y:
                MAX_Y=v2[1]
    
    return((MIN_X,MAX_X),(MIN_Y,MAX_Y))

## Yes
def RelocateCoords(pos,MEAS):
    posn={}
    # Changing coords to fit all sub networks
    LX0=0; LX1=0
    LY0=0; LY1=0
    xn=0; yn=0
    SIGN=(+1,-1,-1,+1)
    N=len(pos); cc=0
    SIDE=0; SIDE0=[ii*4 for ii in range(N)]; SIDE2=[ii+2 for ii in SIDE0]
    while cc<N:
        if cc==0: # First iteration moves to the right
            posn[cc]=pos[cc].copy()
            LX1+=2*abs(MEAS[cc][1])
            UCR=[MEAS[cc][0][0]-LX1/2,MEAS[cc][0][0]+LX1/2]
            LCR=[MEAS[cc][0][0]+LX1/2,MEAS[cc][0][0]-LX1/2]
            cc=+1
            # plotDOTS(posn)
        elif cc==1: # Second iteration moves to the bottom
            xn+=(MEAS[cc-1][1]+MEAS[cc][1])
            posn[cc]=UpdateCoords(pos[cc],xn,yn)
            LX1+=2*abs(MEAS[cc][1])
            LY1+=2*abs(max(MEAS[cc-1][1],MEAS[cc][1]))
            SIDE+=1
            cc+=1
            # plotDOTS(posn)
        elif cc==2: # Third iteration moves to the left
            yn+=SIGN[SIDE%4]*(MEAS[cc-1][1]+MEAS[cc][1])#(LY1)
            posn[cc]=UpdateCoords(pos[cc],xn,yn)
            # SIDE+=1
            X_d,Y_d=CheckMinMax(posn)
            LCR[0]=X_d[1]; #LCR[1]=Y_d[0]
            LX1=abs(X_d[0])+abs(X_d[1])
            LY1=abs(Y_d[0])+abs(Y_d[1])
            cc+=1
            # plotDOTS(posn)
        elif cc>2: # Everything is now initialized 
            if cc==3:
                # Checking if the next circle is inside the area
                xc=xn+SIGN[SIDE%4]*(MEAS[cc-1][1]+MEAS[cc][1])
                posn[cc]=UpdateCoords(pos[cc],xc,yn)
                if not INSIDESQR(posn[cc],UCR,LCR):
                    SIDE+=1
                    
            if SIDE%2==0:
                INSIDE=True
                X_d,Y_d=CheckMinMax(posn) # Update LY0
                LX0=abs(X_d[0])+abs(X_d[1])
                LX1=0
                while not (LX1>LX0 and not INSIDE): # 
                    xn+=SIGN[SIDE%4]*(MEAS[cc-1][1]+MEAS[cc][1])
                    posn[cc]=UpdateCoords(pos[cc],xn,yn)
                    # plotDOTS(posn)
                    LX1+=2*abs(MEAS[cc][1])
                    INSIDE=INSIDESQR(posn[cc],UCR,LCR)
                    cc+=1
                    if cc>=N:
                        # break
                        return(posn)
                SIDE+=1
                # yn+=SIGN[SIDE%4]*(MEAS[cc][1]) #+MEAS[cc][1])
                if SIDE in SIDE0:
                    X_d,Y_d=CheckMinMax(posn)
                    UCR=[X_d[0],Y_d[1]]
                if SIDE in SIDE2:
                    X_d,Y_d=CheckMinMax(posn)
                    if X_d[1]>LCR[0]:
                        LCR[0]=X_d[1]
                    if Y_d[0]<LCR[1]:
                        LCR[1]=Y_d[0]
            else:
                INSIDE=True
                X_d,Y_d=CheckMinMax(posn) # Update LY0
                LY0=abs(Y_d[0])+abs(Y_d[1])
                if cc!=3:
                    LY1=0
                
                while not (LY1>LY0 and not INSIDE): # 
                    yn+=SIGN[SIDE%4]*(MEAS[cc-1][1]+MEAS[cc][1])
                    posn[cc]=UpdateCoords(pos[cc],xn,yn)
                    # plotDOTS(posn)
                    LY1+=2*abs(MEAS[cc][1])
                    INSIDE=INSIDESQR(posn[cc],UCR,LCR)
                    cc+=1
                    if cc>=N:
                        # break
                        return(posn)
                SIDE+=1
                # xn+=SIGN[SIDE%4]*(MEAS[cc][1]) #+MEAS[cc][1])
                if SIDE in SIDE0:
                    X_d,Y_d=CheckMinMax(posn)
                    UCR=[X_d[0],Y_d[1]]
                if SIDE in SIDE2:
                    X_d,Y_d=CheckMinMax(posn)
                    if X_d[1]>LCR[0]:
                        LCR[0]=X_d[1]
                    if Y_d[0]<LCR[1]:
                        LCR[1]=Y_d[0]
        else:
            print("There is something wrong!")
        
    return(posn)

## Yes
def PosModularity(WORDS_DB,COMMU=None,dw=30, delta=5, ROTATE=False,weight=None,
                  delta_edge=1,min_edge=None,resolution=1):
    
    # T_ART=len(np.unique(WORDS_DB.ID_ART))
    
    # WORDS_DB=WORDS_DB.drop(columns='ID_ART',axis=1)
    
    # WORDS_DB=WORDS_DB.groupby(["source","target","SIGN"],as_index=False).\
    #     sum("proportion").sort_values(['proportion'], ascending=False)
    # WORDS_DB["proportion"]=WORDS_DB["proportion"]#/T_ART
    # Standardizing between 0 and 1
    MAX=max(WORDS_DB["edge_weight"]); 
    MIN=min(WORDS_DB["edge_weight"])
    if MAX>1 and (MAX!=MIN):
        WORDS_DB["edge_weight"]=delta_edge*((WORDS_DB["edge_weight"]-MIN)/(MAX-MIN))
    elif MAX==MIN:
        WORDS_DB["edge_weight"]=[delta_edge for ii in range(WORDS_DB.shape[0])]
    WORDS_DB=WORDS_DB.rename(columns={"edge_weight":"weight"})
    
    # Counting Degree
    CS = Counter()
    CT = Counter()
    ## Using Networkx
    Ag=nx.DiGraph()
    n=WORDS_DB.shape[0]
    for cc in range(n):
        Ag.add_edge(WORDS_DB.iloc[cc]["source"],WORDS_DB.iloc[cc]["target"], 
            weight=WORDS_DB.iloc[cc]["weight"], sign=WORDS_DB.iloc[cc]["SIGN"])
        CS.update({WORDS_DB.source.iloc[cc]:WORDS_DB.degree.iloc[cc]})
        CT.update({WORDS_DB.target.iloc[cc]:WORDS_DB.degree.iloc[cc]})
    
    EDGES=CS+CT
    # Check the communities
    if type(COMMU)==type(None):
        if not pd.isna(weight):
            COMMU=nx.community.greedy_modularity_communities(Ag,weight="weight",resolution=resolution)
        else:
            COMMU=nx.community.greedy_modularity_communities(Ag,resolution=resolution)
    
    # Sorting nodes by degree
    COMMU2=[]
    for kk in COMMU:
        # break
        NDS=[ii for ii in kk]
        COMMU2.append([k for k in sorted(NDS, key=lambda ii: EDGES[ii],\
                                         reverse=True)])
        
    
    # Save as many positions as communities
    MEAS={}
    pos={}
    for cc, kk in enumerate(COMMU2):
        # break
        NDS=[ii for ii in kk]
        XY=CoordsDistPts(CTR=(0,0),N=len(NDS),dw=dw,INC_LEN=True)
        MEAS[cc]=[(XY[0][0],XY[0][1]),XY[-1][2]+delta]
        for cc2,ii in enumerate(NDS):
            if cc not in pos.keys():
                pos[cc]={ii:(XY[cc2][0],XY[cc2][1])}
            else:
                pos[cc][ii]=(XY[cc2][0],XY[cc2][1])
                
    posn=RelocateCoords(pos,MEAS)
    
    # Rotate coords to avoid overlap
    if ROTATE:
        pos={}
        T=0
        for k,v in posn.items():
            # break
            pos[k]=CoordsRotateDict(v,T)
            T+=0.3
        return(Ag,pos)
    
    return(Ag,posn)
    

def plotNet(G,pos,fontsize=8, VERTEX=True, MODULARITY=False, N_M=None,
            color={"+":"#004D40","+/-":"#FFC107","-":"#D81B60"},
            min_target_margin=15,loc_legend='lower right',
            TOBOLD=None):
    
    # Calculating plot size
    X_d, Y_d= CheckMinMax(pos)
    y=np.sqrt(Y_d[0]**2+Y_d[1]**2)
    x=np.sqrt(X_d[0]**2+X_d[1]**2)
    prop=x/y
    
    if MODULARITY:
        N=len(pos)
        if pd.isna(N_M) or N_M>N:
            MAX=N
        else:
            MAX=N_M
        COLORS=distinctipy.distinctipy.get_colors(N+1, pastel_factor=0.7)
        COLORS=distinctipy.get_colors(N+1, COLORS, colorblind_type="Deuteranomaly")
        COLORS=COLORS[1:]
        COLOR_nd={}
        for k,v in pos.items():
            if k>=MAX:
                break
            for k2,v2 in v.items():
                COLOR_nd[k2]=COLORS[k]
        pos={k2:v2 for k,v in pos.items() for k2,v2 in v.items() if k2 in COLOR_nd.keys()}
    
    if not MODULARITY:
        Ag=G.copy()
        labels = nx.get_edge_attributes(G,'sign')
        COLOR_ed=[color[v] for k,v in labels.items()]
        labels = nx.get_edge_attributes(G,'weight')
        WEIGTH=[v for k,v in labels.items()]
    elif MODULARITY and (pd.isna(N_M) or N_M>N):
        Ag=G.copy()
        labels = nx.get_edge_attributes(G,'sign')
        COLOR_ed=[color[v] for k,v in labels.items()]
        labels = nx.get_edge_attributes(G,'weight')
        WEIGTH=[v for k,v in labels.items()]
    else:
        Ag=G.subgraph(pos.keys()).copy()
        labels = nx.get_edge_attributes(Ag,'sign')
        COLOR_ed=[color[v] for k,v in labels.items()]
        labels = nx.get_edge_attributes(Ag,'weight')
        WEIGTH=[v for k,v in labels.items()]
        
    fig=plt.figure(figsize=(15*prop,15))
    # nx.draw(Ag, pos, with_labels = True) # 
    # Adding figure label
    if VERTEX:
        ax = fig.add_subplot(1,1,1)
        for label in color:
            ax.plot([0],[0],
                    color=color[label],
                    label=label)
            
    if TOBOLD==None:
        for k,v in pos.items():
            # break
            t=plt.text(v[0],v[1],k,
                        fontsize=fontsize,ha='center',va='center',
                        backgroundcolor="w")
            if MODULARITY:
                t.set_bbox(dict(facecolor=COLOR_nd[k], 
                                alpha=0.2, edgecolor=COLOR_nd[k]))
    else:
        for k,v in pos.items():
            # break
            if k in TOBOLD:
                t=plt.text(v[0],v[1],k,
                            fontsize=fontsize,ha='center',va='center',
                            backgroundcolor="w",weight='bold')
                if MODULARITY:
                    t.set_bbox(dict(facecolor=COLOR_nd[k],
                                    alpha=0.2, edgecolor=COLOR_nd[k]))
            else:
                t=plt.text(v[0],v[1],k,
                            fontsize=fontsize,ha='center',va='center',
                            backgroundcolor="w")
                if MODULARITY:
                    t.set_bbox(dict(facecolor=COLOR_nd[k], 
                                    alpha=0.2, edgecolor=COLOR_nd[k]))
                
            
    if VERTEX:
        nx.draw_networkx_edges(Ag,pos,width=WEIGTH,edge_color=COLOR_ed,
                               min_target_margin=min_target_margin,
                               # arrowstyle="simple",
                               connectionstyle="arc3,rad=0.40") # 
    else:
        nx.draw_networkx_edges(Ag,pos,edge_color="w",
                               min_target_margin=min_target_margin,
                               # arrowstyle="simple",
                               connectionstyle="arc3,rad=0.40") # 
    if VERTEX:
        plt.legend(loc=loc_legend,fontsize=fontsize+2,title="Edge Color",
                   ncols=3)
    plt.axis("off")


''' Functions to redistribute nodes and edges weights '''
## Yes
def MapSeries(vector,MIN=0,MAX=1):
    ### Biyective function to map a series into a (MIN,MAX) range
    MIN_0=min(vector)
    MAX_0=max(vector)
    vector2=[MIN+(MAX-MIN)/(MAX_0-MIN_0)*(ii-MIN_0) for ii in vector]
    return(vector2)

## Yes
def SigmoidEdge(vector,q,L=1, x0=0, k=-1):
    ### Function to map values into sigmoid (0,1) range, kind of like probabilities
    # :param float L: the curve’s maximum value
    # :param float x0: the midpoint of the sigmoid
    # :param float k: the logistic growth rate or steepness of the curve
    # :param list(float) vector: the input
    return([L/(1+np.exp(-k*(ii-x0))) for ii in vector])

## Yes
def QuantilesWeight(vector,q,L=1, x0=0, k=1):
    ### Function to give more weight to the points above the q quantile
    # :param float q: the quantile
    # :param float L: the curve’s maximum value
    # :param float x0: the midpoint of the sigmoid
    # :param float k: the logistic growth rate or steepness of the curve
    # :param list(float) vector: the input
    # :return list(float): list of mapped values
    ###
    nq=np.quantile(vector,q=q)
    return(SigmoidEdge(vector,None,L=L, x0=nq, k=k))

## Yes
def QuantilesWeight_upQ(vector,q,L=1, x0=0, k=1):
    ### Function to only give weight to the points above the q quantile
    # :param float q: the quantile
    # :param float L: the curve’s maximum value
    # :param float x0: the midpoint of the sigmoid
    # :param float k: the logistic growth rate or steepness of the curve
    # :param list(float) vector: the input
    # :return list(float): list of mapped values
    ###
    nq=np.quantile(vector,q=q)
    vector=[x if x>=nq else 0 for x in vector]
    return(MapSeries(vector,MIN=0,MAX=L))

## Yes
def plotNet_ChooseColors(G,pos,fontsize=12, VERTEX=True, MODULARITY=False, 
                    N_M=None,min_target_margin=15,loc_legend='lower right',
                    TOBOLD=None,title_fontsize=12,
                    edge_sigmoid=True, 
                    colors_edge={"+":"#004D40","+/-":"#FFC107","-":"#D81B60"},
                    title_edge="Edge Color",
                    colors_node=None,colors_node_leg=None,
                    FunTrans=QuantilesWeight,q=0.5,x0=0,k=100,L=10):
    ### Function to visualize the network
    # :param network G: result of using the function PosModularity
    # :param dict pos: result of using the function PosModularity
    # :param float|dict fontsize=12: 
    #    float: maximum fontsize of text
    #    dict: dictionary with the fontsize of each node
    # :param bolean VERTEX=True
    # :param bolean MODULARITY=False
    # :param None|int|list(int) N_M=None:
    #    None: Displays the whole network
    #    int: Displays up to that cluster
    #    list(int): Displays the clusters in the list. 
    # :param float min_target_margin=15
    # :param string loc_legend='lower right'
    # :param None|dict TOBOLD=None
    # :param float title_fontsize=12
    # :param edge_sigmoid=True
    # :param colors_edge={"+":"#004D40","+/-":"#FFC107","-":"#D81B60"}
    # :param title_edge="Edge Color"
    # :param colors_node=None
    # :param colors_node_leg=None
    # :param FunTrans=QuantilesWeight
    # :param q=0.5
    # :param x0=0
    # :param k=100
    # :param L=10 
    # :return None
    
    ### Keeping only those clusters that will be used
    N=len(pos)
    if not type(N_M)==list: # or N_M>N:
        if type(N_M)==int: # or N_M>N:
            if (N_M)>N:
                N_M=range(N)
            else:
                N_M=range(N_M)
        else:
            N_M=range(N)
    
    pos2={}
    for k,v in pos.items():
        if k in N_M:
            pos2[k]={}
            for k2,v2 in v.items():
                pos2[k][k2]=pos[k][k2]
    
    
    # Calculating plot size
    X_d, Y_d= CheckMinMax(pos2)
    # y=np.sqrt(Y_d[0]**2+Y_d[1]**2)
    # x=np.sqrt(X_d[0]**2+X_d[1]**2)
    # prop=x/y
    
    ### Transform all points to [0,1] values
    pos={}
    for k,v in pos2.items():
        pos[k]={}
        #Giving some marging to the bottom & right for legends (*0.9)
        for k2,v2 in v.items(): 
            pos[k][k2]=((v2[0]-X_d[0])/(X_d[1]-X_d[0]), # * 0.8
                         (v2[1]-Y_d[0])/(Y_d[1]-Y_d[0])) # * 0.8
    
    ### Checking the clusters that will be displayed
    if MODULARITY:
        if pd.isna(colors_node):
            COLORS=distinctipy.distinctipy.get_colors(N+1, pastel_factor=0.7)
            COLORS=distinctipy.get_colors(N+1, COLORS, colorblind_type="Deuteranomaly")
            COLORS=COLORS[1:]
            COLOR_nd={}
            for k,v in pos.items():
                for k2,v2 in v.items():
                    COLOR_nd[k2]=COLORS[k]
        else:
            COLOR_nd={}
            for k,v in pos.items():
                # break
                for k2,v2 in v.items():
                    # break
                    try:
                        COLOR_nd[k2]=colors_node[k2]
                    except: 
                        COLOR_nd[k2]=(0.99, 0.99, 0.99)
        pos={k2:v2 for k,v in pos.items() for k2,v2 in v.items() if k2 in COLOR_nd.keys()}
    
    if not MODULARITY:
        Ag=G.copy()
        labels = nx.get_edge_attributes(G,'sign')
        COLOR_ed=[colors_edge[v] for k,v in labels.items()]
        labels = nx.get_edge_attributes(G,'weight')
        WEIGTH=[v for k,v in labels.items()]
        if edge_sigmoid:
            WEIGTH=FunTrans(WEIGTH,q,L=L, x0=x0, k=k)
        
    elif MODULARITY and N==len(N_M):
        Ag=G.copy()
        labels = nx.get_edge_attributes(G,'sign')
        COLOR_ed=[colors_edge[v] for k,v in labels.items()]
        labels = nx.get_edge_attributes(G,'weight')
        WEIGTH=[v for k,v in labels.items()]
        if edge_sigmoid:
            WEIGTH=FunTrans(WEIGTH,q,L=L, x0=x0, k=k)
    else:
        Ag=G.subgraph(pos.keys()).copy()
        labels = nx.get_edge_attributes(Ag,'sign')
        COLOR_ed=[colors_edge[v] for k,v in labels.items()]
        labels = nx.get_edge_attributes(Ag,'weight')
        WEIGTH=[v for k,v in labels.items()]
        if edge_sigmoid:
            WEIGTH=FunTrans(WEIGTH,q,L=L, x0=x0, k=k)
    
    
    # Parameters main fgure:
    fig=plt.figure(figsize=(15,15))
    gs = fig.add_gridspec(2, 1,  height_ratios=(10, 0.8),
                      bottom=0, top=1,hspace=0)

    # Adding subplot
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
    ax = fig.add_subplot(gs[0,0], frameon=False)
    for label in colors_edge:
        ax.plot([0],[0],color=colors_edge[label],label=label)
        
    if TOBOLD==None:
        for k,v in pos.items():
            # break
            t=plt.text(v[0],v[1],k,
                        fontsize=[fontsize if type(fontsize) in [float,int] else fontsize[k]][0],
                        ha='center',va='center',
                        backgroundcolor="w")
            if MODULARITY:
                t.set_bbox(dict(facecolor=COLOR_nd[k], 
                                alpha=0.7,edgecolor=COLOR_nd[k])) #  
    else:
        for k,v in pos.items():
            # break
            if k in TOBOLD:
                t=plt.text(v[0],v[1],k,
                            fontsize=[fontsize if type(fontsize) in [float,int] else fontsize[k]][0],
                            ha='center',va='center',
                            backgroundcolor="w",weight='bold')
                if MODULARITY:
                    t.set_bbox(dict(facecolor=COLOR_nd[k],
                                    alpha=0.7,edgecolor=COLOR_nd[k])) # alpha=0.2, 
            else:
                t=plt.text(v[0],v[1],k,
                            fontsize=[fontsize if type(fontsize) in [float,int] else fontsize[k]][0],
                            ha='center',va='center',
                            backgroundcolor="w")
                if MODULARITY:
                    t.set_bbox(dict(facecolor=COLOR_nd[k], 
                                   alpha=0.7,edgecolor=COLOR_nd[k])) # alpha=0.2, 
    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    
    legends=[]
    
    if VERTEX:
        # Adding figure label
        nx.draw_networkx_edges(Ag,pos,width=WEIGTH,edge_color=COLOR_ed,
                               min_target_margin=min_target_margin,
                               # arrowstyle="simple",
                               connectionstyle="arc3,rad=0.40") # 
        legends.append(ax.legend(loc=loc_legend,fontsize=title_fontsize,
                                 title_fontsize=title_fontsize+2,
                                 title=title_edge,ncols=3))
    else:
        nx.draw_networkx_edges(Ag,pos,edge_color="w",
                               min_target_margin=min_target_margin,
                               # arrowstyle="simple",
                               connectionstyle="arc3,rad=0.40") # 
    
    if not pd.isna(colors_node_leg):
        ax2 = fig.add_subplot(gs[1, 0], frameon=False)
        ## Checking the labels that appeared
        COLORS_NODES={}
        for k,v in colors_node.items():
            if k not in COLORS_NODES.keys():
                COLORS_NODES[colors_node_leg[v]]=v
        
        for k,v in COLORS_NODES.items():
            ax2.scatter([], [], color=v, label=k,s=title_fontsize*20,marker=",") #alpha=0.4,
        
        legends.append(ax2.legend(scatterpoints=1,labelspacing=1, 
                                  title_fontsize=title_fontsize+2,
                                  title='Constraint',fontsize=title_fontsize,
                                  loc="lower center",ncols=4))
    
    for ii in range(len(legends)):
        if ii==0:
            legends[ii]
        else:
            fig.gca().add_artist(legends[ii])
    
    plt.axis("off")


# =============================================================================
# Functions to fix labels and text
# =============================================================================
## Yes
def AddBreakLine(STR,n=2,TYPE="words",breakAfter=None):
    STR2=''
    if pd.isna(breakAfter):
        if TYPE=="words":
            JJ=STR.split()
            if len(JJ)>1:
                for cc,jj in enumerate(JJ):
                    if cc%n==0:
                        STR2+=" "+jj+"\n"
                    else:
                        STR2+=" "+jj
                return(STR2.strip())
            else:
                return(STR)
        elif TYPE=="chars":
            JJ=STR
            if len(JJ)>1:
                for cc,jj in enumerate(JJ):
                    if cc%n==0:
                        STR2+=jj+"\n"
                    else:
                        STR2+=jj
                return(STR2.strip())
            else:
                return(STR)
        else:
            print(f'TYPE="{TYPE}" is not valid. Choose between "words" and "chars".')
    else:
        if TYPE=="words":
            JJ=STR.split()
            if len(JJ)>1:
                for cc,jj in enumerate(JJ):
                    if cc==breakAfter:
                        break
                    if cc%n==0:
                        STR2+=" "+jj+"\n"
                    else:
                        STR2+=" "+jj
                if cc==breakAfter:
                    STR2+="..."
                return(STR2.strip())
            else:
                return(STR)
        elif TYPE=="chars":
            JJ=STR
            if len(JJ)>1:
                for cc,jj in enumerate(JJ):
                    if cc==breakAfter:
                        break
                    if cc%n==0:
                        STR2+=jj+"\n"
                    else:
                        STR2+=jj
                if cc==breakAfter:
                    STR2+="..."
                return(STR2.strip())
            else:
                return(STR)
        else:
            print(f'TYPE="{TYPE}" is not valid. Choose between "words" and "chars".')






