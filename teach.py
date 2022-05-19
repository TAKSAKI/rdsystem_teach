import matplotlib.pyplot as plt#提案モデル周期境界条件なし
import numpy as np
import networkx as nx
from numba import jit


@jit
def min_d(X,d):#中央のノード次数を取得
    (size1,size2)=X.shape
    min=1
    max=0
    index=0
    for i in range(size1):
        if (X[i][0]-0.5)*(X[i][0]-0.5)+(X[i][1]-0.5)*(X[i][1]-0.5)<min:
            min=(X[i][0]-0.5)*(X[i][0]-0.5)+(X[i][1]-0.5)*(X[i][1]-0.5)
            index=i
    #for i in range(size1):
        #if d[i][i]>max:
            #max=d[i][i]
            #index=i
    print("index",index)
    print("min",int(d[index][index]))
    return index         

@jit
def adj(N,A,u,index):#隣接行列に初期値を与える
    #for i in range(N):
        #if A[index,i]==1 or i==index:
            #u[i]=0.1
    u[index]=0.3

@jit
def laplacian(s,L):#ラプラシアンを求める
    L1=s.shape
    S=int(s.size)
    ts = np.zeros(S)
    for i in range(S):
        for j in range(S):
            ts[i]+=L[i,j]*s[j]
    return ts

@jit
def calc(a, h, a2, h2, La,c):#状態量を求める
    L = a.size
    (L2,L2)=La.shape
    dt=0.01
    Dh=0.5#パラメーター始
    ca=0.08
    ch=0.11
    da=0.08
    #dh=0
    mua=0.03
    muh=0.12
    #aとhの密度が0.1になるように設定
    #roa=0.003
    #roh=0.001
    roa=(da+mua-ca)/10
    roh=(muh-ch)/10
    #roa=mua/10
    #roh=muh/10
    fa=ca-mua
    fh=-da
    ga=ch
    gh=-muh
    if c==0:
        Da=(Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)
    elif c==1:
        Da=0.057
    elif c==2:
        Da=0.059
    elif c==3:
        Da=0.02
    mina=0
    minh=0
    maxa=1
    maxh=1 
    sa = (ca*a)-(da*h)+roa-mua*a -Da * laplacian(a,La) ##反応項と拡散項を計算
    sh = (ch*a)+roh-muh*h -Dh * laplacian(h,La)  
    for i in range(L):
            a2[i] = a[i]+(sa[i])*dt #-mua*a[i,j]
            h2[i] = h[i]+(sh[i])*dt # -muh*h[i,j]           
            if a2[i]<mina:
                a2[i]=mina
            if h2[i]<minh:
                h2[i]=minh
            if a2[i]>maxa:
                a2[i]=maxa
            if h2[i]>maxh:
                h2[i]=maxh


def pic(N,u,v,G,pos,indexlist):#図示する
    for j in range(N):
        u[j]=round(u[j],2)
        v[j]=round(v[j],2)      
    print("maxu",np.max(u),"minu",np.min(u),"maxv",np.max(v),"minv",np.min(v))
    cent=u
    node_size = list(map(lambda x:x*500, cent))
    nodes = nx.draw_networkx_nodes(G, pos,node_size=30,
                               cmap='cool',
                               node_color=list(cent),
                               nodelist=list(indexlist))
    edges = nx.draw_networkx_edges(G, pos, width = 0.25)
    plt.colorbar(nodes)
    plt.show()
    cent1=v
    node_size = list(map(lambda x:x*500, cent))
    nodes = nx.draw_networkx_nodes(G, pos,node_size=30,
                               cmap='cool',
                               node_color=list(cent1),
                               nodelist=list(indexlist))
    edges = nx.draw_networkx_edges(G, pos, width = 0.25)
    plt.colorbar(nodes)
    plt.show()



def main():
    N = 1000# the number of points # N行2列の配列を作成
    indexlist=np.zeros(N)
    for i in range(N):
      indexlist[i]=i
    k=4
    np.random.seed(seed=314) 
    X = np.random.rand(N, 2)
    inf=100
    Dis=np.zeros((N, N))+inf#距離
    for i in range(N):
        for j in range(N):
            if i !=j:
                Disx=X[i][0]-X[j][0]
                Disy=X[i][1]-X[j][1]#距離を更新
                #if abs(X[i][0]-X[j][0])>0.5:#周期境界条件
                    #Disx=(abs(X[i][0]-X[j][0])-1)
                #if abs(X[i][1]-X[j][1])>0.5:
                    #Disy=(abs(X[i][1]-X[j][1])-1)
                Dis[i][j]=Disx*Disx+Disy*Disy#距離を更新
    kouho=np.zeros(N)#候補先のindex
    B=np.argmin(Dis, axis=1)
    for i in range(N):
        kouho[i]=B[i]#各ノードの中で距離が最小のノードを更新
    d=np.zeros((N,N))
    A=np.zeros((N,N))
#処理3から５    
    while True:
        index1=-1
        index2=-1
        min=inf
        for i in range(N):#全てのノードの中で，隣接ノードの候補との距離が最も短いノード
            j=int(kouho[i])
            if Dis[i][j]<min and d[i][i]<k and d[j][j]<k: #最小距離を求める
                min=Dis[i][j]
                index1=i
                index2=j
    #print("index1",index1,"index2",index2)
        if min==inf:#無向リンクが貼れなかったら終了
            break
        A[index1][index2]=A[index2][index1]=1
        Dis[index1][index2]=Dis[index2][index1]=inf
        d[index1][index1]+=1
        d[index2][index2]+=1
        if d[index1][index1]<k:#ポイントとなる次数がk未満なら新しい候補ノードを更新
            min1=inf
            indexp1=-1
            for i in range(N):
                if Dis[index1][i]<min1 and d[i][i]<k:#距離が最小のノードを設定
                    min1=Dis[index1][i]
                    indexp1=i
            if indexp1!=-1:
                kouho[index1]=indexp1#候補行列を更新
        #print("indexp1",indexp1)
        if d[index2][index2]<k:#ポイントとなる次数がk未満なら新しい候補ノードを更新
            min2=inf
            indexp2=-1
            for i in range(N):
                if Dis[index2][i]<min2 and d[i][i]<k: #距離が最小のノードを設定
                    min2=Dis[index2][i]
                    indexp2=i
            if indexp2!=-1:
                kouho[index2]=indexp2
        for i in range(N):#候補先のノードの次数が4のとき新しい候補を更新
            if d[i][i]<k:
                j=int(kouho[i])
                if d[j][j]==k:
                    min_kouho=inf
                    index_kouho=inf
                    for l in range(N):
                        if Dis[i][l]<min_kouho and d[l][l]<k:
                            min_kouho=Dis[i][l]
                            index_kouho=l
                    if index_kouho != inf:
                        kouho[i]=index_kouho
    L=d-A # (asymmetric) adjacecy matrix#mode=connectivity(０と1の接続行列)
    edges = []                                                              
    for i in range(N):
        for j in range(N):
            if A[i, j] != 0:
                edges.append((i, j)) #どのノードとどのノードがつながっているかのリスト作成

    ### Creating k-nearest neighbor graph from edge lists
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, pos=X[i]) # setting (x, y)-coordinates of nodes
    G.add_edges_from(edges)
    pos = {
        i: (X[i][0], X[i][1])
        for i in range(N)
    }
    #print(pos)
    ary = nx.to_numpy_matrix(G)
    index=min_d(X,d)#中央に初期値
    u0= np.zeros(N)#最大次数のノードと隣接してる行列の値に初期値を与える
    a=adj(N,ary,u0,index)
    u02 =np.zeros(N) 
    v0 = np.zeros(N)+0.1
    v02 =np.zeros(N)
    plt.subplot()
    plt.figure(figsize=(6,4))
    nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=20)
    plt.tight_layout()
    plt.show()
    time=100000
    for k in range(4):
        a=np.zeros(N)
        a2=np.zeros(N)
        h=np.zeros(N)
        h2=np.zeros(N)
        for i in range(N):#これができないと配列が初期化できない
            a[i]=u0[i]
            a2[i]=u02[i]
            h[i]=v0[i]
            h2[i]=v02[i]
        for i in range(time):
            if i % 2 == 0:
                calc(a, h, a2, h2, L,k)
            else:
                calc(a2, h2, a, h, L,k)
                    #現在のステップの状態u2,v2から次のステップの状態u,vを計算する
            if i==0 and k==0:   
                    pic(N,a,h,G,pos,indexlist) 
            if i==time-1:   
                    pic(N,a,h,G,pos,indexlist) 
                        
main=main()
