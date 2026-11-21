import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

st.set_page_config(page_title="PCA-GMM Process Monitor", page_icon="🩷", layout="wide")


st.markdown("""
<style>

/* ===== MAIN DASHBOARD ===== */
.stApp {
    background-color: #FFF9FC !important;
    color: #3B2430 !important;
}

/* Main headings */
h1, h2, h3 {
    color: #E94B8A !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #FCE7F1 !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #000000 !important;
}

/* Sidebar widget labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Sidebar description and caption */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* ===== DASHBOARD TABS ===== */

/* All tabs, including inactive tabs */
.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"],
.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab-list"] button,
.stTabs [data-baseweb="tab-list"] button *,
[data-testid="stTabs"] [role="tab"],
[data-testid="stTabs"] [role="tab"] * {
    color: #C00065 !important;
    -webkit-text-fill-color: #C00065 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* Selected tab */
.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] *,
[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-testid="stTabs"] [role="tab"][aria-selected="true"] * {
    color: #99004F !important;
    -webkit-text-fill-color: #99004F !important;
    font-weight: 800 !important;
}

/* ===== BUTTONS ===== */
div.stButton > button {
    background-color: #E94B8A !important;
    color: #FFFFFF !important;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
SENSORS=["Temperature (°C)","Pressure (kPa)","Flow rate (L/min)","Concentration (%)"]
CLASSES=["Normal","Temperature bias","Pressure drift","Flow drop","Concentration spike"]

def normal_data(n,rng):
    mode=rng.integers(0,2,n)
    temp=70+5*mode+rng.normal(0,.8,n)
    pressure=220+12*mode+1.7*(temp-70)+rng.normal(0,2,n)
    flow=42-3*mode-.18*(pressure-220)+rng.normal(0,1.1,n)
    conc=8+.35*mode+.025*(temp-70)-.012*(flow-42)+rng.normal(0,.18,n)
    return np.c_[temp,pressure,flow,conc]

def inject(x,name,strength):
    x=x.copy(); n=len(x)
    if name=="Temperature bias": x[:,0]+=2.2*strength
    elif name=="Pressure drift": x[:,1]+=np.linspace(0,16*strength,n)
    elif name=="Flow drop": x[:,2]-=5*strength
    elif name=="Concentration spike": x[:,3]+=.9*strength
    return x

def fit_detection(train):
    scaler=StandardScaler().fit(train); z=scaler.transform(train)
    pca=PCA(n_components=3,random_state=0).fit(z); scores=pca.transform(z)
    gmm=GaussianMixture(n_components=2,covariance_type="full",n_init=3,random_state=0).fit(scores)
    t2=np.sum(scores**2/np.maximum(pca.explained_variance_,1e-9),axis=1)
    q=np.sum((z-pca.inverse_transform(scores))**2,axis=1)
    return scaler,pca,gmm,np.quantile(t2,.99),np.quantile(q,.99)

def detect(x,model):
    scaler,pca,gmm,tlim,qlim=model
    z=scaler.transform(x); scores=pca.transform(z)
    t2=np.sum(scores**2/np.maximum(pca.explained_variance_,1e-9),axis=1)
    q=np.sum((z-pca.inverse_transform(scores))**2,axis=1)
    return scores,t2,q,(t2>tlim)|(q>qlim)

def fit_diagnosis(seed):
    rng=np.random.default_rng(seed+99); X=[]; y=[]
    for cls in CLASSES:
        a=inject(normal_data(400,rng),cls,1.0); X.append(a); y += [cls]*len(a)
    X=np.vstack(X); y=np.array(y)
    scaler=StandardScaler().fit(X); xz=scaler.transform(X)
    lda=LinearDiscriminantAnalysis(n_components=4).fit(xz,y); d=lda.transform(xz)
    gmms={}; priors={}
    for cls in CLASSES:
        subset=d[y==cls]
        gmms[cls]=GaussianMixture(n_components=1,random_state=seed).fit(subset)
        priors[cls]=np.mean(y==cls)
    return scaler,lda,gmms,priors

def diagnose(x,model):
    scaler,lda,gmms,priors=model; d=lda.transform(scaler.transform(x))
    lp=np.column_stack([gmms[c].score_samples(d)+np.log(priors[c]) for c in CLASSES])
    lp-=lp.max(axis=1,keepdims=True); p=np.exp(lp)
    return p/p.sum(axis=1,keepdims=True)

st.title("Real-Time Process Fault Detection & Diagnosis")
st.caption("Simulated sensor stream • PCA • Gaussian mixture model • Bayesian-style fault diagnosis")
with st.sidebar:
    st.header("Simulation controls")
    fault=st.selectbox("Injected condition",CLASSES)
    strength=st.slider("Fault strength",.25,2.0,1.0,.05)
    n=st.slider("Samples",60,300,150,10)
    seed=st.number_input("Random seed",0,99999,20)
    st.caption("Synthetic demonstration only; not connected to plant equipment.")
rng=np.random.default_rng(int(seed))
train=normal_data(1200,rng); model=fit_detection(train); diagnosis_model=fit_diagnosis(int(seed))
live=inject(normal_data(n,rng),fault,strength)
scores,t2,q,alarm=detect(live,model); probs=diagnose(live,diagnosis_model)
pred=np.array(CLASSES)[probs.argmax(axis=1)]
tlim,qlim=model[3],model[4]
c1,c2,c3,c4=st.columns(4)
c1.metric("Latest status","ALARM" if alarm[-1] else "NORMAL")
c2.metric("Alarm samples",f"{alarm.sum()} / {n}")
c3.metric("Latest diagnosis",pred[-1])
c4.metric("Posterior confidence",f"{probs[-1].max():.1%}")
tabs=st.tabs(["Live monitor","Fault diagnosis","PCA view","Method & limitations"])
df=pd.DataFrame(live,columns=SENSORS); df.insert(0,"Sample",np.arange(1,n+1))
with tabs[0]:
    selected=st.multiselect("Sensors",SENSORS,default=SENSORS)
    long=df.melt(id_vars="Sample",value_vars=selected,var_name="Sensor",value_name="Reading")
    st.plotly_chart(px.line(long,x="Sample",y="Reading",color="Sensor",title="Sensor trends"),use_container_width=True)
    a,b=st.columns(2)
    with a:
        f=go.Figure(); f.add_scatter(x=df["Sample"],y=t2,name="T²"); f.add_hline(y=tlim,line_dash="dash")
        f.update_layout(title="Hotelling T² (dashed = 99% baseline limit)"); st.plotly_chart(f,use_container_width=True)
    with b:
        f=go.Figure(); f.add_scatter(x=df["Sample"],y=q,name="Q"); f.add_hline(y=qlim,line_dash="dash")
        f.update_layout(title="Squared prediction error Q (dashed = limit)"); st.plotly_chart(f,use_container_width=True)
    st.dataframe(df.assign(T2=t2,Q=q,Alarm=alarm,Diagnosis=pred),use_container_width=True,hide_index=True)
with tabs[1]:
    st.subheader("Latest-sample diagnosis probabilities")
    st.plotly_chart(px.bar(pd.DataFrame({"Fault":CLASSES,"Probability":probs[-1]}),x="Fault",y="Probability",range_y=[0,1]),use_container_width=True)
    st.info(f"Most likely class: {pred[-1]} ({probs[-1].max():.1%}). Detection alarm: {'Yes' if alarm[-1] else 'No'}.")
    st.write(pd.Series(pred).value_counts().rename_axis("Predicted class").reset_index(name="Samples"))
with tabs[2]:
    p=pd.DataFrame(scores[:,:2],columns=["PC1","PC2"])
    p["Detection"]=np.where(alarm,"Alarm","Normal"); p["Diagnosis"]=pred
    st.plotly_chart(px.scatter(p,x="PC1",y="PC2",color="Detection",symbol="Diagnosis",title="Live samples in PCA score space"),use_container_width=True)
    st.write("Explained variance ratio:",[f"{v:.1%}" for v in model[1].explained_variance_ratio_])
with tabs[3]:
    st.markdown("""
**Workflow**
1. Generate correlated normal sensor data and standardize the variables.
2. Apply PCA to retain three principal components.
3. Fit a two-component GaussianMixture to normal PCA scores (scikit-learn estimates it using EM).
4. Monitor Hotelling's T² in retained score space and Q in the residual space. Limits are empirical 99th percentiles from training data.
5. For diagnosis, transform labelled simulated classes using LDA, fit one Gaussian mixture per class, and combine class likelihoods with priors.

**Important:** This is an educational adaptation inspired by PCA/GMM and discriminant-analysis ideas. It is not a line-by-line reproduction of the paper, a validated industrial system, or a safety instrument. The live stream is generated on each dashboard run.
""")
