import numpy as np, pandas as pd, json
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
n = 12000
regiones = ["Lima","Norte","Sur","Centro","Oriente"]
reg = rng.choice(regiones, n, p=[.45,.2,.17,.11,.07])
canal = rng.choice(["Web","App"], n, p=[.55,.45])
pagos = ["Tarjeta","Billetera digital","Contra entrega"]
pmap = {"Lima":[.45,.45,.10],"Norte":[.35,.40,.25],"Sur":[.38,.40,.22],"Centro":[.30,.38,.32],"Oriente":[.25,.35,.40]}
pago = np.array([rng.choice(pagos, p=pmap[r]) for r in reg])
mu = {"Lima":4.95,"Norte":4.85,"Sur":4.88,"Centro":4.78,"Oriente":4.75}
ticket = np.exp(np.array([rng.normal(mu[r],0.55) for r in reg]))
ticket = ticket * np.where(canal=="App",1.04,1.0)
# outliers corporativos
idx = rng.choice(n, 40, replace=False); ticket[idx] *= rng.uniform(8,15,40)
dias_base = {"Lima":1.8,"Norte":3.2,"Sur":3.5,"Centro":3.9,"Oriente":5.2}
dias = np.clip(np.round(np.array([rng.gamma(4, dias_base[r]/4) for r in reg]),0),1,15)
satis = np.clip(np.round(4.8 - 0.22*dias + rng.normal(0,0.8,n)),1,5)
df = pd.DataFrame(dict(region=reg,canal=canal,metodo_pago=pago,ticket=np.round(ticket,2),dias_entrega=dias,satisfaccion=satis))
# faltantes
df.loc[rng.choice(n,180,replace=False),"satisfaccion"] = np.nan
df.to_csv("pedidos_mercaandina_2026.csv", index=False)

R = {}
t = df.ticket
R["desc"] = dict(n=int(t.count()), media=t.mean(), mediana=t.median(), trim=stats.trim_mean(t,0.10),
  de=t.std(), q1=t.quantile(.25), q3=t.quantile(.75), iqr=t.quantile(.75)-t.quantile(.25),
  cv=t.std()/t.mean()*100, mad=stats.median_abs_deviation(t, scale="normal"),
  asim=stats.skew(t), curt=stats.kurtosis(t), minimo=t.min(), maximo=t.max(), p95=t.quantile(.95), p99=t.quantile(.99))
q1,q3 = R["desc"]["q1"],R["desc"]["q3"]; iqr=q3-q1
R["out_iqr"] = int(((t<q1-1.5*iqr)|(t>q3+1.5*iqr)).sum())
z = (t-t.mean())/t.std(); R["out_z"]=int((z.abs()>3).sum())
zr = 0.6745*(t-t.median())/stats.median_abs_deviation(t); R["out_zr"]=int((zr.abs()>3.5).sum())
lt = np.log(t)
R["log_asim"]=stats.skew(lt)
samp = rng.choice(t,500,replace=False); samp_l=np.log(samp)
R["shapiro"]=stats.shapiro(samp).pvalue; R["shapiro_log"]=stats.shapiro(samp_l).pvalue
R["ad"]=stats.anderson(samp,dist="norm").statistic
R["dagostino"]=stats.normaltest(t).pvalue
# IC media e IC bootstrap mediana
ci = stats.t.interval(0.95, len(t)-1, loc=t.mean(), scale=stats.sem(t))
R["ic_media"]=ci
bs = stats.bootstrap((t.values,), np.median, n_resamples=2000, method="percentile", random_state=1)
R["ic_mediana"]=(bs.confidence_interval.low, bs.confidence_interval.high)
# Web vs App
w = df[df.canal=="Web"].ticket; a = df[df.canal=="App"].ticket
wt = stats.ttest_ind(a,w,equal_var=False)
R["welch"]=dict(t=wt.statistic,p=wt.pvalue,mw=w.mean(),ma=a.mean(),medw=w.median(),meda=a.median())
mwu = stats.mannwhitneyu(a,w); R["mwu_p"]=mwu.pvalue
R["mwu_rbc"] = 1 - 2*mwu.statistic/(len(a)*len(w))  # rank-biserial (signo)
sp = np.sqrt(((len(a)-1)*a.var()+(len(w)-1)*w.var())/(len(a)+len(w)-2)); R["cohen_d"]=(a.mean()-w.mean())/sp
R["levene"]=stats.levene(a,w).pvalue
# ANOVA / Kruskal por región (log ticket)
grupos=[np.log(df[df.region==r].ticket) for r in regiones]
fo = stats.f_oneway(*grupos); R["anova"]=dict(F=fo.statistic,p=fo.pvalue)
allv=np.concatenate(grupos); gm=allv.mean()
ssb=sum(len(g)*(g.mean()-gm)**2 for g in grupos); sst=((allv-gm)**2).sum(); R["eta2"]=ssb/sst
kw = stats.kruskal(*[df[df.region==r].ticket for r in regiones]); R["kw"]=dict(H=kw.statistic,p=kw.pvalue)
R["med_region"]={r: float(df[df.region==r].ticket.median()) for r in regiones}
R["n_region"]={r: int((df.region==r).sum()) for r in regiones}
# Pairwise Mann-Whitney vs Lima + corrección
pv=[];pares=[]
for r in regiones[1:]:
    pares.append(f"Lima vs {r}"); pv.append(stats.mannwhitneyu(df[df.region=="Lima"].ticket, df[df.region==r].ticket).pvalue)
# add a spurious comparison Norte vs Sur
pares.append("Norte vs Sur"); pv.append(stats.mannwhitneyu(df[df.region=="Norte"].ticket, df[df.region=="Sur"].ticket).pvalue)
_,pb,_,_=multipletests(pv,method="bonferroni"); _,ph,_,_=multipletests(pv,method="holm"); _,pf,_,_=multipletests(pv,method="fdr_bh")
R["mult"]=[dict(par=p,raw=a_,bonf=b,holm=h,bh=f) for p,a_,b,h,f in zip(pares,pv,pb,ph,pf)]
# Chi-cuadrado
ct = pd.crosstab(df.region, df.metodo_pago).loc[regiones, pagos]
chi = stats.chi2_contingency(ct); R["chi"]=dict(chi2=chi.statistic,p=chi.pvalue,gl=chi.dof)
R["cramer"]=stats.contingency.association(ct.values, method="cramer")
R["ct"]=ct.to_dict()
R["ct_pct"]=(pd.crosstab(df.region, df.metodo_pago, normalize="index").loc[regiones,pagos]*100).round(1).to_dict()
# Correlación
d2=df.dropna(subset=["satisfaccion"])
R["pearson"]=stats.pearsonr(d2.dias_entrega,d2.satisfaccion)
R["spearman"]=stats.spearmanr(d2.dias_entrega,d2.satisfaccion)
R["kendall"]=stats.kendalltau(d2.dias_entrega,d2.satisfaccion)
R["pear_ticket_satis"]=stats.pearsonr(d2.ticket,d2.satisfaccion)
R["n_na"]=int(df.satisfaccion.isna().sum())
# A/B test checkout
convA, nA = 1896, 20000; convB, nB = 2090, 20000
zst,pz = proportions_ztest([convB,convA],[nB,nA])
R["ab"]=dict(pA=convA/nA,pB=convB/nB,z=zst,p=pz,lift=(convB/nB)/(convA/nA)-1,
  icA=proportion_confint(convA,nA,method="wilson"), icB=proportion_confint(convB,nB,method="wilson"))
dif = convB/nB-convA/nA; se=np.sqrt(convA/nA*(1-convA/nA)/nA + convB/nB*(1-convB/nB)/nB)
R["ab"]["ic_dif"]=(dif-1.96*se,dif+1.96*se)
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
es=proportion_effectsize(0.1045,0.095); R["n_needed"]=NormalIndPower().solve_power(es,alpha=0.05,power=0.8)
# Big n paradox
big = rng.normal(100,20,1_000_000); big2=rng.normal(100.1,20,1_000_000)
tt=stats.ttest_ind(big,big2); R["bign"]=dict(p=tt.pvalue,d=(big2.mean()-big.mean())/20)

def conv(o):
    if isinstance(o,dict): return {str(k):conv(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [conv(v) for v in o]
    if hasattr(o,"_asdict"): return conv(tuple(o))
    try: return float(o)
    except: return str(o)
json.dump(conv(R),open("resultados.json","w"),indent=1,ensure_ascii=False)
print(json.dumps(conv(R),indent=1,ensure_ascii=False))

# ---- Gráficos (grises para Word, colores para PPT)
def graficos(sfx, c1, c2):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    ax[0].hist(t[t<t.quantile(.99)],bins=60,color=c1,edgecolor="white" if sfx=="c" else "black",linewidth=.3)
    ax[0].axvline(t.mean(),color=c2,ls="--",lw=2,label=f"Media = {t.mean():.1f}")
    ax[0].axvline(t.median(),color=c2,ls="-",lw=2,label=f"Mediana = {t.median():.1f}")
    ax[0].set_title("Ticket (S/) - hasta P99");ax[0].legend()
    ax[1].hist(lt,bins=60,color=c1,edgecolor="white" if sfx=="c" else "black",linewidth=.3); ax[1].set_title("log(Ticket)")
    for a_ in ax: a_.spines[["top","right"]].set_visible(False)
    plt.tight_layout();plt.savefig(f"img/hist_{sfx}.png",dpi=200);plt.close()
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    stats.probplot(samp,dist="norm",plot=ax[0]); ax[0].set_title("Q-Q: Ticket")
    stats.probplot(samp_l,dist="norm",plot=ax[1]); ax[1].set_title("Q-Q: log(Ticket)")
    for a_ in ax:
        a_.get_lines()[0].set_markerfacecolor(c1);a_.get_lines()[0].set_markeredgecolor(c1);a_.get_lines()[0].set_markersize(3)
        a_.set_xlabel("Cuantiles teóricos (normal)"); a_.set_ylabel("Valores ordenados"); a_.get_lines()[1].set_color(c2); a_.spines[["top","right"]].set_visible(False)
    plt.tight_layout();plt.savefig(f"img/qq_{sfx}.png",dpi=200);plt.close()
    fig,ax=plt.subplots(figsize=(10,3.8))
    data=[df[df.region==r].ticket for r in regiones]
    bp=ax.boxplot(data,tick_labels=regiones,showfliers=True,patch_artist=True,flierprops=dict(marker=".",markersize=2,markeredgecolor=c1))
    for b in bp["boxes"]: b.set_facecolor("white" if sfx=="g" else "#CFE8EE"); b.set_edgecolor(c1)
    for m in bp["medians"]: m.set_color(c2); m.set_linewidth(2)
    ax.set_yscale("log");ax.set_ylabel("Ticket S/ (escala log)");ax.set_title("Distribución del ticket por región")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout();plt.savefig(f"img/box_{sfx}.png",dpi=200);plt.close()
    fig,ax=plt.subplots(figsize=(10,3.6))
    bsd=bs.bootstrap_distribution
    ax.hist(bsd,bins=40,color=c1,edgecolor="white" if sfx=="c" else "black",linewidth=.3)
    for v in R["ic_mediana"]: ax.axvline(v,color=c2,ls="--",lw=2)
    ax.set_title("Distribución bootstrap de la mediana (2000 remuestreos) e IC 95%")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout();plt.savefig(f"img/boot_{sfx}.png",dpi=200);plt.close()
    fig,ax=plt.subplots(figsize=(10,3.6))
    m=d2.groupby("dias_entrega").satisfaccion.agg(["mean","count"]); m=m[m["count"]>=30]
    ax.plot(m.index,m["mean"],marker="o",color=c1,lw=2); ax.set_xlabel("Días de entrega");ax.set_ylabel("Satisfacción media (1-5)")
    ax.set_title(f"Días de entrega vs. satisfacción (Spearman ρ = {R['spearman'].statistic:.2f})")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout();plt.savefig(f"img/corr_{sfx}.png",dpi=200);plt.close()
graficos("g","#404040","black")
graficos("c","#1C7293","#1E2761")
