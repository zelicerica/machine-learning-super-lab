"""Inspectable machine learning laboratories. All bundled tasks fit on a central processing unit.
Run: python foundations.py --help. Outputs are measured, never illustrative placeholders.
"""
from __future__ import annotations
import argparse, json, math, platform, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import __version__ as sklearn_version
from threadpoolctl import threadpool_limits
from sklearn.datasets import load_digits, make_blobs, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, silhouette_score, adjusted_rand_score

EXPERIMENTS=('linear','logistic','tree','ensemble','neighbors','kernel','bayes','gaussian','cluster','projection','dense','convolution','graph','autoencoder','variational','adversarial','ngram','reinforcement')
plt.rcParams.update({'figure.facecolor':'#f7f8f2','axes.facecolor':'#ffffff','axes.edgecolor':'#b8c6aa','axes.labelcolor':'#35472f','text.color':'#35472f','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def partition(n,seed,labels=None):
    """60/20/20; retain test indices without fitting anything to them."""
    indices=np.arange(n)
    train,held=train_test_split(indices,test_size=.4,random_state=seed,stratify=labels)
    valid,test=train_test_split(held,test_size=.5,random_state=seed+1,stratify=None if labels is None else labels[held])
    return train,valid,test

def selected(parts,split):return parts[1] if split=='validation' else parts[2]
def figure(a,name):
    plt.tight_layout();plt.savefig(a.out/name,dpi=140,bbox_inches='tight');plt.close();return name

def classification(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(y,p).tolist()}

def confusion(a,y,p,title):
    m=confusion_matrix(y,p);fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(m,cmap='YlGn');fig.colorbar(im,ax=ax)
    ax.set(title=title,xlabel='Predicted class',ylabel='Actual class',xticks=range(len(m)),yticks=range(len(m)))
    if len(m)<=10:
        for i in range(len(m)):
            for j in range(len(m)):ax.text(j,i,str(m[i,j]),ha='center',va='center',fontsize=8,color='white' if m[i,j]>m.max()/2 else '#253a20')
    return figure(a,'confusion.png')

def regression(y,p):return {'mean_absolute_error':float(mean_absolute_error(y,p)), 'root_mean_squared_error':float(mean_squared_error(y,p)**.5),'r_squared':float(r2_score(y,p))}

def payload(a,metrics,parts=None,dataset='',notes='',artifacts=None,history=None):
    result={'schema':'superlab.report.v1','experiment':a.experiment,'split':a.split,'seed':a.seed,'dataset':dataset,'metrics':metrics,'limitations':notes,'artifacts':artifacts or [],'history':history or [],'settings':{k:v for k,v in vars(a).items() if k not in ('out','output')}}
    if parts is not None:result['split_sizes']={'train':len(parts[0]),'validation':len(parts[1]),'test':len(parts[2])}
    if a.experiment=='adversarial':
        result.update(split='independent_target_samples',evaluation={'protocol':'Fresh samples from the same known simulator; not a reserved final test','target_samples':1500,'generated_samples':1500,'evaluation_seed':a.seed+1})
    elif a.experiment=='reinforcement':
        result.update(split='same_environment_policy_evaluation',evaluation={'protocol':'Same deterministic map and fixed start; not generalization to new tasks','episodes_per_policy':100,'start_state':0,'step_cap':50})
    return result

def export_manifest(a,kind,sample,method='predict',output='values'):
    contracts={
        'linear':'Rows of [record_count, transformation_count, compression_setting], in that order.',
        'gaussian':'Rows containing one sensor-position value in the training units.',
        'neighbors':'Rows of two measurements; the saved pipeline applies the training scaler.',
        'kernel':'Rows of two measurements; the saved pipeline applies the training scaler.',
        'cluster':'Rows of two measurements; outputs are arbitrary K-means cluster IDs.',
        'graph':'All 180 node feature rows in the original ordering; the saved graph structure is fixed.',
        'adversarial':'Rows of two noise values; each row produces one synthetic scalar.',
        'reinforcement':'Integer grid states from 0 through 15; outputs use action order up, right, down, left.',
        'ngram':'A text context using the saved word tokenizer and vocabulary.'}
    contract=contracts.get(a.experiment,'Rows of eight numeric event measurements in the feature order shown in example-input.json.')
    if a.experiment in ('projection','dense','convolution','autoencoder','variational'):
        contract='Rows of 64 grayscale values normalized to 0 through 1, flattened from 8-by-8 images.'
    (a.out/'model-manifest.json').write_text(json.dumps({'format':kind,'experiment':a.experiment,'method':method,'output':output,'input_contract':contract,'trusted_artifacts_only':True},indent=2))
    (a.out/'example-input.json').write_text(json.dumps(sample.tolist() if hasattr(sample,'tolist') else sample,indent=2))

def export_sklearn(a,models,sample,method='predict'):
    import joblib
    joblib.dump(models,a.out/'model.joblib');export_manifest(a,'joblib',sample,method)

def export_torch(a,model,sample,output='values'):
    import torch
    model=model.to('cpu').eval();sample=sample.detach().cpu()
    torch.jit.trace(model,sample).save(str(a.out/'model.pt'))
    export_manifest(a,'torchscript',sample.numpy(),output=output)

def linear(a):
    from sklearn.linear_model import LinearRegression
    rng=np.random.default_rng(a.seed);x=rng.uniform([0,0,0],[100,10,1],size=(600,3))
    y=5+2*x[:,0]+3*x[:,1]+8*x[:,2]+rng.normal(0,a.noise,len(x));parts=partition(len(x),a.seed);tr=parts[0];ev=selected(parts,a.split)
    model=LinearRegression().fit(x[tr],y[tr]);pred=model.predict(x[ev]);baseline=np.full(len(ev),y[tr].mean())
    export_sklearn(a,{'linear':model},x[ev[:2]])
    plt.figure(figsize=(6,4));plt.scatter(y[ev],pred,s=14,alpha=.6,color='#52782c');bounds=[min(y[ev].min(),pred.min()),max(y[ev].max(),pred.max())];plt.plot(bounds,bounds,color='#aa694d');plt.xlabel('Actual simulated duration');plt.ylabel('Predicted duration');plt.title('Prediction error: distance from the diagonal')
    return payload(a,regression(y[ev],pred)|{'baseline_mean_absolute_error':float(mean_absolute_error(y[ev],baseline)),'learned_weights':model.coef_.tolist(),'learned_bias':float(model.intercept_)},parts,'Synthetic processing duration: 3 known features','The relationship was generated to be linear. This is not an operational service-time benchmark.',[figure(a,'predictions.png')])

def tabular(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    if a.experiment in ('neighbors','kernel'):
        x,y=make_moons(n_samples=650,noise=a.noise,random_state=a.seed);dataset='Synthetic interleaving half-circles: two measured features'
    else:
        x,y=make_classification(n_samples=900,n_features=8,n_informative=5,n_redundant=1,weights=[.65,.35],flip_y=min(a.noise,.35),class_sep=1.15,random_state=a.seed);dataset='Synthetic labeled events: 8 generated features; not real security telemetry'
    parts=partition(len(x),a.seed,y);tr=parts[0];ev=selected(parts,a.split)
    models={
      'logistic':lambda:{'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=a.strength,max_iter=1000,random_state=a.seed))},
      'tree':lambda:{'tree':DecisionTreeClassifier(max_depth=a.depth,random_state=a.seed)},
      'ensemble':lambda:{'single_tree':DecisionTreeClassifier(max_depth=a.depth,random_state=a.seed),'random_forest':RandomForestClassifier(n_estimators=80,max_depth=a.depth,random_state=a.seed,n_jobs=1),'gradient_boosting':HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,random_state=a.seed)},
      'neighbors':lambda:{'neighbors':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=a.neighbors))},
      'kernel':lambda:{'linear_kernel':make_pipeline(StandardScaler(),SVC(kernel='linear',C=a.strength)),'radial_basis_kernel':make_pipeline(StandardScaler(),SVC(kernel='rbf',C=a.strength,gamma='scale'))},
      'bayes':lambda:{'naive_bayes':GaussianNB(),'logistic_comparison':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=a.seed))},
    }[a.experiment]()
    metrics={};predictions={}
    for name,model in models.items():
        model.fit(x[tr],y[tr]);p=model.predict(x[ev]);predictions[name]=p;metrics[name]=classification(y[ev],p)|{'training_accuracy':float(model.score(x[tr],y[tr]))}
    metrics['majority_baseline_accuracy']=float(np.mean(y[ev]==np.bincount(y[tr]).argmax()))
    export_sklearn(a,models,x[ev[:2]])
    last=list(models)[-1];artifacts=[confusion(a,y[ev],predictions[last],f'{last}: {a.split} confusion matrix')]
    if a.experiment in ('neighbors','kernel'):
        fig,axes=plt.subplots(1,len(models),figsize=(6*len(models),4),squeeze=False)
        gx,gy=np.meshgrid(np.linspace(x[:,0].min()-.4,x[:,0].max()+.4,150),np.linspace(x[:,1].min()-.4,x[:,1].max()+.4,150))
        for ax,(name,model) in zip(axes[0],models.items()):
            ax.contourf(gx,gy,model.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape),alpha=.18,cmap='viridis');ax.scatter(x[ev,0],x[ev,1],c=y[ev],s=14,cmap='viridis');ax.set(title=name,xlabel='Feature one',ylabel='Feature two')
        artifacts.append(figure(a,'boundaries.png'))
    elif a.experiment=='tree':
        plt.figure(figsize=(14,7));plot_tree(models['tree'],max_depth=2,filled=True,feature_names=[f'feature {i}' for i in range(8)],fontsize=8);plt.title('First three displayed levels; deeper branches may be omitted');artifacts.append(figure(a,'tree.png'))
    else:
        plt.figure(figsize=(7,4));names=list(models);plt.bar(names,[metrics[n]['accuracy'] for n in names],color='#91b95d');plt.ylim(0,1);plt.ylabel('Accuracy');plt.title(f'{a.split.title()} model comparison');artifacts.append(figure(a,'comparison.png'))
    return payload(a,metrics,parts,dataset,'Use validation for choices; inspect final test only after freezing settings. Generated features do not establish a real-world fraud or security detector.',artifacts)

def gaussian(a):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    rng=np.random.default_rng(a.seed);x=np.linspace(0,8,150)[:,None];y=np.sin(x[:,0])+rng.normal(0,a.noise,len(x));parts=partition(len(x),a.seed);tr=parts[0];ev=selected(parts,a.split)
    model=GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(max(a.noise**2,.001)),normalize_y=True,random_state=a.seed).fit(x[tr],y[tr]);p,std=model.predict(x[ev],return_std=True)
    grid=np.linspace(-2,10,250)[:,None];mean,sigma=model.predict(grid,return_std=True)
    export_sklearn(a,{'gaussian_process':model},x[ev[:2]])
    plt.figure(figsize=(8,4));plt.scatter(x[tr,0],y[tr],s=10,label='Training observations');plt.plot(grid[:,0],mean,label='Predictive mean',color='#52782c');plt.fill_between(grid[:,0],mean-1.96*sigma,mean+1.96*sigma,alpha=.25,color='#a4c76e',label='Model-based 95% interval');plt.xlabel('Synthetic sensor position');plt.ylabel('Reading');plt.legend(fontsize=8)
    return payload(a,regression(y[ev],p)|{'interval_coverage':float(np.mean(abs(y[ev]-p)<=1.96*std)),'mean_predictive_standard_deviation':float(std.mean())},parts,'Synthetic noisy sine measurements','Intervals rely on the chosen kernel and noise assumptions. A nominal 95% interval need not attain 95% empirical coverage.',[figure(a,'uncertainty.png')])

def cluster(a):
    from sklearn.cluster import KMeans, DBSCAN
    x,truth=make_blobs(n_samples=450,centers=3,n_features=2,cluster_std=a.noise,random_state=a.seed);parts=partition(len(x),a.seed);tr=parts[0];ev=selected(parts,a.split)
    scaler=StandardScaler().fit(x[tr]);xt=scaler.transform(x[tr]);xe=scaler.transform(x[ev]);model=KMeans(n_clusters=a.clusters,n_init=10,random_state=a.seed).fit(xt);labels=model.predict(xe)
    density=DBSCAN(eps=.35,min_samples=5).fit_predict(xt)
    export_sklearn(a,{'kmeans':make_pipeline(scaler,model)},x[ev[:2]])
    score=float(silhouette_score(xe,labels)) if 1<len(np.unique(labels))<len(xe) else None
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].scatter(xe[:,0],xe[:,1],c=labels,cmap='viridis',s=20);axes[0].set_title('K-means: held-out assignments');axes[1].scatter(xt[:,0],xt[:,1],c=density,cmap='viridis',s=14);axes[1].set_title('Density clustering: training set only')
    for ax in axes:ax.set(xlabel='Standardized feature one',ylabel='Standardized feature two')
    return payload(a,{'heldout_silhouette':score,'heldout_adjusted_rand_index':float(adjusted_rand_score(truth[ev],labels)),'training_inertia':float(model.inertia_),'density_noise_points':int((density==-1).sum())},parts,'Synthetic three-cluster observations; true groups used only for evaluation','Density-based spatial clustering of applications with noise has no built-in predict method here. Its panel is an exploratory fit on training data. Cluster numbers have no intrinsic business meaning.',[figure(a,'clusters.png')])

def projection(a):
    from sklearn.decomposition import PCA
    x,y=load_digits(return_X_y=True);x=x/16;parts=partition(len(x),a.seed,y);tr=parts[0];ev=selected(parts,a.split)
    model=PCA(n_components=a.components,random_state=a.seed).fit(x[tr]);z=model.transform(x[ev]);restored=model.inverse_transform(z);loss=float(np.mean((x[ev]-restored)**2))
    export_sklearn(a,{'principal_components':model},x[ev[:2]],method='transform')
    fig,axes=plt.subplots(2,8,figsize=(10,3))
    for i in range(8):
        axes[0,i].imshow(x[ev[i]].reshape(8,8),cmap='gray',vmin=0,vmax=1);axes[1,i].imshow(restored[i].reshape(8,8),cmap='gray',vmin=0,vmax=1)
        axes[0,i].axis('off');axes[1,i].axis('off')
    fig.suptitle('Original digits above; reconstructed digits below')
    return payload(a,{'reconstruction_mean_squared_error':loss,'retained_training_variance':float(model.explained_variance_ratio_.sum()),'input_dimensions':64,'representation_dimensions':a.components},parts,'scikit-learn bundled handwritten digits: 1797 images, 8 by 8 pixels','Variance retention is not classification accuracy. Fitting principal components before the split leaks information.',[figure(a,'reconstructions.png')])

# Neural networks below are deliberately short enough to inspect end to end.
def neural_setup(a):
    import torch
    torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('Graphics processor requested but unavailable. Use --device cpu or install the appropriate graphics-enabled PyTorch on the rented machine.')
    return torch,torch.device(a.device)

def digits(a):
    x,y=load_digits(return_X_y=True);x=(x/16).astype('float32');parts=partition(len(x),a.seed,y);return x,y,parts

def neural_classifier(a):
    torch,device=neural_setup(a);nn=torch.nn;x,y,parts=digits(a);tr=parts[0];ev=selected(parts,a.split)
    xt=torch.tensor(x[tr],device=device);yt=torch.tensor(y[tr],device=device);xe=torch.tensor(x[ev],device=device)
    if a.experiment=='convolution':
        model=nn.Sequential(nn.Unflatten(1,(1,8,8)),nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(16*4*4,10)).to(device)
    else:model=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,10)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01);history=[]
    for epoch in range(a.epochs):
        model.train();order=torch.randperm(len(tr),device=device);total=0
        for ids in order.split(64):
            optimizer.zero_grad();loss=nn.functional.cross_entropy(model(xt[ids]),yt[ids]);loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
        history.append({'step':epoch+1,'loss':total/len(tr)})
    model.eval()
    with torch.no_grad():p=model(xe).argmax(1).cpu().numpy()
    torch.save(model.state_dict(),a.out/'weights.pt')
    export_torch(a,model,xe[:2],output='class_index')
    return payload(a,classification(y[ev],p)|{'trainable_parameters':sum(p.numel() for p in model.parameters())},parts,'Bundled handwritten digits: 8 by 8 pixels','Small digit recognition task, not a document-recognition product. A saved parameter dictionary needs the matching architecture to reload.',[confusion(a,y[ev],p,f'{a.experiment}: held-out digits')],history)

def graph(a):
    torch,device=neural_setup(a);nn=torch.nn;rng=np.random.default_rng(a.seed);n=180;y=np.repeat(np.arange(3),n//3);x=rng.normal(0,.8,(n,6)).astype('float32');x[np.arange(n),y]+=.65
    probability=np.where(y[:,None]==y[None,:],.15,.008);upper=np.triu(rng.random((n,n))<probability,1);adj=(upper|upper.T).astype('float32')
    if a.ablate_edges:adj[:]=0
    adj+=np.eye(n);degree=adj.sum(1);normalized=adj/np.sqrt(degree[:,None]*degree[None,:]);at=torch.tensor(normalized,device=device);xt=torch.tensor(x,device=device);yt=torch.tensor(y,device=device);parts=partition(n,a.seed,y);tr=torch.tensor(parts[0],device=device);ev=selected(parts,a.split)
    class GraphNetwork(nn.Module):
        def __init__(self):super().__init__();self.first=nn.Linear(6,16);self.second=nn.Linear(16,3)
        def forward(self,features):return self.second(at@torch.relu(self.first(at@features)))
    model=GraphNetwork().to(device);opt=torch.optim.Adam(model.parameters(),lr=.03);history=[]
    for epoch in range(a.epochs):
        opt.zero_grad();logits=model(xt);loss=nn.functional.cross_entropy(logits[tr],yt[tr]);loss.backward();opt.step();history.append({'step':epoch+1,'loss':float(loss.detach())})
    with torch.no_grad():p=model(xt)[ev].argmax(1).cpu().numpy()
    # Store the fixed graph with the traced inference module; this is transductive.
    at=at.cpu()
    export_torch(a,model,xt,output='class_index')
    plt.figure(figsize=(6,5));plt.imshow(adj,cmap='Greens');plt.xlabel('Node identifier');plt.ylabel('Node identifier');plt.title('Adjacency including self-connections')
    return payload(a,classification(y[ev],p)|{'edges_excluding_self':int((adj.sum()-n)/2),'transductive':True},parts,'Synthetic community graph with noisy features','Transductive setting: all node features and edges are visible, but only training node labels enter the loss. New unseen graphs need a separate inductive assessment. Graph structure was generated from hidden communities.',[figure(a,'adjacency.png')],history)

def representation(a):
    torch,device=neural_setup(a);nn=torch.nn;x,y,parts=digits(a);tr=parts[0];ev=selected(parts,a.split);xt=torch.tensor(x[tr],device=device);xe=torch.tensor(x[ev],device=device);variational=a.experiment=='variational'
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Sequential(nn.Linear(64,32),nn.ReLU());self.mean=nn.Linear(32,a.latent);self.logvar=nn.Linear(32,a.latent);self.decoder=nn.Sequential(nn.Linear(a.latent,32),nn.ReLU(),nn.Linear(32,64),nn.Sigmoid())
        def forward(self,v,sample=True):
            h=self.encoder(v);mu=self.mean(h);lv=self.logvar(h).clamp(-10,10);z=mu+torch.exp(.5*lv)*torch.randn_like(mu) if variational and sample else mu
            return self.decoder(z),mu,lv
    model=Autoencoder().to(device);opt=torch.optim.Adam(model.parameters(),lr=.003);history=[]
    for epoch in range(a.epochs):
        total=0
        for ids in torch.randperm(len(tr),device=device).split(64):
            original=xt[ids];noisy=(original+a.noise*torch.randn_like(original)).clamp(0,1) if not variational else original
            reconstruction,mu,lv=model(noisy);reconstruction_loss=(reconstruction-original).square().sum(1).mean();divergence=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean();loss=reconstruction_loss+(a.beta*divergence if variational else 0)
            opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(ids)
        history.append({'step':epoch+1,'loss':total/len(tr)})
    with torch.no_grad():
        inputs=(xe+a.noise*torch.randn_like(xe)).clamp(0,1) if not variational else xe
        reconstructed,mu,lv=model(inputs,sample=False);generated=model.decoder(torch.randn(8,a.latent,device=device));error=float((reconstructed-xe).square().mean());baseline=float((xt.mean(0)-xe).square().mean())
    fig,axes=plt.subplots(3,8,figsize=(10,4))
    for i in range(8):
        for row,data in enumerate([xe,reconstructed,generated if variational else inputs]):axes[row,i].imshow(data[i].detach().cpu().numpy().reshape(8,8),cmap='gray',vmin=0,vmax=1);axes[row,i].axis('off')
    fig.suptitle('Original / reconstruction / '+('prior samples' if variational else 'noisy input'))
    torch.save(model.state_dict(),a.out/'weights.pt')
    export_torch(a,nn.Sequential(model.encoder,model.mean,model.decoder),inputs[:2])
    return payload(a,{'reconstruction_mean_squared_error':error,'mean_image_baseline_error':baseline,'latent_dimensions':a.latent,'labels_used_for_training':False},parts,'Bundled handwritten digits; labels used only for stratified splitting','Reconstruction is evaluated at the latent mean. Plausible-looking prior samples are not proof of coverage, privacy, or downstream utility. Labels are not reconstruction targets.',[figure(a,'representations.png')],history)

def adversarial(a):
    torch,device=neural_setup(a);nn=torch.nn;rng=np.random.default_rng(a.seed)
    def real(n,generator=None):
        modes=torch.randint(0,2,(n,1),device=device,generator=generator);return torch.randn(n,1,device=device,generator=generator)*.4+torch.where(modes==0,-2.,2.)
    generator=nn.Sequential(nn.Linear(2,32),nn.LeakyReLU(.2),nn.Linear(32,32),nn.LeakyReLU(.2),nn.Linear(32,1)).to(device)
    discriminator=nn.Sequential(nn.Linear(1,32),nn.LeakyReLU(.2),nn.Linear(32,16),nn.LeakyReLU(.2),nn.Linear(16,1)).to(device)
    go=torch.optim.Adam(generator.parameters(),lr=.001,betas=(.5,.999));do=torch.optim.Adam(discriminator.parameters(),lr=.001,betas=(.5,.999));bce=nn.BCEWithLogitsLoss();history=[]
    for step in range(a.steps):
        observations=real(128);noise=torch.randn(128,2,device=device);fake=generator(noise)
        do.zero_grad();dl=bce(discriminator(observations),torch.ones_like(observations))+bce(discriminator(fake.detach()),torch.zeros_like(fake));dl.backward();do.step()
        for p in discriminator.parameters():p.requires_grad_(False)
        go.zero_grad();fake=generator(torch.randn(128,2,device=device));gl=bce(discriminator(fake),torch.ones_like(fake));gl.backward();go.step()
        for p in discriminator.parameters():p.requires_grad_(True)
        if step%20==0 or step==a.steps-1:history.append({'step':step+1,'loss':float(gl.detach()),'discriminator_loss':float(dl.detach())})
    evaluation_rng=torch.Generator(device=device).manual_seed(a.seed+1)
    with torch.no_grad():generated=generator(torch.randn(1500,2,device=device,generator=evaluation_rng)).cpu().numpy().ravel();observed=real(1500,evaluation_rng).cpu().numpy().ravel()
    plt.figure(figsize=(8,4));plt.hist(observed,bins=45,density=True,alpha=.5,label='Independent target samples');plt.hist(generated,bins=45,density=True,alpha=.5,label='Generated samples');plt.legend();plt.xlabel('Synthetic scalar value');plt.ylabel('Density');plt.title('Check both modes, not just a decreasing loss')
    torch.save(generator.state_dict(),a.out/'generator.pt')
    export_torch(a,generator,torch.randn(2,2,device=device))
    return payload(a,{'fraction_generated_below_zero':float((generated<0).mean()),'generated_mean':float(generated.mean()),'target_sample_mean':float(observed.mean()),'generated_standard_deviation':float(generated.std()),'target_standard_deviation':float(observed.std())},dataset='Two synthetic Gaussian modes at -2 and +2',notes='This one-dimensional adversarial training lab can collapse to one mode. Loss alone is not a quality score. Independent target samples are drawn after optimization; the task is not image generation.',artifacts=[figure(a,'distribution.png')],history=history)

def ngram(a):
    """Add-one-smoothed word model with a bounded context; training is counting."""
    rng=np.random.default_rng(a.seed);subjects=['service','worker','router','server'];verbs=['starts','stops','retries','waits'];modifiers=['quickly','slowly','safely','again']
    texts=[f'the {s} {v} {m}' for s in subjects for v in verbs for m in modifiers];parts=partition(len(texts),a.seed);tr=parts[0];ev=selected(parts,a.split);vocab=sorted(set(' '.join(texts[i] for i in tr).split())|{'</s>','<unk>'});counts=defaultdict(Counter);context_size=a.order-1
    def tokens(text):return [w if w in vocab else '<unk>' for w in text.split()]+['</s>']
    for i in tr:
        words=['<s>']*context_size+tokens(texts[i])
        for j in range(context_size,len(words)):counts[tuple(words[max(0,j-context_size):j])][words[j]]+=1
    loss=0;n=0
    for i in ev:
        words=['<s>']*context_size+tokens(texts[i])
        for j in range(context_size,len(words)):
            row=counts[tuple(words[max(0,j-context_size):j])];probability=(row[words[j]]+1)/(sum(row.values())+len(vocab));loss-=math.log(probability);n+=1
    context=['<s>']*context_size;sample=[]
    for _ in range(20):
        row=counts[tuple(context[-context_size:]) if context_size else ()];p=np.array([row[w]+1 for w in vocab],dtype=float);w=str(rng.choice(vocab,p=p/p.sum()))
        if w=='</s>':break
        sample.append(w);context.append(w)
    example_context=tuple(['<s>']*context_size);row=counts[example_context];p=np.array([row[w]+1 for w in vocab],dtype=float);p/=p.sum()
    plt.figure(figsize=(9,4));plt.bar(vocab,p,color='#8aae5e');plt.xticks(rotation=45,ha='right');plt.ylabel('Smoothed probability');plt.title('First-token distribution from training counts')
    (a.out/'counts.json').write_text(json.dumps({'order':a.order,'vocabulary':vocab,'rows':[{'context':list(k),'counts':dict(v)} for k,v in counts.items()]}))
    export_manifest(a,'ngram','the server')
    return payload(a,{'perplexity':float(math.exp(loss/n)),'uniform_baseline_perplexity':len(vocab),'vocabulary_size':len(vocab),'context_tokens':context_size,'generated_text':' '.join(sample)},parts,'64 distinct synthetic four-word operations sentences','Vocabulary is learned from training sentences; unknown evaluation words map to an unknown symbol. Add-one smoothing prevents zero probabilities but does not guarantee fluent text.',[figure(a,'token-probabilities.png')])

# A complete small decision environment, without external simulator dependencies.
def grid_step(state,action,size=4):
    row,col=divmod(state,size);dr,dc=[(-1,0),(0,1),(1,0),(0,-1)][action];nr,nc=np.clip([row+dr,col+dc],0,size-1);nxt=int(nr*size+nc);done=nxt==size*size-1
    return nxt,(10.0 if done else -1.0),done

def q_update(table,state,action,reward,next_state,done,rate=.2,discount=.95):
    target=reward+(0 if done else discount*table[next_state].max());table[state,action]+=rate*(target-table[state,action])

def reinforcement(a):
    rng=np.random.default_rng(a.seed);table=np.zeros((16,4));history=[]
    for episode in range(a.episodes):
        state=0;total=0;epsilon=max(.05,1-episode/(a.episodes*.8))
        for _ in range(50):
            action=int(rng.integers(4)) if rng.random()<epsilon else int(table[state].argmax());nxt,reward,done=grid_step(state,action);q_update(table,state,action,reward,nxt,done);state=nxt;total+=reward
            if done:break
        history.append({'episode':episode+1,'return':total})
    returns=[];baseline=[];success=0
    for policy in ('learned','random'):
        values=[]
        for _ in range(100):
            state=0;total=0
            for step in range(50):
                action=int(table[state].argmax()) if policy=='learned' else int(rng.integers(4));state,reward,done=grid_step(state,action);total+=reward
                if done:break
            values.append(total)
            if policy=='learned':success+=int(done)
        if policy=='learned':returns=values
        else:baseline=values
    plt.figure(figsize=(8,4));window=min(25,len(history));series=np.array([h['return'] for h in history]);plt.plot(np.arange(window,len(series)+1),np.convolve(series,np.ones(window)/window,mode='valid'),color='#648f36');plt.xlabel('Training episode');plt.ylabel(f'Mean reward over {window} episodes');plt.title('Training reward: exploration is enabled')
    (a.out/'action_values.json').write_text(json.dumps(table.tolist(),indent=2))
    export_manifest(a,'action_table',[0,1,4])
    return payload(a,{'greedy_success_rate':success/100,'greedy_mean_return':float(np.mean(returns)),'random_mean_return':float(np.mean(baseline)),'greedy_policy':table.argmax(1).reshape(4,4).tolist(),'action_order':['up','right','down','left']},dataset='Deterministic four-by-four grid; upper-left start and lower-right goal',notes='Evaluation repeats the same environment without exploration. It verifies learned behavior here, not generalization to new maps or physical robots.',artifacts=[figure(a,'rewards.png')])

def run(a):
    if a.experiment in ('adversarial','reinforcement'):
        if a.split is not None:raise ValueError('--split does not apply to this simulator evaluation; see the lesson protocol.')
    else:a.split=a.split or 'validation'
    a.out=Path(a.output)
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('Choose a new output folder to preserve previous evidence.')
    a.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    fn={'linear':linear,'gaussian':gaussian,'cluster':cluster,'projection':projection,'dense':neural_classifier,'convolution':neural_classifier,'graph':graph,'autoencoder':representation,'variational':representation,'adversarial':adversarial,'ngram':ngram,'reinforcement':reinforcement}.get(a.experiment,tabular)
    with threadpool_limits(limits=a.threads):result=fn(a)
    from predict_saved import predict
    result['reload_prediction']=predict(a.out,json.loads((a.out/'example-input.json').read_text()))
    result['reload_verified']=True
    result['seconds']=time.perf_counter()-start;result['environment']={'python':platform.python_version(),'numpy':np.__version__,'scikit_learn':sklearn_version};result['output_directory']=str(a.out).replace('\\','/')
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2,allow_nan=False));return result

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',choices=EXPERIMENTS);p.add_argument('--output',default='outputs/foundation-run');p.add_argument('--seed',type=int,default=42);p.add_argument('--split',choices=['validation','test'],default=None);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=2);p.add_argument('--epochs',type=int,default=20);p.add_argument('--steps',type=int,default=600);p.add_argument('--episodes',type=int,default=500);p.add_argument('--depth',type=int,default=4);p.add_argument('--neighbors',type=int,default=5);p.add_argument('--clusters',type=int,default=3);p.add_argument('--components',type=int,default=16);p.add_argument('--latent',type=int,default=8);p.add_argument('--order',type=int,default=2);p.add_argument('--noise',type=float,default=.12);p.add_argument('--strength',type=float,default=1);p.add_argument('--beta',type=float,default=.1);p.add_argument('--ablate-edges',action='store_true');return p

if __name__=='__main__':
    args=parser().parse_args()
    for name,maximum in [('epochs',500),('steps',20000),('episodes',20000),('depth',30),('neighbors',100),('clusters',12),('components',64),('latent',64),('order',5),('threads',32)]:
        if not 1<=getattr(args,name)<=maximum:raise SystemExit(f'{name} must be between 1 and {maximum}')
    if not 0<=args.noise<=(100 if args.experiment=='linear' else 5) or args.strength<=0 or args.beta<0:raise SystemExit('Noise, regularization strength, or divergence weight is outside the supported range.')
    run(args)
