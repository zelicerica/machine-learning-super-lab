"""Rebuild the exact atlas relationship graphic; requires Matplotlib only."""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
fig.patch.set_facecolor('#f4f7f1')
ax.set(xlim=(0, 1200), ylim=(800, 0))
ax.axis('off')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

def box(x, y, w, h, color, edge='#cdd8c6'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=16',
                             linewidth=1.3,facecolor=color,edgecolor=edge))

def text(x, y, value, size=17, color='#20352b', weight='normal'):
    ax.text(x,y,value,fontsize=size,color=color,fontweight=weight,va='top',
            fontfamily='DejaVu Sans',linespacing=1.5)

text(48,30,'Language models belong to machine learning',28,weight='bold')
text(48,82,'One task, several ways to learn. A language model is not necessarily a neural network.',16)
box(40,134,1120,548,'#e8efdf')
text(64,155,'MACHINE LEARNING',18,weight='bold')
text(64,192,'Learn from examples; evaluate on new inputs.',16)

box(64,245,516,302,'#ffffff')
text(88,269,'Count-based models',22,weight='bold')
text(88,316,'Estimate probabilities from token counts.\nAn n-gram uses a short context window.',15)
box(88,391,468,127,'#edf4f8','#cadce6')
text(108,408,'Operations-note completion',16,weight='bold')
text(108,446,'“server” → probabilities for the next word\nLearns from data; no neural layers required.',14)

box(608,245,528,302,'#ffffff')
text(632,269,'Neural language models',20,weight='bold')
text(632,316,'Learn numerical weights and representations.\nFeed-forward, recurrent, attention, and more.',14)
box(632,391,480,127,'#dcebc9','#afc58f')
text(652,408,'DEEP LANGUAGE MODELS',16,weight='bold')
text(652,446,'Use deep learning: multiple learned layers.\nExample: Transformer language models.',14)

text(66,574,'The wider field also includes:',18,weight='bold')
text(66,614,'Trees, regression, clustering, image classifiers, and many other learned models.',16)

text(48,707,'Read model labels as different dimensions',20,weight='bold')
text(48,750,'Language modeling = task  •  Transformer = architecture  •  Quantization = representation',15)
target = ROOT/'course/images/expanded'
target.mkdir(parents=True,exist_ok=True)
for extension in ('svg','png'):
    path=target/f'language-within-ml.{extension}'
    fig.savefig(path,facecolor=fig.get_facecolor())
    destination=ROOT/'public/images/expanded'/path.name
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,destination)
plt.close(fig)
