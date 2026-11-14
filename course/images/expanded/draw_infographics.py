"""Generate exact, self-contained teaching SVG assets using Python's standard library."""
from pathlib import Path
from html import escape
import math
ROOT=Path(__file__).parent
BG='#F5F7F1'; INK='#24352B'; MUTED='#64775F'; LINE='#DCE5D3'; GREEN='#C7ED91'; PALE='#EAF3DE'; RUST='#A66947'; WHITE='#FFFFFF'

class Figure:
    def __init__(self,name,title,subtitle,height,desc):
        self.name=name; self.height=height
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title description">',f'<title id="title">{escape(title)}</title><desc id="description">{escape(desc)}</desc>',f'<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="{MUTED}"/></marker></defs>',f'<rect width="1200" height="{height}" fill="{BG}"/>']
        self.text(44,42,'MACHINE LEARNING SUPER LAB',16,MUTED,700,spacing=2)
        self.text(44,94,title,36,INK,700)
        self.text(44,128,subtitle,18,MUTED)
    def text(self,x,y,lines,size=20,color=INK,weight=400,anchor='start',spacing=None):
        if isinstance(lines,str):lines=[lines]
        extra=f' letter-spacing="{spacing}"' if spacing is not None else ''
        for i,line in enumerate(lines): self.parts.append(f'<text x="{x}" y="{y+i*size*1.45}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}"{extra}>{escape(str(line))}</text>')
    def rect(self,x,y,w,h,fill=WHITE,stroke=LINE,r=18):self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
    def line(self,x1,y1,x2,y2,arrow=False,dash=False,color=MUTED,width=2):self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="6 5"' if dash else '')+'/>')
    def path(self,d,color=MUTED,width=2,fill='none',arrow=False):self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def circle(self,x,y,r=9,fill=GREEN,stroke=INK):self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>')
    def card(self,x,y,w,h,num,title,subtitle=None):
        self.rect(x,y,w,h);self.rect(x+22,y+23,42,32,PALE,PALE,8);self.text(x+43,y+46,num,18,MUTED,700,'middle');self.text(x+80,y+47,title,23,INK,700)
        if subtitle:self.text(x+24,y+88,subtitle,18,MUTED)
    def box(self,x,y,w,h,label,fill=PALE,size=18):
        self.rect(x,y,w,h,fill,LINE,10);lines=label if isinstance(label,list) else [label];self.text(x+w/2,y+h/2-(len(lines)-1)*size*.725+size*.35,lines,size,INK,600,'middle')
    def save(self,footnote):
        self.line(44,self.height-61,1156,self.height-61,color=LINE)
        self.text(44,self.height-30,footnote,16,MUTED)
        self.parts.append('</svg>');(ROOT/(self.name+'.svg')).write_text('\n'.join(self.parts),encoding='utf-8')

f=Figure('learning-map','Four questions describe one model','Use independent dimensions. A learning method is not the same thing as an architecture.',940,'Four independent dimensions describe a model: computational mechanism, learning signal, task, and adaptation or deployment. A single handwritten-digit classifier is described by all four.')
items=[(44,170,'01','What computes?',['The mathematical mechanism'],['Linear weights · decision trees','Convolution · attention · recurrence'],['Example: a small convolutional network','shares filters across image locations.']),
       (622,170,'02','What teaches?',['The source of the learning signal'],['Labeled examples · data-derived targets','Unlabeled structure · action rewards'],['Example: known digit labels provide','supervised feedback during training.']),
       (44,470,'03','What is the job?',['The prediction or decision required'],['Classification · regression · retrieval','Generation · clustering · control'],['Example: assign one of ten digit labels','to a handwritten image.']),
       (622,470,'04','How does it run?',['The resource and adaptation choices'],['Full training · adapters · distillation','Quantization · local or rented compute'],['Example: train with full-precision numbers','and run locally on a central processor.'])]
for x,y,n,t,sub,examples,specific in items:
 f.card(x,y,534,270,n,t,sub);f.text(x+24,y+131,examples,20);f.text(x+24,y+207,specific,18,MUTED)
f.rect(44,785,1112,83,PALE,LINE,14);f.text(67,818,'ONE MODEL CAN HAVE ALL FOUR LABELS',16,MUTED,700);f.text(67,850,'Convolutional network + supervised learning + digit classification + local execution',21,INK,600)
f.save('These are useful organizing questions, not an exhaustive or mutually exclusive taxonomy.')

f=Figure('split-leakage','Keep learning separate from measuring','Split first. Fit preprocessing on training examples only. Preserve the final test.',1020,'A dataset is split into training, validation, and test partitions. Learned preprocessing and model parameters fit the training partition. Validation informs choices; the final test is used after choices are frozen. A leakage example shows why fitting before the split leaks test information.')
f.box(400,169,400,65,['Available examples','Group or time boundaries respected'],size=18)
f.path('M600 234V265M600 265H216V295',arrow=True);f.path('M600 265V295',arrow=True);f.path('M600 265H984V295',arrow=True)
for x,n,t,lines in [(44,'01','TRAIN',['Fit scaling, vocabulary, imputation','and model parameters here.']), (428,'02','VALIDATION',['Choose features, settings, thresholds','and stopping rules here.']), (812,'03','FINAL TEST',['Estimate performance only after','the complete recipe is frozen.'])]:
 f.card(x,300,344,230,n,t);f.text(x+22,400,lines,17,MUTED);f.box(x+22,465,300,42,{'TRAIN':'Fit → learned state','VALIDATION':'Compare → choose','FINAL TEST':'Measure → report'}[t],size=18)
f.line(216,548,216,585,True);f.line(600,548,600,585,True);f.line(984,548,984,585,True)
f.box(44,590,344,90,['Learned scaler + model','from training data'],size=18)
f.box(428,590,344,90,['Apply the same learned scaler','without fitting it again'],size=18)
f.box(812,590,344,90,['Apply the frozen pipeline','without fitting it again'],size=18)
f.line(389,634,421,634,True);f.line(774,634,805,634,True)
f.rect(44,727,1112,201,WHITE,LINE);f.text(69,769,'A small leak can make a large claim look convincing.',24,INK,700)
f.text(69,811,['Leaky recipe: scale every row → split → evaluate. The test distribution influenced scaling.','Better recipe: split → fit the scaler on training rows → transform each partition.','For repeated customers or machines, keep each entity in one partition; for forecasts, split by time.'],18,MUTED)
f.save('Repeated validation can overfit choices too. Keep the final test separate from development decisions.')

f=Figure('model-mechanisms','Four ways to turn inputs into predictions','A numerical example makes each mechanism visible. Real training learns the useful choices.',1130,'Four panels illustrate weighted-sum prediction, branching tree rules, nearest-neighbor voting, and radial-basis kernel similarity. All numerical values are explicit examples, not trained production-model results.')
f.card(44,170,534,385,'01','Weighted sum','Linear regression combines feature contributions.')
f.box(69,284,141,61,['2 × 1.5','feature 1'],size=18);f.text(223,321,'+',25);f.box(251,284,141,61,['3 × 0.5','feature 2'],size=18);f.text(405,321,'+',25);f.box(432,284,120,61,['1','intercept'],size=18)
f.text(311,408,'3 + 1.5 + 1 = 5.5',30,INK,700,'middle');f.text(69,468,['Learning adjusts the weights to reduce error.','The interpretation depends on feature units.'],18,MUTED)
f.card(622,170,534,385,'02','Branching questions','Decision trees learn feature thresholds.')
f.box(771,275,235,49,'Latency > 200?',size=18);f.path('M800 324V359H712V387',arrow=True);f.path('M972 324V359H1042V387',arrow=True);f.text(728,350,'No',17);f.text(1009,350,'Yes',17);f.box(649,392,126,47,'Routine',size=18);f.box(916,392,212,47,'Errors > 3?',size=18)
f.text(648,486,['For latency = 220 and errors = 4,','both questions answer “yes” → escalate.'],18,MUTED)
f.card(44,582,534,440,'03','Nearest-neighbor vote','Nearby examples vote on a label.')
for i,(x,y,label) in enumerate([(105,721,'A'),(170,703,'A'),(226,742,'A'),(303,709,'A'),(370,744,'B'),(435,705,'B'),(491,751,'B')]):
 f.circle(x,y,23, GREEN if label=='A' else '#EAD8C6');f.text(x,y+7,label,20,INK,700,'middle')
f.text(69,821,'7 neighbors: 4 label A, 3 label B',24,INK,700);f.text(69,866,['Majority prediction: A','Vote fraction: 4 ÷ 7 ≈ 0.57','A vote fraction is not automatically','a calibrated probability.'],18,MUTED)
f.card(622,582,534,440,'04','Kernel similarity','A kernel measures similarity implicitly.')
f.text(648,713,['Radial basis similarity = exp(−γ × distance²)','Take γ = 0.5 and reference point (1, 0).'],18,INK)
f.box(649,779,480,58,'Point (1, 1): distance² = 1 → similarity 0.607',size=17)
f.box(649,850,480,58,'Point (3, 0): distance² = 4 → similarity 0.135',size=17)
f.text(649,947,['A classifier combines similarities to reference','points; the resulting boundary can be curved.'],18,MUTED)
f.save('The formulas illustrate mechanisms. A realistic model needs representative data and independent evaluation.')

f=Figure('neural-shapes','Follow the shapes through a network','Dimensions tell you what each layer consumes and produces. The batch dimension is omitted.',1325,'Three networks are shown with exact tensor shapes. A dense digit classifier maps 64 pixels to 32 hidden units to 10 scores. A convolutional digit classifier maps one by eight by eight through convolution and pooling to 256 flattened values and 10 scores. A graph network combines node features and connections with neighbor aggregation.')
f.card(44,170,1112,244,'01','Dense network','An eight-by-eight image becomes a vector with 64 numbers.')
for x,w,labels in [(76,260,['Input vector','64 values']),(465,260,['Hidden layer','32 activations']),(854,260,['Output layer','10 class scores'])]:f.box(x,282,w,82,labels,size=21)
f.line(341,323,454,323,True);f.line(730,323,844,323,True)
f.card(44,438,1112,419,'02','Convolutional network','Each shape is channels × height × width. Convolutions use a three-by-three filter with padding one.')
for x,w,labels in [(76,550,['Input','1 × 8 × 8']),(465,550,['Convolution','8 × 8 × 8']),(854,550,['Two-by-two pooling','8 × 4 × 4'])]:f.box(x,w,260,80,labels,size=21)
f.line(341,590,454,590,True);f.line(730,590,844,590,True);f.path('M984 632V668H206V698',arrow=True)
for x,y,labels in [(76,703,['Convolution','16 × 4 × 4']),(465,703,['Flatten','256 values']),(854,703,['Output layer','10 class scores'])]:f.box(x,y,260,80,labels,size=21)
f.line(341,743,454,743,True);f.line(730,743,844,743,True);f.text(76,826,'Flattening preserves the count: 16 × 4 × 4 = 256. The output scores are converted to probabilities if needed.',17,MUTED)
f.card(44,881,1112,326,'03','Graph network','Objects become nodes; known relationships become edges. Neighbor information is aggregated.')
f.box(76,994,260,111,['Node features','N nodes × F features','+ graph connections'],size=19)
f.box(465,994,260,111,['Aggregate neighbors','then transform','N nodes × H features'],size=19)
f.box(854,994,260,111,['Predict for each node','N nodes × C scores','C possible classes'],size=19)
f.line(341,1048,454,1048,True);f.line(730,1048,844,1048,True)
f.text(76,1163,'N = number of nodes; F = input features; H = hidden features; C = classes. These are symbols, not acronyms.',17,MUTED)
f.save('Architecture is the computational structure. The learning signal and evaluation task are separate choices.')

f=Figure('generative-comparison','Three ways to learn a generative structure','These objectives teach different behaviors. A low training loss does not guarantee useful generated samples.',1325,'An autoencoder reconstructs its input through a bottleneck. A variational autoencoder samples a latent vector from learned mean and log variance and balances reconstruction with distribution regularization. An adversarial model trains a generator against a discriminator that sees both generated and real samples.')
f.card(44,170,1112,255,'01','Autoencoder','Learning signal: reconstruct the input after passing through a bottleneck.')
for x,w,label in [(76,185,['Input image','x']),(333,185,['Encoder','compress']),(590,185,['Latent vector','z']),(847,265,['Decoder','reconstruction of x'])]:f.box(x,285,w,70,label,size=20)
for a,b in [(266,323),(523,580),(780,837)]:f.line(a,320,b,320,True)
f.text(76,394,'Compare reconstruction with the original image. A basic autoencoder does not define a reliable sampling prior.',17,MUTED)
f.card(44,449,1112,354,'02','Variational autoencoder','Learning signal: reconstruction plus a penalty that shapes the latent distribution.')
f.box(76,570,177,77,['Input image','x'],size=20);f.box(305,570,263,77,['Encoder predicts','mean and log variance'],size=19);f.box(620,570,185,77,['Sample latent','vector z'],size=19);f.box(857,570,255,77,['Decoder predicts','reconstruction'],size=19)
for a,b in [(258,295),(573,610),(810,847)]:f.line(a,608,b,608,True)
f.text(76,696,'Sampling rule: z = mean + exp(0.5 × log variance) × standard-normal noise',20,INK,600)
f.text(76,745,['The distribution penalty encourages a structured latent space; it is balanced against reconstruction.','The relative weighting affects both reconstructions and samples. Inspect both, not only the loss.'],18,MUTED)
f.card(44,827,1112,384,'03','Generative adversarial network','Learning signal: competition between a generator and a discriminator.')
f.box(76,948,185,68,'Random noise',size=20);f.box(334,948,235,68,'Generator',size=21);f.box(654,948,185,68,'Generated data',size=19);f.box(914,1019,198,74,['Discriminator','real or generated?'],size=17)
f.line(266,982,324,982,True);f.line(574,982,644,982,True);f.path('M844 982H1013V1009',arrow=True)
f.box(654,1092,185,68,'Real examples',size=19);f.path('M844 1126H1013V1103',arrow=True)
f.text(76,1082,['Update the discriminator to separate the two sources.','Update the generator to improve its generated examples.'],18,MUTED)
f.text(76,1189,'Watch for mode collapse: plausible samples may still lack diversity.',18,RUST,600)
f.save('This compares objectives, not mutually exclusive layer types. All three can use dense or convolutional layers.')

f=Figure('evaluation-guide','Match the measurement to the task','Always identify the dataset, split, baseline, sample count, and limitations alongside a score.',1130,'Classification example: 30 true positives, 10 false negatives, 5 false positives, 55 true negatives; accuracy 85 percent, precision about 85.7 percent, recall 75 percent. Regression errors of 2, minus 1, and 3 give mean absolute error 2 and root mean squared error about 2.16. Clustering silhouette measures separation, not semantic truth. Generation requires quality and diversity checks.')
f.card(44,170,534,415,'01','Classification','A confusion matrix shows the kinds of errors.')
f.text(240,290,'Predicted +',16,MUTED,600,'middle');f.text(422,290,'Predicted −',16,MUTED,600,'middle');f.text(69,345,'Actual +',17,MUTED);f.text(69,407,'Actual −',17,MUTED)
for x,y,n in [(174,307,30),(356,307,10),(174,370,5),(356,370,55)]:f.box(x,y,164,52,str(n),PALE if n in [30,55] else '#F5E7DC',25)
f.text(69,467,['Accuracy = (30 + 55) ÷ 100 = 85%','Precision = 30 ÷ (30 + 5) ≈ 85.7%','Recall = 30 ÷ (30 + 10) = 75%'],19,INK)
f.card(622,170,534,415,'02','Regression','Keep the meaning and units of the error.')
f.text(648,291,['Actual:       10, 20, 30','Predicted:   12, 19, 33','Errors:         2, −1, 3'],22,INK)
f.text(648,411,['Mean absolute error = (2 + 1 + 3) ÷ 3 = 2','Root mean squared error','= √((4 + 1 + 9) ÷ 3) ≈ 2.16'],18,INK,600)
f.text(648,523,['Larger errors receive more emphasis','when squared. Compare a simple baseline.'],18,MUTED)
f.card(44,611,534,413,'03','Clustering','Separation is not the same as meaning.')
f.text(69,735,['Silhouette compares within-cluster distance','with distance to another cluster.'],19,INK)
f.box(69,810,484,57,'Silhouette = (b − a) ÷ max(a, b)',size=21)
f.text(69,914,['a: average distance within the assigned cluster','b: smallest average distance to another cluster','A higher value does not prove useful categories.'],17,MUTED)
f.card(622,611,534,413,'04','Generation','Inspect quality, diversity, and held-out behavior.')
f.text(648,737,['A model can produce one good-looking example','and repeat it, missing most of the data.'],19,INK)
f.box(649,807,480,59,'Quality + coverage + independent checks',size=20)
f.text(648,911,['Check reconstructions or generated samples.','Compare against held-out data and a baseline.','Training loss alone is not a quality certificate.'],18,MUTED)
f.save('The numerical examples are illustrative. Report real results from your own run, including failures.')
print('Generated six SVG infographics in',ROOT)
