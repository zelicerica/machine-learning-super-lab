"""Draw exact, lesson-specific teaching illustrations. Standard library only.

Run ``python course/draw_foundations.py`` from any directory. The SVGs are
illustrative worked examples, never captured training results. Keep this source
and both identical course/public copies together when updating a figure.
"""
from pathlib import Path
from html import escape
import math

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, LINE = '#0b1020', '#131c31', '#344563'
INK, MUTED = '#e6edf7', '#b4c1d5'
GREEN, VIOLET, AMBER, CORAL = '#4ade80', '#b5a0ff', '#fbbf24', '#fb8fa1'
W, H = 1120, 700


class Figure:
    def __init__(self, ident, title, subtitle, description):
        self.ident = ident
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#8192ae"/></marker></defs>',
                      '<g font-family="DejaVu Sans, sans-serif">']
        self.rect(0, 0, W, H, BG, radius=18)
        self.text(40, 38, 'FOUNDATIONS / '+ident.replace('-', ' ').upper(), 13, GREEN, weight=700, spacing=2)
        self.text(40, 84, title, 30, INK, weight=700)
        self.text(40, 117, subtitle, 17, MUTED)
        self.line(40, 645, 1080, 645, LINE, width=1)
        self.text(40, 674, 'Teaching illustration • Use your own lab run for measured results.', 14, MUTED)

    def raw(self, s): self.parts.append(s)
    def text(self, x, y, s, size=20, color=INK, weight=400, anchor='start', spacing=0):
        self.raw(f'<text x="{x:g}" y="{y:g}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" letter-spacing="{spacing}">{escape(str(s))}</text>')
    def lines(self, x, y, strings, size=18, color=MUTED, gap=28, **kw):
        for i, s in enumerate(strings): self.text(x, y+i*gap, s, size, color, **kw)
    def rect(self, x, y, w, h, fill=PANEL, stroke=None, radius=12):
        self.raw(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{radius}" fill="{fill}"'+(f' stroke="{stroke}"' if stroke else '')+'/>')
    def line(self, x1, y1, x2, y2, color=LINE, width=2, arrow=False, dash=None):
        self.raw(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def path(self, d, stroke=LINE, width=2, fill='none', arrow=False, dash=None):
        self.raw(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def circle(self, x, y, r=8, fill=GREEN, stroke=None):
        self.raw(f'<circle cx="{x:g}" cy="{y:g}" r="{r:g}" fill="{fill}"'+(f' stroke="{stroke}"' if stroke else '')+'/>')
    def diamond(self, x, y, r=10, fill=AMBER):
        self.path(f'M{x-r} {y} L{x} {y-r} L{x+r} {y} L{x} {y+r} Z',fill,1,fill)
    def note(self, x, y, w, lines, color=GREEN):
        self.rect(x, y, w, 32+len(lines)*25)
        self.rect(x, y, 4, 32+len(lines)*25, color, radius=2)
        self.lines(x+20, y+29, lines, 17, INK, gap=25)
    def panel_title(self, x, y, title, color=GREEN): self.text(x, y, title, 20, color, 700)
    def finish(self): return '\n'.join(self.parts+['</g></svg>'])


def linear():
    f=Figure('linear','Build a prediction from feature contributions','The example predicts a duration from three measurements and an intercept.',
             'Intercept 5 plus 2 times 10, 3 times 2, and 8 times 1 gives 39. An observed duration of 43 has residual 4 and squared error 16.')
    rows=[('Intercept','5',5,GREEN),('Record count','2 × 10',20,VIOLET),('Transformations','3 × 2',6,AMBER),('Compression','8 × 1',8,CORAL)]
    f.panel_title(48,171,'Add the learned contributions')
    cumulative=0
    for i,(label,formula,value,color) in enumerate(rows):
        y=211+i*76
        f.text(48,y+22,label,18)
        f.text(275,y+22,formula,19,color,700,anchor='end')
        f.rect(315,y,480,35,BG,LINE,4)
        f.rect(315+cumulative*10,y,value*10,35,color,radius=4)
        f.text(820,y+24,str(cumulative+value),21,color,700)
        cumulative+=value
    f.lines(881,225,['Running','total'],17,MUTED)
    f.line(315,510,795,510)
    for n in (0,10,20,30,40):
        x=315+n*10;f.line(x,506,x,516);f.text(x,540,n,15,MUTED,anchor='middle')
    f.note(48,554,1024,['Prediction: 5 + 20 + 6 + 8 = 39     Observed: 43', 'Residual: 43 − 39 = 4     Squared error: 4² = 16'])
    return f


def logistic():
    f=Figure('logistic','From a score to a probability to a decision','The sigmoid curve maps every real-valued score into the interval (0, 1).',
             'Sigmoid score zero is probability 0.5. Score 2.20 gives approximately 0.90 and exceeds the illustrative review threshold 0.80.')
    x=lambda z:90+(z+5)*64;y=lambda p:530-p*320
    for p in (0,.5,.8,1):
        f.line(90,y(p),730,y(p),LINE,1,dash='4 5');f.text(74,y(p)+5,f'{p:.1f}',15,MUTED,anchor='end')
    f.line(90,530,750,530,arrow=True);f.line(90,530,90,185,arrow=True)
    pts=[(x(z),y(1/(1+math.exp(-z)))) for z in [i/20 for i in range(-100,101)]]
    f.path('M'+' L'.join(f'{a:.2f} {b:.2f}' for a,b in pts),GREEN,4)
    f.line(90,y(.8),730,y(.8),AMBER,2,dash='8 6')
    for z in (-4,-2,0,2,4):f.text(x(z),555,z,15,MUTED,anchor='middle')
    f.text(400,585,'Weighted score',17,MUTED,anchor='middle')
    f.text(92,169,'Estimated probability',17,MUTED)
    for z,p,col in [(0,.5,VIOLET),(2.2,1/(1+math.exp(-2.2)),GREEN)]:
        f.circle(x(z),y(p),8,col);f.line(x(z),y(p)+10,x(z),530,col,1,dash='4 5')
    f.text(x(0)+15,y(.5)+26,'0 → 0.50',18,VIOLET,700)
    f.text(x(2.2)-7,y(.9)-20,'2.20 → ≈0.90',18,GREEN,700,anchor='middle')
    f.rect(780,190,300,340)
    f.panel_title(804,226,'Review threshold: 0.80',AMBER)
    f.lines(804,270,['0.50 < 0.80','Do not flag this example.'],17)
    f.lines(804,353,['0.90 ≥ 0.80','Flag this example.'],17,GREEN)
    f.lines(804,446,['A probability estimate','still needs calibration','and held-out evaluation.'],16)
    f.text(804,580,'p = 1 / (1 + exp(−score))',17,INK)
    return f


def tree():
    f=Figure('tree','Split a mixed group into clearer branches','Follow the exact normal/fault counts from the lesson example.',
             'A root contains six normal and four fault examples. One split creates a left branch with five normal and zero faults, and a right branch with one normal and four faults. Counts sum to the original ten.')
    def node(x,y,w,label,n,k):
        f.rect(x,y,w,135,PANEL,LINE);f.text(x+w/2,y+31,label,18,INK,700,anchor='middle')
        for i in range(n+k):f.circle(x+30+i*26,y+65,8,GREEN if i<n else CORAL)
        f.text(x+22,y+107,f'{n} normal  /  {k} faults',19,MUTED)
    node(380,155,360,'Starting node: 10 examples',6,4)
    f.path('M560 290 V330 H285 V365',arrow=True);f.path('M560 330 H835 V365',arrow=True)
    f.text(315,337,'Split condition holds',15,MUTED,anchor='middle')
    f.text(811,337,'Condition does not hold',15,MUTED,anchor='middle')
    node(95,370,380,'Branch A: 5 examples',5,0)
    node(645,370,380,'Branch B: 5 examples',1,4)
    f.text(285,540,'All normal',23,GREEN,700,anchor='middle')
    f.text(835,540,'Mostly faults',23,CORAL,700,anchor='middle')
    f.note(95,555,930,['Training searches feature thresholds that reduce label impurity.','A clearer training split still needs validation on held-out examples.'])
    return f


def ensemble():
    f=Figure('ensemble','Combine learners in two different ways','A simple voting example contrasts with successive boosting corrections.',
             'Five illustrative tree votes are normal, fault, normal, normal, fault, so three of five vote normal. The bundled random forest averages tree class probabilities rather than these hard votes. Boosting instead adds a scaled correction to a current prediction function.')
    f.rect(40,155,500,470);f.rect(560,155,520,470)
    f.panel_title(64,190,'Hard-vote illustration')
    for i,v in enumerate(['N','F','N','N','F']):
        x=100+i*95;col=GREEN if v=='N' else CORAL
        f.path(f'M{x} 235 L{x-23} 273 M{x} 235 L{x+23} 273',LINE,2)
        f.circle(x,227,14,VIOLET);f.circle(x-23,278,8,col);f.circle(x+23,278,8,col)
        f.text(x,321,v,24,col,700,anchor='middle')
        f.line(x,339,x,369,arrow=True)
    f.line(100,376,480,376);f.line(290,376,290,411,arrow=True)
    f.text(290,454,'3 normal  >  2 fault',27,GREEN,700,anchor='middle')
    f.lines(64,514,['This is an illustrative hard vote.','The bundled forest averages class','probabilities before choosing a label.'],17)
    f.panel_title(586,190,'Boosting: improve the current model',VIOLET)
    for y,title,sub,color in [(222,'Current function','F(x)',MUTED),(338,'New correction','η × h(x)',VIOLET),(454,'Updated function','F(x) + η × h(x)',GREEN)]:
        f.rect(610,y,420,83,BG,LINE);f.text(632,y+31,title,17,color);f.text(820,y+63,sub,24,INK,700,anchor='middle')
    f.line(820,307,820,333,arrow=True);f.line(820,423,820,449,arrow=True)
    f.text(586,584,'η controls the correction’s contribution.',17,MUTED)
    return f


def neighbors():
    f=Figure('neighbors','Let nearby labeled cases vote','For k = 5, the lesson’s labels are fault, normal, fault, fault, normal.',
             'A new query is connected to five selected neighbors. Three fault labels beat two normal labels. The vote fraction 3/5 is not a guarantee of a calibrated fault probability.')
    f.rect(40,155,615,465)
    f.line(88,566,615,566,arrow=True);f.line(88,566,88,185,arrow=True)
    f.text(322,602,'Standardized feature 1',15,MUTED,anchor='middle')
    f.text(105,189,'Standardized feature 2',15,MUTED)
    q=(350,383)
    for x,y,label in [(279,308,'F'),(425,310,'N'),(260,421,'F'),(390,452,'F'),(454,404,'N')]:
        col=CORAL if label=='F' else GREEN;f.line(*q,x,y,LINE,2,dash='4 5');f.circle(x,y,22,col);f.text(x,y+7,label,19,BG,700,anchor='middle')
    for x,y in [(144,251),(570,211),(145,493),(575,502),(513,245)]:f.circle(x,y,8,LINE)
    f.diamond(*q,17,AMBER);f.text(350,364,'New case',17,AMBER,700,anchor='middle')
    f.panel_title(700,211,'Count the selected labels')
    for y,n,label,col in [(257,3,'Fault',CORAL),(335,2,'Normal',GREEN)]:
        f.text(700,y+26,label,20,INK);f.rect(804,y,n*65,38,col,radius=4);f.text(1025,y+27,n,22,col,700)
    f.text(700,436,'Majority prediction: fault',24,CORAL,700)
    f.lines(700,481,['Vote fraction: 3 / 5 = 0.60','This is not a calibration guarantee.'],16)
    f.lines(700,561,['Scale features using training data.','Otherwise units can dominate distance.'],15)
    return f


def kernel():
    f=Figure('kernel','Curved geometry can need a richer representation','Nested circles are a visual example; the runnable lab uses interleaving half-circles.',
             'Two concentric classes cannot be separated by a straight line in the original plane. Squared radius maps the inner ring near one and outer ring near four. This explicit radial illustration is not the exact radial-basis-kernel feature mapping.')
    f.rect(40,155,500,390);f.rect(580,155,500,390)
    f.panel_title(64,192,'Original two-dimensional inputs')
    cx,cy=290,367
    f.line(105,cy,480,cy,arrow=True);f.line(cx,522,cx,217,arrow=True)
    f.circle(cx,cy,70,'none',LINE);f.circle(cx,cy,140,'none',LINE)
    for r,col,phase in [(70,GREEN,0),(140,VIOLET,.15)]:
        for i in range(16):
            a=2*math.pi*i/16+phase;f.circle(cx+r*math.cos(a),cy+r*math.sin(a),7,col)
    f.text(107,511,'Inner ring: A',15,GREEN);f.text(343,511,'Outer ring: B',15,VIOLET)
    f.panel_title(604,192,'Illustrative radius-related coordinate')
    f.text(830,250,'r² = x₁² + x₂²',25,INK,700,anchor='middle')
    f.line(640,435,1030,435,arrow=True)
    for n in range(5):
        xx=650+n*82;f.line(xx,430,xx,441);f.text(xx,467,n,16,MUTED,anchor='middle')
    for n,col in [(1,GREEN),(4,VIOLET)]:
        for i in range(5):f.circle(650+n*82,399-i*24,8,col)
    f.line(855,284,855,435,AMBER,2,dash='7 5');f.text(856,493,'A threshold can separate them',16,AMBER,anchor='middle')
    f.note(40,555,1040,['A radial basis kernel supplies nonlinear similarities without this explicit radius calculation.','The sketch explains the geometry; it does not show the kernel’s exact feature map.'])
    return f


def bayes():
    f=Figure('bayes','Update a prior using evidence, then normalize','A binary “unusual reading” event makes the probability arithmetic concrete.',
             'Illustrative prior fault probability is 0.1 and normal probability is 0.9. Event likelihoods 0.8 under fault and 0.1 under normal yield joint scores 0.08 and 0.09. Normalizing gives fault probability 0.08/0.17, approximately 47.1 percent.')
    columns=[(50,'Class prior'),(390,'Event likelihood'),(750,'Unnormalized score')]
    for x,t in columns:f.panel_title(x,185,t)
    for yy,cl,prior,like,score,col in [(224,'Fault','0.10','0.80','0.08',CORAL),(362,'Normal','0.90','0.10','0.09',GREEN)]:
        f.rect(40,yy,260,105);f.text(60,yy+35,cl,20,col,700);f.text(260,yy+78,prior,30,INK,700,anchor='end')
        f.text(333,yy+65,'×',30,MUTED,anchor='middle')
        f.rect(370,yy,300,105);f.text(520,yy+65,like,32,INK,700,anchor='middle')
        f.line(683,yy+53,732,yy+53,arrow=True)
        f.rect(750,yy,330,105);f.text(916,yy+65,score,32,col,700,anchor='middle')
    f.text(40,514,'Normalize over both classes: 0.08 + 0.09 = 0.17',23,INK,700)
    f.rect(40,546,1040,35,LINE,radius=5);f.rect(40,546,1040*8/17,35,CORAL,radius=5)
    f.text(40,613,'Posterior fault: 0.08 / 0.17 ≈ 47.1%',23,CORAL,700)
    f.text(1075,613,'Normal: ≈52.9%',21,GREEN,700,anchor='end')
    return f


def gaussian():
    f=Figure('gaussian','Predict a mean and an uncertainty interval','At one input, assume a Gaussian predictive mean of 10 and standard deviation of 2.',
             'The plotted Gaussian density has mean 10 and standard deviation 2. Its interval from 10 minus 1.96 times 2 to 10 plus 1.96 times 2 is 6.08 to 13.92, conditional on the model assumptions.')
    px=lambda v:95+(v-4)*77;py=lambda v:470-245*math.exp(-.5*((v-10)/2)**2)
    xx=[4+i*.03 for i in range(401)]
    curve='M'+' L'.join(f'{px(v):.2f} {py(v):.2f}' for v in xx)
    vals=[6.08+i*(13.92-6.08)/200 for i in range(201)]
    area=f'M{px(6.08)} 470 L'+' L'.join(f'{px(v):.2f} {py(v):.2f}' for v in vals)+f' L{px(13.92)} 470 Z'
    f.path(area,'none',0,'#1c4035');f.path(curve,GREEN,4)
    f.line(85,470,1040,470,arrow=True)
    for v in (4,6.08,10,13.92,16):
        x=px(v);f.line(x,470,x,480);f.text(x,511,f'{v:g}',18,GREEN if v in (6.08,13.92) else MUTED,anchor='middle')
    f.line(px(10),223,px(10),470,VIOLET,2,dash='7 5');f.text(px(10),194,'Mean = 10',22,VIOLET,700,anchor='middle')
    f.line(px(6.08),py(6.08),px(6.08),470,GREEN,1,dash='4 5');f.line(px(13.92),py(13.92),px(13.92),470,GREEN,1,dash='4 5')
    f.text(100,176,'Predictive density (illustration)',17,MUTED)
    f.text(560,551,'10 ± 1.96 × 2  →  [6.08, 13.92]',27,INK,700,anchor='middle')
    f.note(95,575,930,['Approximate 95% model interval. Check actual held-out coverage; assumptions can fail.'])
    return f


def cluster():
    f=Figure('cluster','Move a cluster center to the mean of its points','Hold the assignments fixed for this one-dimensional center update.',
             'Assigned points 1, 2 and 3 have mean 2. An illustrative old center at 1.5 moves to 2. K-means alternates assignments and center updates; geometric clusters do not automatically establish real-world categories.')
    f.panel_title(55,180,'Three assigned points')
    f.line(110,280,1010,280,arrow=True)
    X=lambda v:110+v*250
    for n in (1,2,3):
        f.circle(X(n),280,15,GREEN);f.text(X(n),245,n,26,GREEN,700,anchor='middle')
    f.diamond(X(1.5),323,12,AMBER);f.text(X(1.5),363,'Old center: 1.5',18,AMBER,anchor='middle')
    f.line(X(1.5)+15,323,X(2)-15,323,AMBER,3,arrow=True)
    f.diamond(X(2),323,12,VIOLET);f.text(X(2)+30,329,'New center: 2',20,VIOLET,700)
    f.rect(55,400,1010,103);f.text(560,464,'(1 + 2 + 3) / 3 = 2',38,INK,700,anchor='middle')
    for x,txt in [(80,'Assign to nearest center'),(445,'Move center to mean'),(799,'Repeat')]:
        f.text(x,548,txt,18,GREEN,700)
    f.line(325,541,421,541,arrow=True);f.line(666,541,771,541,arrow=True)
    f.lines(55,601,['A compact geometric grouping may still miss the meaningful groups in the real problem.'],18,MUTED)
    return f


def projection():
    f=Figure('projection','Keep the direction with the most variation','When two measurements rise together, one diagonal coordinate can retain their main pattern.',
             'Points vary mainly along a diagonal principal direction. Dashed perpendicular segments show information discarded when projecting to that one direction. Principal component analysis fits directions using training data; retained variation does not guarantee predictive usefulness.')
    f.rect(40,155,580,465)
    f.line(90,555,577,555,arrow=True);f.line(90,555,90,188,arrow=True)
    f.text(345,598,'Measurement 1',16,MUTED,anchor='middle');f.text(110,189,'Measurement 2',16,MUTED)
    f.line(128,522,545,208,GREEN,3,arrow=True)
    for i,offset in enumerate([15,-26,33,-13,24,-34,17,8,-22,20]):
        px=150+i*38;py=506-i*28.5
        # A screen-space perpendicular to the retained direction (38, -28.5).
        x=px+offset*.6;y=py+offset*.8
        f.line(x,y,px,py,VIOLET,1,dash='4 4');f.circle(x,y,7,VIOLET);f.circle(px,py,3,GREEN)
    f.text(387,223,'Keep this direction',18,GREEN,700,anchor='middle')
    f.panel_title(670,208,'Project → compress → reconstruct')
    f.lines(670,255,['Each point keeps one coordinate','along the principal direction.'],18)
    f.rect(670,332,400,76);f.text(870,379,'2 coordinates → 1 coordinate',22,GREEN,700,anchor='middle')
    f.lines(670,461,['The perpendicular displacement','is lost during this compression.'],18,VIOLET)
    f.lines(670,547,['Fit the directions on training data.','High variance does not necessarily','mean useful predictive information.'],16)
    return f


def dense():
    f=Figure('dense','Turn pixels into learned hidden features','An 8 × 8 digit becomes 64 inputs, then 32 hidden values, then 10 class scores.',
             'A representative network drawing shows only a subset of units, labeled with full sizes 64, 32 and 10. A nonlinear activation follows the hidden weighted sums. Each hidden neuron receives 64 weights and one bias.')
    cols=[(220,64,'64 inputs',GREEN),(560,32,'32 hidden values',VIOLET),(900,10,'10 class scores',AMBER)]
    coords=[]
    for x,num,label,color in cols:
        ys=[235+i*57 for i in range(6 if num!=10 else 5)];coords.append([(x,y) for y in ys])
        f.text(x,174,label,23,color,700,anchor='middle')
    for a,b in zip(coords,coords[1:]):
        for x,y in a:
            for xx,yy in b:f.line(x+13,y,xx-13,yy,'#263650',1)
    for points,(_,num,label,color) in zip(coords,cols):
        for x,y in points:f.circle(x,y,15,PANEL,color)
    f.text(220,566,'Flatten 8 × 8 pixels',18,MUTED,anchor='middle')
    f.text(560,566,'Weighted sum + activation',18,VIOLET,anchor='middle')
    f.text(900,566,'Scores, not labels',18,MUTED,anchor='middle')
    f.note(70,578,980,['Only representative units are drawn. One hidden neuron has 64 weights + 1 bias.'])
    return f


def convolution():
    f=Figure('convolution','Track the image tensor through the network','Shape notation is channels × height × width; flattening preserves the element count.',
             'A single-channel 8 by 8 image becomes eight 8 by 8 feature maps, then eight 4 by 4 pooled maps, then sixteen 4 by 4 maps. Flattening gives 256 values, followed by ten class scores.')
    items=[(50,165,'1 / Input','1 × 8 × 8',1,8,GREEN),
           (410,165,'2 / Padded convolution','8 × 8 × 8',3,8,VIOLET),
           (770,165,'3 / Pooling','8 × 4 × 4',3,4,AMBER),
           (50,398,'4 / Second convolution','16 × 4 × 4',4,4,VIOLET),
           (410,398,'5 / Flatten','16 × 4 × 4 = 256',0,0,GREEN),
           (770,398,'6 / Classifier','10 class scores',0,0,AMBER)]
    for x,y,title,shape,layers,side,col in items:
        f.rect(x,y,300,195);f.text(x+16,y+28,title,17,col,700)
        if layers:
            s=75;xx=x+105;yy=y+55
            for layer in range(layers-1,-1,-1):
                f.rect(xx+layer*8,yy-layer*7,s,s,BG,LINE,3)
            for a in range(1,side):
                f.line(xx+a*s/side,yy,xx+a*s/side,yy+s,LINE,1);f.line(xx,yy+a*s/side,xx+s,yy+a*s/side,LINE,1)
        elif 'Flatten' in title:
            for i in range(14):f.rect(x+24+i*18,y+68,12,48,col,radius=2)
            f.text(x+150,y+140,'All values in one vector',15,MUTED,anchor='middle')
        else:
            for i in range(10):f.rect(x+36+i*23,y+126-(20+(i*17)%62),14,20+(i*17)%62,col,radius=2)
        f.text(x+150,y+174,shape,21,INK,700,anchor='middle')
    for y in (263,496):f.line(358,y,399,y,arrow=True);f.line(718,y,759,y,arrow=True)
    f.path('M1070 263 H1095 V379 H25 V496 H40',arrow=True)
    f.text(560,625,'A stack is symbolic: the shape label gives the exact channel count.',16,MUTED,anchor='middle')
    return f


def graph():
    f=Figure('graph','Combine a node with its neighbors','Use the lesson’s simple unweighted mean to make message aggregation visible.',
             'A service with feature vector 1,0 and neighbors 0,1 and 1,1 gives component sum 2,2 and an unweighted mean 2/3,2/3. The actual implementation uses degree-normalized weights and self-connections, so it need not use this unweighted mean.')
    nodes=[(170,228,'Neighbor A','[0, 1]',VIOLET),(170,452,'Neighbor B','[1, 1]',AMBER),(445,355,'Service','[1, 0]',GREEN)]
    f.line(222,238,393,335,arrow=True);f.line(222,442,393,375,arrow=True)
    f.path('M426 312 C388 230 505 226 475 312',GREEN,2,arrow=True)
    f.text(450,242,'Include self',15,GREEN,anchor='middle')
    for x,y,title,vec,col in nodes:
        f.circle(x,y,50,PANEL,col);f.text(x,y+8,vec,22,col,700,anchor='middle');f.text(x,y+82,title,18,MUTED,anchor='middle')
    f.line(507,355,596,355,arrow=True)
    f.rect(620,190,460,338)
    f.panel_title(645,231,'Coordinate-wise aggregation')
    f.lines(645,279,['[1, 0] + [0, 1] + [1, 1]','                 = [2, 2]'],25,INK,gap=47)
    f.line(645,361,1054,361,LINE,1)
    f.text(850,407,'[2, 2] / 3',30,GREEN,700,anchor='middle')
    f.text(850,466,'= [2/3, 2/3]',32,GREEN,700,anchor='middle')
    f.note(40,555,1040,['This unweighted mean is a teaching example. The bundled graph model uses','degree-normalized aggregation with self-connections, followed by learned transformations.'])
    return f


def autoencoder():
    f=Figure('autoencoder','Reconstruct through a small bottleneck','The denoising encoder sees corrupted pixels; the reconstruction target stays clean.',
             'An encoder maps 64 input pixels through 32 hidden values into an eight-number code. A decoder reconstructs 64 pixels. The clean original image supplies the reconstruction target; the digit category is not the target.')
    f.panel_title(40,178,'Encode the noisy input',GREEN);f.panel_title(732,178,'Decode a reconstruction',VIOLET)
    boxes=[(40,225,174,160,'64 pixels','Noisy input',GREEN),(256,245,174,120,'32 values','Encoder',GREEN),
           (472,270,174,70,'8 values','Code',AMBER),(688,245,174,120,'Decoder','Learned mapping',VIOLET),(904,225,174,160,'64 pixels','Reconstruction',VIOLET)]
    for x,y,w,h,main,sub,col in boxes:
        f.rect(x,y,w,h,PANEL,col);f.text(x+w/2,y+h/2+7,main,23,col,700,anchor='middle');f.text(x+w/2,y+h+31,sub,16,MUTED,anchor='middle')
    for x1,x2 in [(217,251),(433,467),(649,683),(865,899)]:f.line(x1,305,x2,305,arrow=True)
    f.rect(65,481,250,87);f.text(190,518,'Clean original',22,GREEN,700,anchor='middle');f.text(190,549,'64-pixel target',17,MUTED,anchor='middle')
    f.rect(688,481,390,87);f.text(883,518,'Reconstruction loss',22,INK,700,anchor='middle');f.text(883,549,'Compare reconstructed and clean pixels',16,MUTED,anchor='middle')
    f.line(325,525,677,525,arrow=True);f.line(991,421,991,471,arrow=True)
    f.lines(65,614,['The target is the original image, not the digit label. Both encoder and decoder learn.'],18,MUTED)
    return f


def variational():
    f=Figure('variational','Sample a latent value without losing the gradient','The encoder predicts distribution parameters; random noise supplies the sample variation.',
             'For mean 2, standard deviation 0.5 and sampled standard-normal noise minus 1, the latent sample is 2 plus 0.5 times minus 1, which is 1.5. Training combines a reconstruction term with a weighted KL divergence to the prior.')
    f.panel_title(40,170,'One latent coordinate')
    vals=[(40,'Encoder mean','μ = 2',GREEN),(400,'Encoder standard deviation','σ = 0.5',VIOLET),(760,'Sampled standard-normal noise','ε = −1',AMBER)]
    for x,label,value,col in vals:
        f.rect(x,196,320,120);f.text(x+160,232,label,16,MUTED,anchor='middle');f.text(x+160,284,value,32,col,700,anchor='middle')
    for x in (200,560,920):f.path(f'M{x} 320 V350 H560 V377',arrow=True)
    f.rect(250,384,620,104,BG,LINE);f.text(560,430,'z = μ + σ × ε',27,INK,700,anchor='middle');f.text(560,470,'2 + 0.5 × (−1) = 1.5',27,GREEN,700,anchor='middle')
    f.line(560,496,560,522,arrow=True)
    f.text(560,554,'Decode z to reconstruct the input',23,VIOLET,700,anchor='middle')
    f.text(560,598,'Training objective: reconstruction term + β × KL(q(z|x) ‖ p(z))',22,INK,anchor='middle')
    f.text(560,626,'KL: Kullback–Leibler divergence to the prior',14,MUTED,anchor='middle')
    return f


def adversarial():
    f=Figure('adversarial','A plausible sample can hide a missing mode','Compare the target’s two groups with a generator that produces only the right-hand group.',
             'Illustrative distributions show the target with half its mass near minus 2 and half near plus 2. A collapsed generator has all mass near plus 2. Curves are schematic densities, not measured outputs from the runnable lab.')
    for x,title,collapsed in [(40,'Target: two modes',False),(585,'Collapsed generator: one mode',True)]:
        f.rect(x,165,495,340);f.text(x+25,202,title,22,CORAL if collapsed else GREEN,700)
        px=lambda v:x+45+(v+4)*50;py=lambda v:440-140*v
        xx=[-4+i*.025 for i in range(321)]
        def density(v):return (0 if collapsed else .5*math.exp(-.5*((v+2)/.45)**2))+(1 if collapsed else .5)*math.exp(-.5*((v-2)/.45)**2)
        # One common vertical scale preserves the difference in mixture weights.
        p='M'+' L'.join(f'{px(v):.2f} {py(density(v)):.2f}' for v in xx)
        f.path(p,CORAL if collapsed else GREEN,4);f.line(x+40,440,x+455,440,arrow=True)
        for v in (-2,0,2):f.text(px(v),474,f'{v:+g}' if v else '0',18,MUTED,anchor='middle')
        if collapsed:f.text(x+145,330,'Missing group',17,AMBER,700,anchor='middle');f.line(x+145,346,x+145,425,AMBER,2,arrow=True)
        else:f.text(x+147,280,'≈ half',18,GREEN,anchor='middle');f.text(x+345,280,'≈ half',18,GREEN,anchor='middle')
    f.note(40,535,1040,['The generator learns through discriminator feedback, but a loss score alone is insufficient.','Inspect distribution coverage and failure cases. These curves are illustrative densities.'])
    return f


def ngram():
    f=Figure('ngram','Predict the next word from counted transitions','The short context is “server”; the training counts are three “starts” and one “stops”.',
             'After server, starts appears three times and stops once. Their unsmoothed probabilities are 3/4 and 1/4. Add-one smoothing gives each vocabulary option one additional count and uses denominator four plus the vocabulary size V.')
    f.rect(40,180,270,220);f.text(175,227,'Context',19,MUTED,anchor='middle');f.text(175,286,'“server”',35,GREEN,700,anchor='middle');f.text(175,353,'4 observed transitions',17,MUTED,anchor='middle')
    f.path('M319 290 H360 V238 H408',arrow=True);f.path('M360 290 V365 H408',arrow=True)
    for y,word,count,col in [(190,'starts',3,GREEN),(317,'stops',1,VIOLET)]:
        f.text(430,y+31,word,25,col,700);f.text(1055,y+31,f'{count}/4 = {count/4:.2f}',24,col,700,anchor='end')
        f.rect(430,y+54,625,30,LINE,radius=4);f.rect(430,y+54,625*count/4,30,col,radius=4)
    f.rect(40,456,1040,163);f.panel_title(64,491,'Add one count to every vocabulary option',AMBER)
    f.text(560,540,'P(word | server) = (count + 1) / (4 + V)',29,INK,700,anchor='middle')
    f.lines(64,583,['V is the full vocabulary size. An unseen option receives 1 / (4 + V), not zero.'],18,MUTED)
    return f


def reinforcement():
    f=Figure('reinforcement','Move an action value toward its target','The learning rate determines how much of the current error enters one update.',
             'Q-learning current action value 2 is updated toward target 6 with learning rate 0.2. Twenty percent of the gap of four is 0.8, so the new value is 2.8. At a terminal goal the target contains the immediate reward without a future value.')
    f.rect(40,165,1040,126)
    for x,title,value,col in [(80,'Current value','2',VIOLET),(432,'Learning rate','0.2',AMBER),(784,'Target','6',GREEN)]:
        f.text(x,203,title,18,MUTED);f.text(x,259,value,39,col,700)
    x=lambda v:100+v*145
    f.line(85,391,1032,391,arrow=True)
    for n in (0,2,2.8,4,6):f.line(x(n),385,x(n),402);f.text(x(n),436,f'{n:g}',19,MUTED,anchor='middle')
    f.circle(x(2),391,9,VIOLET);f.circle(x(6),391,9,GREEN);f.diamond(x(2.8),391,12,AMBER)
    f.path(f'M{x(2)} 369 V337 H{x(2.8)} V369',AMBER,3,arrow=True)
    f.text(600,346,'Move 20% of the gap: 0.2 × 4 = 0.8',18,AMBER,700)
    f.text(560,510,'2 + 0.2 × (6 − 2) = 2.8',35,INK,700,anchor='middle')
    f.note(40,551,1040,['Nonterminal target: reward + discounted next-state value.','At the terminal goal, use the immediate reward only; there is no future-state value.'])
    return f


DRAWINGS = [linear, logistic, tree, ensemble, neighbors, kernel, bayes, gaussian,
            cluster, projection, dense, convolution, graph, autoencoder,
            variational, adversarial, ngram, reinforcement]


def main():
    for draw in DRAWINGS:
        figure = draw()
        svg = figure.finish()
        for base in (ROOT/'course/images/expanded/foundations', ROOT/'public/images/expanded/foundations'):
            base.mkdir(parents=True, exist_ok=True)
            (base/(figure.ident+'.svg')).write_text(svg, encoding='utf-8')
    print(f'Wrote {len(DRAWINGS)} lesson-specific SVGs to both course and public assets.')


if __name__ == '__main__':
    main()
