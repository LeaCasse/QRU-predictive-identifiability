"""Vector research schematic linking the question, criterion and experiment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figure_style import BLUE, TEAL, ORANGE, GRAY, style, save


def main():
    style()
    fig, ax = plt.subplots(figsize=(6.6,3.25))
    fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.975)
    ax.set(xlim=(0,10),ylim=(0,5.1));ax.set_axis_off()
    ax.text(5,4.89,'What must be measured now to predict after encoding drift?',
            ha='center',va='center',fontsize=10.3,color=BLUE,fontweight='semibold')
    cards=[(.04,BLUE,'1  Describe the model',
            'Physically attainable circuit paths',r'$a=a(\theta)$'),
           (3.47,ORANGE,'2  Define the prediction',
            'Same present, different future?',r'$Oa=Ob\quad\; Fa\ne Fb$'),
           (6.90,TEAL,'3  Measure the target',
            'Apply a phase to the unknown target',r'$\phi\ \longrightarrow\ \mathrm{target\ measurement}$')]
    for x,col,title,subtitle,formula in cards:
        ax.add_patch(FancyBboxPatch((x,1.85),3.05,2.58,
                     boxstyle='round,pad=0.02,rounding_size=.13',
                     facecolor='#f7f9fa',edgecolor='#d8e0e5',lw=.8))
        ax.plot([x+.16,x+2.89],[4.05,4.05],color=col,lw=2.5)
        ax.text(x+1.525,4.23,title,ha='center',va='center',fontsize=8.1,
                color=col,fontweight='semibold')
        ax.text(x+1.525,3.78,subtitle,ha='center',fontsize=7.5)
        ax.text(x+1.525,3.32,formula,ha='center',fontsize=10 if x<6 else 8.1,color=col)
    # Separate paths can enter the same present frequency.
    for y,lab,ls in [(2.86,'Path A','-'),(2.31,'Path B','--')]:
        ax.text(.38,y,lab,va='center',fontsize=7.6)
        ax.annotate('',xy=(2.08,2.59),xytext=(1.12,y),
                    arrowprops={'arrowstyle':'->','color':BLUE,'lw':1.15,'linestyle':ls})
    ax.text(2.28,2.59,'One\nfrequency',ha='center',va='center',fontsize=7.3)
    ax.text(5.0,2.72,'Hidden differences matter',ha='center',fontsize=7.5)
    ax.text(5.0,2.30,'if future responses differ',ha='center',fontsize=7.5)
    ax.text(8.43,2.72,'Resolve relevant path differences',ha='center',fontsize=7.5)
    ax.text(8.43,2.30,'before the encoding changes',ha='center',fontsize=7.5)
    for x in [3.15,6.58]:
        ax.annotate('',xy=(x+.22,3.1),xytext=(x-.04,3.1),
                    arrowprops={'arrowstyle':'->','color':GRAY,'lw':1.2})
    ax.annotate('',xy=(8.43,1.44),xytext=(8.43,1.80),
                arrowprops={'arrowstyle':'->','color':TEAL,'lw':1.2})
    ax.add_patch(FancyBboxPatch((.04,.15),9.91,1.22,
                 boxstyle='round,pad=0.02,rounding_size=.13',
                 facecolor='#edf5f4',edgecolor='#c6dedd',lw=.8))
    ax.text(.31,1.09,'VALIDATE',fontsize=8,color=TEAL,fontweight='semibold')
    ax.text(1.69,1.09,'Equal shot budgets  |  Blind-phase control  |  Matched classical model',fontsize=7.8)
    ax.text(.31,.63,'Forecast gains require persistent hidden structure and a future query sensitive to it.',
            fontsize=8.2,color=BLUE)
    save(fig,'fig0_research_roadmap')


if __name__=='__main__':
    main()
