import matplotlib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
matplotlib.rcParams['ps.fonttype'] = 42
import pandas as pd


def step_3d(df,sort_by=None,colors=None,alpha=0.5):
    '''
    Takes a dataframe and iterates through the rows plotting 3d step functions

    example:
    colors=sns.color_palette("cool", len(df.dropna()))
    fig,ax=narplot.step_d3(df,sort_by=1,colors=colors,alpha=0.5)




    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    #drop nan
    if pd.isnull(df).values.any():
        print('dropping NaNs')
        df=df.dropna()


    def td_interpolate(row, loc):
        f1 = interp1d(row.index, row.values, kind='nearest')
        x = np.asarray(row.index)

        xnew = np.linspace(min(x), max(x), num=len(x)*50, endpoint=True)

        return xnew, np.ones(len(xnew)) * loc, f1(xnew)


    #optional sorting
    if sort_by is not None:
        df=df.sort_values(by=sort_by)

    # optional colors
    if colors is None:
        colors=['gray']*len(df)


    j = 0
    for idx, row in df.dropna().iterrows():
        ax.plot3D(*td_interpolate(row, j), color=colors[j], alpha=alpha)
        j += 1
    ax.grid(False)
    return fig,ax


def plus_save(fig,target):
    '''
    Just moving the defaults to tight bounding boxing and higher DPI
    '''
    fig.savefig(target,bbox_inches='tight',dpi=300)