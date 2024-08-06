import matplotlib.pyplot as plt
import numpy as np
import pathlib
import seaborn as sns
from matplotlib.lines import Line2D



def setupPlotting():
    """
    Initial setup for the plotting tool.
    """
    palette = sns.color_palette("colorblind")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
    
    # Global settings for plots
    plt.rcParams.update({
        'figure.figsize': [10, 6],
        'axes.grid': True,
        'grid.alpha': 0.8,
        'lines.linewidth': 2,
        'grid.linestyle': '--',
        'axes.facecolor': 'w',
        'legend.frameon': False,
        'legend.loc': 'upper left',
        'legend.fontsize': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14
    })

def savePlot(filename, dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=False, output_folder='Fig_Output'):
    """
    Save the current plot to a PNG file in the specified folder.
    
    Args:
        filename (str): The name of the file to save the plot.
        dpi (int): Dots per inch for the image resolution (default is 600).
        bbox_inches (str or 'tight'): Bounding box in inches for the figure (default is 'tight').
        pad_inches (float): Amount of padding around the figure (default is 0.1).
        transparent (bool): Whether to save the figure with a transparent background (default is False).
        output_folder (str): The folder where the image will be saved (default is 'Fig_Output').
    """
    output_path = pathlib.Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    
    plt.savefig(file_path, 
                format='png', 
                dpi=dpi, 
                bbox_inches=bbox_inches, 
                pad_inches=pad_inches, 
                transparent=transparent)



##############################################

def plotRaw(Data, grid):
    plt.figure()
    plt.plot(grid, Data, color = "#717171")
    plt.title(f'Plotting of {Data.shape[1]} normal distributions')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    legend_elements = [
        Line2D([0], [0], color="#717171", label='Density Function Data'),
    ]
    plt.legend(handles=legend_elements)
    
def plotPrototype(Data, theta, grid):

    palette = sns.color_palette("colorblind")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
    
    plt.figure()
    plt.plot(grid, Data, color = "#717171", label = "Density Function Data")
    plt.plot(grid, theta, linewidth = 3, label = "Prototype")
    plt.title(f'Plotting of {theta.shape[1]} prototypes')
    legend_elements = [
        Line2D([0], [0], color="#717171", label='Density Function Data'),
    ]
    plt.legend(handles=legend_elements)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')


def plotTarget(Data, labels, grid):
    palette = sns.color_palette("colorblind")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
    plt.figure()
    
    unique_labels = np.unique(labels)
    
    for group in unique_labels:
        group_ = Data.loc[:, labels == group]
        
        for col in group_.columns:
            pdf = group_[col]
            plt.plot(grid, pdf, color=palette[group])
    
    legend_elements = [Line2D([0], [0], color=palette[group], label=f'Cluster {group}') for group in unique_labels]
    plt.legend(handles=legend_elements)
    
    plt.title(f'Plotting of {len(unique_labels)} clusters of distributions')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()


def plotIDX(Data, thet, labels, grid):
    palette = sns.color_palette("colorblind")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
    plt.figure()
    
    unique_labels = np.unique(labels)
    
    for group in unique_labels:
        group_ = Data.loc[:, labels == group]
        
        for col in group_.columns:
            pdf = group_[col]
            plt.plot(grid, pdf, color=palette[group], alpha=0.3)
            plt.plot(grid, thet[group], linewidth=3, color=palette[group])
    
    plt.title(f'Plotting of {len(unique_labels)} clusters of distributions')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    
    legend_elements = [Line2D([0], [0], color=palette[group], label=f'Cluster {group}') for group in unique_labels]
    legend_elements.append(Line2D([0], [0], color=palette[0], label='Prototypes'))
    plt.legend(handles=legend_elements)
    
    plt.show()



    
def plotPartition(U):
    plt.figure()
    sns.heatmap(U, annot=False, cbar=True)

    plt.xlabel('Density Function Elements')
    plt.ylabel('Clusters')
    plt.title('Heatmap of Fuzzy Partition')

    plt.yticks(ticks=np.arange(U.shape[0]) + 0.5, labels=[f'Cluster {i+1}' for i in range(U.shape[0])], rotation=90)


def plotTrainTest(train_df, test_df, grid):
    palette = sns.color_palette("colorblind")
    
    plt.figure()
    plt.plot(grid, train_df, color="#717171", alpha=0.7)
    plt.plot(grid, test_df, color = palette[0], label='Testing Data')
    plt.title(f'Plotting density distributions')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    legend_elements = [
        Line2D([0], [0], color="#717171", label='Training Data'),
        Line2D([0], [0], color=palette[0], label='Testing Data')
    ]
    plt.legend(handles=legend_elements)
