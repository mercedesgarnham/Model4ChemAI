import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from plotly.subplots import make_subplots  # <--- ESTA ES LA LÍNEA QUE FALTA
import plotly.figure_factory as ff
import plotly.graph_objects as go

def plot_pca_space(df):
    features = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'Rotatable_Bonds', 'QED']
    data = df.dropna(subset=features)
    X = StandardScaler().fit_transform(data[features])
    
    pca = PCA() # Calculamos todos los componentes para el gráfico de varianza
    components = pca.fit_transform(X)
    
    # 1. Gráfico de Dispersión PCA (2D)
    pca_df = pd.DataFrame(data=components[:, :2], columns=['PC1', 'PC2'])
    pca_df['Activity'] = data['activity'].astype(str)
    
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Activity',
                         title="Chemical Space (PCA)",
                         color_discrete_map={'0': '#e74c3c', '1': '#2ecc71'},
                         template="plotly_white", opacity=0.7)

    # 2. Gráfico de Varianza Explicada (Scree Plot)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig_var = px.bar(
        x=range(1, len(exp_var_cumul) + 1),
        y=pca.explained_variance_ratio_,
        labels={"x": "Component", "y": "Explained Variance"},
        title="Explained Variance per Component",
        template="plotly_white"
    )
    fig_var.add_scatter(x=list(range(1, len(exp_var_cumul) + 1)), y=exp_var_cumul, name="Cumulative")
    
    return fig_pca, fig_var

def plot_braun_blanquet_heatmap(df, max_samples=200):
    # 1. Preparación de datos
    sample_df = df.sample(min(len(df), max_samples)).copy().reset_index()
    mols = [Chem.MolFromSmiles(s) for s in sample_df['smiles']]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    sample_df = sample_df.iloc[valid_indices]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=1024) for i in valid_indices]
    
    if len(fps) < 2: return None

    # 2. Cálculo de Matriz Braun-Blanquet
    n = len(fps)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = (fps[i] & fps[j]).GetNumOnBits()
            a_bits = fps[i].GetNumOnBits()
            b_bits = fps[j].GetNumOnBits()
            matrix[i, j] = intersection / max(a_bits, b_bits) if max(a_bits, b_bits) > 0 else 0

    dist_matrix = 1 - matrix
    np.fill_diagonal(dist_matrix, 0)
    
    # 3. Crear el Dendrograma (El Árbol)
    labels = [f"{idx} ({int(act)})" for idx, act in zip(sample_df.index, sample_df['activity'])]
    
    # El linkage es necesario para el orden y para el árbol
    L = linkage(squareform(dist_matrix, checks=False), method='average')
    
    fig_tree = ff.create_dendrogram(dist_matrix, orientation='left', labels=labels,
                                   linkagefun=lambda x: L)

    # 4. Crear el Heatmap reordenado
    order = leaves_list(L)
    matrix_reordered = matrix[order, :][:, order]
    
    # 5. Combinar en una figura con Subplots
    # El dendrograma ocupará el 20% y el heatmap el 80%
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, 
                        column_widths=[0.2, 0.8], horizontal_spacing=0.01)

    # Añadir las líneas del árbol al primer subplot
    for trace in fig_tree.data:
        fig.add_trace(trace, row=1, col=1)

    # Añadir el heatmap al segundo subplot
    # Usamos las etiquetas ordenadas para el eje Y
    ordered_labels = [labels[i] for i in order]
    
    fig.add_trace(
        go.Heatmap(
            z=matrix_reordered,
            x=ordered_labels,
            y=ordered_labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sim", x=1.05)
        ), row=1, col=2
    )

    # 6. Ajustes de Layout
    fig.update_layout(
        title="Clustered Structural Similarity (Tree + Heatmap)",
        width=1000, height=800,
        template="plotly_white",
        showlegend=False
    )
    
    # Limpiar ejes del árbol
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(tickangle=-90, row=1, col=2) # Rotar etiquetas del heatmap
    
    return fig