import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, DataStructs
import plotly.express as px

def calculate_dataset_metrics(df, smiles_col='smiles', activity_col='activity'):
    """
    Calculates key metrics from the uploaded dataset to feed the recommender.
    """
    n_compounds = len(df)
    active_count = len(df[df[activity_col] == 1])
    class_balance = active_count / n_compounds if n_compounds > 0 else 0
    
    # Usamos tu función optimizada de Braun-Blanquet para el score
    # Esto asegura que la métrica sea la misma que usaste en tu investigación
    overlap_score = calculate_structural_overlap(df)

    return {
        "n_compounds": n_compounds,
        "class_balance": class_balance,
        "overlap_score": overlap_score,  # <--- AHORA COINCIDE CON EL RECOMENDADOR
        "is_balanced": 0.3 < class_balance < 0.7
    }

# --- FUNCIÓN DE APOYO (Lógica basada en tu repo) ---
def calculate_chemical_properties(df):
    """
    Calculates essential medicinal chemistry descriptors using RDKit.
    """
    mols = [Chem.MolFromSmiles(s) for s in df['smiles'] if s]
    
    # Descriptores de Lipinski y Fisicoquímicos
    df['MW'] = [Descriptors.MolWt(m) if m else None for m in mols]
    df['LogP'] = [Descriptors.MolLogP(m) if m else None for m in mols]
    df['HBD'] = [Descriptors.NumHDonors(m) if m else None for m in mols] # Puentes Hidrógeno Donores
    df['HBA'] = [Descriptors.NumHAcceptors(m) if m else None for m in mols] # Puentes Hidrógeno Aceptores
    df['TPSA'] = [Descriptors.TPSA(m) if m else None for m in mols] # Polar Surface Area
    df['Rotatable_Bonds'] = [Descriptors.NumRotatableBonds(m) if m else None for m in mols]
    
    # Descriptores de Complejidad
    # El QED (Quantitative Estimate of Drug-likeness) es muy útil para ver qué tan "droga" es la molécula
    df['QED'] = [Descriptors.qed(m) if m else None for m in mols]
    return df

def calculate_structural_overlap(df):
    """
    Calculates the Braun-Blanquet similarity between Active and Inactive populations.
    """
    # Separar activos e inactivos
    actives_df = df[df['activity'] == 1]
    inactives_df = df[df['activity'] == 0]
    
    # Submuestreo para no colapsar la memoria en datasets gigantes
    if len(df) > 1000:
        actives_df = actives_df.sample(min(200, len(actives_df)))
        inactives_df = inactives_df.sample(min(200, len(inactives_df)))

    actives = [Chem.MolFromSmiles(s) for s in actives_df['smiles']]
    inactives = [Chem.MolFromSmiles(s) for s in inactives_df['smiles']]
    
    actives = [m for m in actives if m]
    inactives = [m for m in inactives if m]

    if not actives or not inactives:
        return 0.0

    # Fingerprints ECFP4 (Morgan Radius 2)
    fps_a = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in actives]
    fps_i = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in inactives]
    
    # Braun-Blanquet: c / max(a, b)
    similarities = []
    for fa in fps_a:
        for fi in fps_i:
            intersection = (fa & fi).GetNumOnBits()
            a_bits = fa.GetNumOnBits()
            b_bits = fi.GetNumOnBits()
            # La fórmula Braun-Blanquet que sustenta tu repo
            bb_sim = intersection / max(a_bits, b_bits) if max(a_bits, b_bits) > 0 else 0
            similarities.append(bb_sim)
    
    return np.mean(similarities) if similarities else 0.0