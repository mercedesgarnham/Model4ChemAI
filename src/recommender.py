def recommend_model(data_info):
    """
    Suggests the best model based on dataset size, class overlap (Braun-Blanquet),
    and the benchmarking evidence from the Trypanosomatics Lab.
    """
    size = data_info['n_compounds']
    overlap = data_info['overlap_score'] # Valor numérico (0.0 a 1.0)
    
    rec = {"model": "", "descriptors": "", "reason": "", "expected_auc": ""}

    # Caso 1: Datasets muy pequeños (Ej: CHEMBL2850)
    if size < 800:
        rec["model"] = "XGBoost"
        rec["descriptors"] = "RDKit Physicochemical Descriptors"
        rec["reason"] = "Small datasets are prone to overfitting. XGBoost with global descriptors (22 properties) provides better regularization and robustness."
        rec["expected_auc"] = "0.72 - 0.74"

    # Caso 2: Datasets medianos con Solapamiento Extremo (Ej: CHEMBL4072)
    elif 800 <= size < 5000 and overlap > 0.25:
        rec["model"] = "Random Forest"
        rec["descriptors"] = "ECFP Fingerprints (1024-bit)"
        rec["reason"] = "High structural overlap detected. Random Forest with high-resolution ECFP fingerprints is required to distinguish subtle SAR signals."
        rec["expected_auc"] = "0.88 - 0.90"

    # Caso 3: Datasets medianos con baja complejidad (Métrica de Similitud domina)
    elif 800 <= size < 5000 and overlap <= 0.25:
        rec["model"] = "Similarity Baseline"
        rec["descriptors"] = "ASP Fingerprints + Braun-Blanquet"
        rec["reason"] = "Low overlap detected. The similarity-based baseline is highly efficient and likely to achieve top performance with minimal cost."
        rec["expected_auc"] = "0.84 - 0.94"

    # Caso 4: Datasets grandes - Búsqueda de Generalización (Scaffold Hopping)
    else:
        rec["model"] = "TrypanoDEEPscreen"
        rec["descriptors"] = "2D Lewis Structure Images"
        rec["reason"] = "Large datasets allow for deep feature extraction. TrypanoDEEPscreen (2D-CNN) is recommended to discover novel scaffolds beyond structural similarity."
        rec["expected_auc"] = "0.76 - 0.88"

    return rec