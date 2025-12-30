import streamlit as st
import pandas as pd
import plotly.express as px
import time
from src.featurization import calculate_dataset_metrics, calculate_chemical_properties, calculate_structural_overlap
from src.recommender import recommend_model
from src.predictor import run_prediction
from src.visualizations import plot_pca_space, plot_braun_blanquet_heatmap

# 1. Configuración de página y Estado
st.set_page_config(page_title="Model4ChemAI", page_icon="🔬", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = 'intro'

def go_to_app():
    st.session_state.page = 'app'

def go_to_readme(): st.session_state.page = 'readme'
def go_to_intro(): st.session_state.page = 'intro'

# Solo se muestra si NO estamos en la intro
if st.session_state.page != 'intro':
    with st.sidebar:
        try:
            st.image("img/logo.png", use_container_width=True)
        except:
            pass
        
        st.title("Model4ChemAI")
        st.markdown("---")
        
        # Botones de navegación de la sidebar
        if st.button("Home"): 
            go_to_intro()
            st.rerun()
        if st.session_state.page != 'readme':
            if st.button("Read Documentation"): 
                go_to_readme()
                st.rerun()
        if st.session_state.page != 'app':
            if st.button("Run Tool"): 
                go_to_app()
                st.rerun()
        
        st.divider()

# --- PÁGINA DE INTRODUCCIÓN ---
if st.session_state.page == 'intro':
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.title("Welcome to Model4ChemAI 🔬")
        # Asegúrate de que la ruta 'img/abstract.png' sea correcta en tu PC
        try:
            st.image("img/abstract.png", use_container_width=True)
        except:
            st.warning("Cover image not found at img/abstract.png")
            
        st.markdown("""
        ### Strategic Modeling for Bioactivity
        Model4ChemAI is a scientific platform designed to streamline the selection of machine learning models for single target bioactivity datasets. 
        
        **Key Features:**
        - **Automated Data Characterization:** Exploratory analysis of the dataset.
        - **Smart Selection:** Recommendation of best AI Model.
        """)
        st.divider()
        st.button("Start Analysis & Upload Data →", on_click=go_to_app, type="primary", use_container_width=True)
        st.button("Read Technical Documentation", on_click=go_to_readme, use_container_width=True)
        st.caption("Developed by Mercedes Didier Garnham | Funded by [RSTMH](https://www.rstmh.org/) and supported by [Trypanosomatics Lab](https://www.trypanosomatics.org/)")

# --- PÁGINA DE LA APLICACIÓN ---
elif st.session_state.page == 'app':
    st.title("Data Analysis Engine")
    
    # --- 1. CARGA DE DATOS ---
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Dataset must contain 'smiles' and 'activity' columns.")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Validar columnas necesarias
        cols = [c.lower() for c in df.columns]
        if 'smiles' not in cols or 'activity' not in cols:
            st.error("🚨 Error: Dataset must contain 'SMILES' and 'Activity' columns.")
        else:
            st.success("✅ Dataset loaded successfully!")
            
            # --- 2. EXPLORACIÓN INICIAL ---
            tabs = st.tabs(["Data Preview", "Distribution Analysis", "Chemical Properties", "Chemical Space Characterization"])
            
            with tabs[0]:
                st.subheader("Raw Data Sample")
                st.dataframe(df.head(10), use_container_width=True)
                st.write(f"**Total rows:** {df.shape[0]} | **Total columns:** {df.shape[1]}")

            with tabs[1]:
                st.subheader("Class Balance & Dataset Size")
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    # Gráfico de Torta para Balance de Clases
                    class_counts = df['activity'].value_counts()
                    fig_pie = px.pie(
                        names=['Inactives (0)', 'Actives (1)'], 
                        values=class_counts.values,
                        color=class_counts.index,
                        color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                        hole=0.4,
                        title="Active vs Inactive Ratio"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_exp2:
                    # Métricas clave para el usuario
                    actives = class_counts.get(1, 0)
                    inactives = class_counts.get(0, 0)
                    ratio = round(actives/inactives, 2) if inactives > 0 else "N/A"
                    
                    st.metric("Total Actives", actives)
                    st.metric("Total Inactives", inactives)
                    st.metric("A/I Ratio", ratio)
                    
                    if ratio != "N/A" and (ratio < 0.1 or ratio > 10):
                        st.warning("⚠️ High class imbalance detected. This will influence the model recommendation.")

            with tabs[2]:
                    st.subheader("Medicinal Chemistry Overview")
                    df = calculate_chemical_properties(df)
                    
                    # Gráfico comparativo de QED (Drug-likeness)
                    fig_qed = px.box(df, x="activity", y="QED", color="activity",
                                    title="Drug-likeness (QED) Score by Class",
                                    color_discrete_map={0: '#e74c3c', 1: '#2ecc71'})
                    st.plotly_chart(fig_qed, use_container_width=True)
                    
                    # Gráfico de TPSA vs LogP
                    fig_scatter = px.scatter(df, x="LogP", y="TPSA", color="activity", 
                                            size="MW", hover_name="smiles",
                                            title="Chemical Space: TPSA vs LogP (Size = MW)",
                                            color_discrete_map={0: '#e74c3c', 1: '#2ecc71'})
                    st.plotly_chart(fig_scatter, use_container_width=True)
            with tabs[3]:
                st.subheader("Chemical Space Characterization")
                # --- RENGLÓN 1: PCA y VARIANZA ---
                fig_pca, fig_var = plot_pca_space(df)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.plotly_chart(fig_pca, use_container_width=True) # Ahora sí recibe una sola figura

                with col2:
                    st.plotly_chart(fig_var, use_container_width=True) # Aquí la segunda
                st.divider()

                    # --- RENGLÓN 2: EL ÁRBOL (DENDROGRAMA) ---
                st.subheader("🌳 Hierarchical Structural Tree (Braun-Blanquet)")
                st.write("This tree clusters molecules by structural similarity. The labels show 'Index (Activity)'.")
                    
                fig_tree = plot_braun_blanquet_heatmap(df)
                st.plotly_chart(fig_tree, use_container_width=True)

            # --- BOTÓN DE PROCESAMIENTO CIENTÍFICO ---
            st.divider()
            st.header("2. Model Recommendation")
            st.write("Click below to run the characterization engine and find the best AI strategy for your data.")
            
            if st.button("🔍 Analyze & Recommend Model", type="primary", use_container_width=True):
                with st.status("Analyzing Chemical Space...", expanded=True) as status:
                    # 1. Calcular métricas (Llamada a tu función src.featurization)
                    st.write("Calculating Braun-Blanquet overlap...")
                    metrics = calculate_dataset_metrics(df) # Función que ya tienes
                    time.sleep(1)
                    
                    # 2. Generar Recomendación
                    st.write("Applying Benchmarking Heuristics...")
                    rec = recommend_model(metrics) # Función que ya tienes
                    time.sleep(1)
                    
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                # --- 4. RESULTADOS DE LA RECOMENDACIÓN ---
                st.success(f"### 🏆 Recommended Model: {rec['model']}")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"**Selected Descriptors:** `{rec['descriptors']}`")
                    st.markdown(f"**Expected AUC-ROC:** `{rec['expected_auc']}`")
                with res_col2:
                    st.markdown(f"**Structural Overlap:** `{metrics['overlap_score']:.2f}`")
                    st.markdown(f"**Reasoning:** {rec['reason']}")

                # --- 5. PASO FINAL: INFERENCIA ---
                st.divider()
                st.header("3. Implementation Guide")
                st.info(f"Based on our benchmarking, the most effective strategy for your data is **{rec['model']}**.")
                
                # Definición de repositorios según el modelo recomendado
                repo_map = {
                    "TrypanoDEEPscreen": "https://github.com/sebastianjinich/trypanodeepscreen_dev",
                    "Similarity Baseline": "https://github.com/sebastianjinich/Similarity-Based-Bioactivity-Predictor",
                    "XGBoost": "https://github.com/LuzSommariva/Cheminformatics-ML-Trypanosoma",
                    "Random Forest": "https://github.com/LuzSommariva/Cheminformatics-ML-Trypanosoma"
                }
                
                # 1. Obtenemos el repo del mapa
                selected_repo = repo_map.get(rec['model'])

                # 2. Validamos si existe antes de mostrar el botón
                if selected_repo:
                    st.write(f"To proceed with the training, please visit the official repository for **{rec['model']}**.")
                    st.link_button(f"🔗 Go to {rec['model']} Repository", selected_repo, type="primary", use_container_width=True)

                    # Instrucciones rápidas de implementación
                    with st.expander("🛠️ Quick Setup Instructions"):
                        st.markdown(f"""
                        1. **Clone the repository:** `git clone {selected_repo}.git`
                        2. **Install dependencies:** Follow the `requirements.txt` or `environment.yml` provided in the repo.
                        3. **Run Training:** Use your uploaded dataset as the input for the training scripts.
                        4. **Configuration:** Ensure you use the recommended featurization: **{rec['descriptors']}**.
                        """)

                    st.success("Analysis complete. You now have the scientific roadmap and the source code needed for your project.")
        
                else:
                    # Este es tu mensaje de error en caso de que algo falle en la lógica
                    st.error(f"🚨 Configuration Error: No repository found for the suggested model '{rec['model']}'.")
                    st.info("Please contact the Trypanosomatics Lab for technical support.")
                
    else:
        # Estado vacío (Empty state)
        st.info("Please upload a CSV file to begin the exploration.")
        
elif st.session_state.page == 'readme':
    if st.sidebar.button("← Back to Intro"):
        go_to_intro()
        st.rerun()

    st.title("Technical Documentation & Model Architectures")
    st.markdown("""
    **Model4ChemAI** is a comprehensive decision-support system. This section details the theoretical 
    and technical foundations of the models evaluated in our benchmarking, ranging from similarity-based 
    heuristics to deep convolutional architectures.
    """)

    # --- CARACTERIZACIÓN DE DATASETS ---
    st.header("Dataset Composition & Structural Characterization")
    st.write("""
    The predictive power of any model is intrinsically linked to the underlying chemical space. 
    Below is the structural breakdown of the benchmarks used to calibrate Model4ChemAI.
    """)

    # Preparación de los datos del dataset
    dataset_info = {
        "ChEMBL ID": ["CHEMBL262", "CHEMBL2581", "CHEMBL5567", "CHEMBL4657", "CHEMBL4072", "CHEMBL2850"],
        "Total": ["3,452", "2,254", "1,875", "1,604", "1,545", "764"],
        "Actives": [2793, 1199, 570, 548, 1007, 484],
        "Inactives": [659, 1055, 1305, 1056, 538, 280],
        "A/I Ratio": [4.24, 1.14, 0.44, 0.52, 1.87, 1.73],
        "Sim. Peaks (BB)": ["0.20 – 0.30", "A: 0.25 / I: 0.40", "0.18 – 0.25", "0.20 – 0.25", "0.15 – 0.25", "0.20 – 0.30"],
        "Structural Notes": [
            "Moderate diversity; no duplicates.",
            "Compact inactives; risk of memorization.",
            "Chemically homogeneous actives.",
            "Strong class overlap; no duplicates.",
            "High structural similarity between classes.",
            "Partial data leakage in actives (>0.6)."
        ]
    }

    df_composition = pd.DataFrame(dataset_info)
    
    # Mostramos la tabla de composición
    st.dataframe(df_composition, use_container_width=True, hide_index=True)

    

    # --- INSIGHTS ESTRUCTURALES ---
    st.markdown("#### 🔍 Structural Insights & Challenges")
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.write("""
        - **The Overlap Challenge:** Targets like **CHEMBL4072** and **CHEMBL4657** present high structural similarity between active and inactive groups. This explains why high-resolution ECFP4 fingerprints are prioritized over global descriptors in these cases.
        - **Memorization Risks:** In **CHEMBL2581**, the inactive set is highly compact (Sim: 0.40). Models may appear to perform well by simply 'memorizing' the inactive cluster rather than learning bioactivity patterns.
        """)

    with col_ins2:
        st.write("""
        - **Data Leakage Warnings:** In **CHEMBL2850**, we detected similarity peaks >0.6 in the active set, suggesting potential redundancy. The tool uses this to recommend more conservative, regularized models (XGBoost).
        - **Class Imbalance:** The **A/I Ratio** varies from 0.44 to 4.24. Model4ChemAI adjusts the recommendation to ensure the selected algorithm can handle these imbalances without bias.
        """)

    # --- 1. BASELINE MODEL ---
    st.header("1. Baseline Model (Similarity-Based)")
    st.write("""
    The Baseline serves as a reference threshold. It is not a traditional ML algorithm but an implementation 
    of the **Similarity Property Principle**, assuming that structurally similar molecules share biological activities.
    """)
    
    col_bl1, col_bl2 = st.columns([2, 1])
    with col_bl1:
        st.markdown("**Mechanism:**")
        st.write("""
        - **Scoring:** The model identifies the maximum similarity value (`max()`) of a query compound against separate sets of known actives and inactives.
        - **Probability Generation:** These raw scores are processed through a **Softmax function** to generate a final bioactivity probability.
        - **Featurization:** Uses **ASP (All-Shortest Paths)** fingerprints via jCompoundMapper. Molecules are represented as graphs (atoms as nodes, bonds as edges), encoding all shortest paths between atom pairs.
        - **Coefficient:** Paired with the **Braun-Blanquet coefficient** to quantify structural overlap.
        """)
    with col_bl2:
        st.info("💡 *Best for: Establishing a 'floor' of performance. If ML cannot beat this, the SAR signal is purely similarity-driven.*")

    

    st.divider()

    # --- 2. TRADITIONAL MACHINE LEARNING ---
    st.header("2. Random Forest (RF) and XGBoost Models")
    st.write("""
    Supervised learning algorithms optimized for binary classification. They are selected for their 
    ability to map non-linear relationships between molecular features and bioactivity.
    """)

    tabs_ml = st.tabs(["Algorithms", "Featurization Options"])
    
    with tabs_ml[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Random Forest (RF):**")
            st.write("A bagging ensemble of multiple decision trees trained randomly. It is highly robust across various dataset sizes and noise levels.")
        with c2:
            st.markdown("**XGBoost:**")
            st.write("A gradient boosting framework that iteratively optimizes performance. Particularly competitive in small datasets when paired with specific descriptors.")

    with tabs_ml[1]:
        st.markdown("Models are trained using six distinct featurization strategies:")
        st.write("""
        - **ECFP (Extended Connectivity):** 1024-bit circular fingerprints (radius 2) capturing atom neighborhoods.
        - **MACCS Keys:** 166-bit structural keys verifying specific functional groups or rings.
        - **RDKit Physicochemical:** 22 descriptors (MW, LogP/Crippen, TPSA, rotatable bonds, etc.).
        - **Chemprop (C/R):** 300 descriptors derived from pre-trained classification (C) or regression (R) models.
        - **RDKit Fingerprints:** 200 non-binary topological descriptors.
        """)

    
    st.divider()

    # --- 3. DEEP LEARNING ---
    st.header("3. TrypanoDEEPscreen (2D CNN)")
    st.write("""
    An optimized reimplementation of the DEEPscreen system, treating drug-target interaction as an 
    image classification and ranking task.
    """)

    col_dl1, col_dl2 = st.columns([2, 1])
    with col_dl1:
        st.markdown("**Architecture & Mechanism:**")
        st.write("""
        - **Visual Learning:** The input consists of **2D Lewis structure images** (200x200 pixels) generated directly from SMILES. The network learns spatial patterns without pre-calculated fingerprints.
        - **CNN Backbone:** Features 5 convolutional layers (2x2 kernels, max-pooling) followed by a Multi-Layer Perceptron (MLP).
        - **Ensemble Strategy:** To ensure robustness, it averages predictions from an **ensemble of 26 independent models**.
        - **Output:** Employs a Softmax output to enable precise molecular ranking.
        """)
    with col_dl2:
        st.warning("⚠️ *Note: Requires high computational cost but offers potential for identifying novel scaffolds that similarity-based models might miss.*")

    # --- 4. BENCHMARKING RESULTS TABLE ---
    st.header("Benchmarking Performance Matrix")
    st.write("""
    The following table summarizes the **AUC-ROC** performance across different architectures. 
    Our benchmarking compares traditional ML (Random Forest, XGBoost), a Deep Learning approach (TrypanoDEEPscreen), 
    and a custom similarity-based Baseline.
    """)

    # DataFrame con tus resultados precisos
    benchmark_data = {
            "ChEMBL ID": [
                "CHEMBL262", "CHEMBL2581", "CHEMBL5567", 
                "CHEMBL4657", "CHEMBL2850", "CHEMBL4072"
            ],
            "Target Name": [
                "GSK-3β", "Cathepsin D", "Luciferase", 
                "DPP VIII", "GSK-3α", "Cathepsin B"
            ],
            "Baseline (ASP+BB)": ["0.873", "0.933", "0.773", "0.933", "0.693", "0.943"],
            "Best XGBoost": ["0.751", "0.904", "0.886", "0.907", "0.748", "0.877"],
            "Best Random Forest": ["0.802", "0.935", "0.882", "0.902", "0.739", "0.899"],
            "DEEPscreen": ["0.683", "0.833", "0.793", "0.823", "0.663", "0.763"],
            "TrypanoDEEPscreen": ["0.713", "0.883", "0.783", "0.863", "0.693", "0.760"]
        }
    
    st.table(pd.DataFrame(benchmark_data))

    
    # --- SECCIÓN 5: DECISION FRAMEWORK ---
    st.header("Scientific Decision Framework")
    st.write("""
    The platform recommends models based on three fundamental pillars derived from our research findings:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### A. Structural Resolution & Overlap")
        st.write("""
        In targets with high structural overlap (e.g., **CHEMBL4072**, **CHEMBL4657**, **CHEMBL5567**), high-resolution 
        fingerprints like **ECFP** are essential. **Random Forest (RF)** consistently outperformed 
        XGBoost in these scenarios (up to 0.89-0.90 AUC) by capturing subtle local molecular 
        environments that distinguish active compounds from structural analogs.
        """)
        
    with col2:
        st.markdown("#### B. Model Complexity vs. Generalization")
        st.write("""
        **TrypanoDEEPscreen** and original CNN architectures showed a tendency toward **overfitting**, 
        reaching training metrics near 1.0 but failing to surpass the **0.84 Baseline** in test sets. 
        For most target-specific SAR tasks, the optimized Baseline or RF ensemble is recommended 
        due to superior generalization and significantly lower computational cost.
        """)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### C. Data Density & Memorization")
        st.write("""
        Small datasets (e.g., **CHEMBL2850**) result in moderate metrics (0.72-0.74). Conversely, 
        highly clustered data (e.g., **CHEMBL2581**) can yield very high metrics (0.93 AUC) but 
        require caution as they may reflect "memorization" of a compact chemical space 
        rather than true predictive power.
        """)

    with col4:
        st.markdown("#### D. Computational Efficiency")
        st.write("""
        Time-to-result is a critical factor. While the **Baseline** completes analysis in 
        **<1 minute**, Deep Learning architectures can take over **1,000 minutes** per target, 
        often without providing a proportional increase in AUC-ROC.
        """)

    # --- SECCIÓN 6: ALGORITHM SPECIFICATIONS ---   
    st.header("Methodology & Selection Guide")
    st.write("""
    Choosing the right model depends on the balance between **structural diversity**, **data volume**, 
    and **computational resources**. Use the following guide based on our benchmarking results:
    """)

    # Decision Matrix Table for quick reference
    selection_data = {
        "Scenario": ["Quick Benchmark / High Similarity", "Standard SAR Analysis", "Low Data / Small Molecules", "Massive Screening / Novel Scaffolds"],
        "Recommended Model": ["Similarity Baseline", "Random Forest", "XGBoost", "TrypanoDEEPscreen"],
        "Best Featurization": ["ASP + Braun-Blanquet", "ECFP4 / RDKit-2D", "PhysChem / MACCS", "Lewis Structure Images"],
        "Key Strength": ["Interpretability & Speed", "High Accuracy (AUC)", "Robustness to Noise", "Feature Discovery"]
    }
    st.dataframe(pd.DataFrame(selection_data), hide_index=True, use_container_width=True)

    

    # Detailed Methodology
    col_meth1, col_meth2 = st.columns(2)

    with col_meth1:
        st.markdown("### 1. Similarity-Based Baseline")
        st.markdown("**When to use:** As a first-pass reference or when bioactivity is strictly driven by structural analogs.")
        st.write("""
        - **Logic:** Uses the *Similarity Property Principle* via an optimized grid-search approach.
        - **Specs:** **ASP (All-Shortest Path)** fingerprints + **Braun-Blanquet** coefficient.
        - **Advantage:** Extremely fast (<1 min) and provides a "performance floor" that more complex models must beat.
        """)

        st.markdown("### 2. Machine Learning Ensembles")
        st.markdown("**When to use:** For high-performance predictive modeling on well-characterized targets.")
        st.write("""
        - **Random Forest:** Best for handling high-dimensional data like **ECFP4**. Recommended when there is significant structural overlap between actives and inactives.
        - **XGBoost:** Preferred for **small datasets** or when using physicochemical descriptors, as its gradient boosting architecture is highly efficient in low-data regimes.
        """)

    with col_meth2:
        st.markdown("### 3. Deep Learning (TrypanoDEEPscreen)")
        st.markdown("**When to use:** For large-scale datasets (>10k compounds) where you seek 'Scaffold Hopping' (finding active molecules structurally different from the training set).")
        st.write("""
        - **Logic:** 2D-CNN architecture that learns directly from Lewis structure images.
        - **Trade-off:** Higher computational cost and a tendency for strictness (higher penalty for false positives), which may result in a lower average AUC-ROC but better identification of novel chemical entities.
        """)
        
        st.info("💡 **Note:** TrypanoDEEPscreen employs an ensemble of 26 models to stabilize predictions, making it the most robust but slowest option.")

    st.divider()
    # --- FINAL RESEARCH CREDITS SECTION ---
    st.markdown("### Institutional & Research Credits")
    
    col_credits1, col_credits2 = st.columns([2, 1])

    with col_credits1:
        st.markdown(f"""
        This scientific platform is the result of research performed at the **[Trypanosomatics Lab](https://www.trypanosomatics.org/)** (IIB-UNSAM). 
        
        This website is an evolution of the prototype developed during the **10SAJIB Hackathon**.
        
        **Research Team:**
        * **Model generation:** Sebastián Jinich & Luz Sommariva.
        * **Plataform design and Implementation:** Mercedes Didier Garnham
        * **Supervision:** Emir Salas-Sarduy, and Fernán Agüero.
        """)
        
    with col_credits2:
        st.info("""
        **Funding & Support:**
        Supported by the **Royal Society of Tropical Medicine and Hygiene (RSTMH)**.
        """)

    # Pie de página con el link oficial
    st.markdown("---")
    st.caption("© 2025 Model4ChemAI | Trypanosomatics Lab | [Visit Lab Website](https://www.trypanosomatics.org/)")
    
    if st.button("🚀 Start Analysis Now"):
        st.session_state.page = 'app'
        st.rerun()
else:
    st.info("Waiting for dataset upload...")