import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from model import GNN, create_data_object
import io


# Set page configuration for a modern look
st.set_page_config(page_title="GetPARPi Predictor", page_icon="🔬", layout="wide")

# Title and Description
st.title("🧬 GetPARPi Predictor")
st.write("This web application helps predict PARP inhibitor using deep learning")

# Sidebar with Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Menu", ["Home🏠", "Documentation📄", "Batch Prediction📚", "Single Prediction🧬"])

# Define models paths
models = {
    "PARP-1": "Models/model-5b.pth",
    "PARP-2": "Models/model-5b.pth",
    "PARP-5A": "Models/model-5b.pth",
    "PARP-5B": "Models/model-5b.pth",
}

# Home page
if page == "Home🏠":
    st.subheader('Introduction')   
    st.write('Poly ADP‑ribose polymerase (PARP) inhibitors represent a promising therapeutic strategy for cancer treatment, particularly in patients with BRCA mutations. Computerized virtual screening techniques have been used for PARP inhibitors, reducing the time and economic costs of drug discovery. This web server is a practical tool for predicting molecules with inhibitory activities on PARP-1, PARP-2, PARP-5A, and PARP-5B.')

# Documentation page
elif page == "Documentation📄":
    st.subheader('📄How to use')
    st.write("1. Select menu from the sidebar.")
    st.write("2. Choose the predictor to execute from the sidebar.")
    st.write("3. Upload a CSV file containing the list of molecules for virtual screening.")
    st.write("4. Click 'Prediction'.")
    st.write("5. Displaying Prediction Results. After the prediction is complete, the results will be displayed on the webpage.")
    st.write("More details can be found on [GitHub](https://github.com/zonwoo/GetPARPi).")

# Batch Prediction page
elif page == "Batch Prediction📚":
    st.subheader("📚Batch Prediction")
    with open('example_molecule.csv') as f:
        st.download_button('Download Example input file', f,'example_molecule.csv')
    tabs = st.tabs(["PARP-1", "PARP-2", "PARP-5A", "PARP-5B"])

    for i, (tab, model_key) in enumerate(zip(tabs, models.keys())):
        with tab:
            st.write(f"**{model_key}**")
            
            num_chemberta_features = 768  
            num_fingerprint_features = 2048
            num_combined_features = num_chemberta_features + num_fingerprint_features
            num_classes = 2

            model = GNN(num_features=num_combined_features, num_classes=num_classes, hidden_dim=128, dropout=0.3)
            model.load_state_dict(torch.load(models[model_key]))
            model.eval()

            uploaded_file = st.file_uploader(f"Upload your dataset for {model_key} (CSV format)", type="csv", key=f"uploader_{i}")

            if uploaded_file is not None:
                df_test = pd.read_csv(uploaded_file)

                if 'smiles' not in df_test.columns:
                    st.error("Input Error: Upload a csv file containing valid SMILES!")
                else:
                    st.write("Data Preview:")
                    st.dataframe(df_test.head())

                    if st.button(f"Predict"):
                        test_dataset = []
                        valid_molecule = True

                        for _, row in df_test.iterrows():
                            data_obj = create_data_object(row['smiles'])
                            if data_obj is not None:
                                test_dataset.append(data_obj)
                            else:
                                valid_molecule = False

                        if not valid_molecule:
                            st.error("One or more molecules in the dataset are invalid. Please check your input.")
                        else:
                            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                            def predict_model(model, loader):
                                results = []
                                with torch.no_grad():
                                    for data in loader:
                                        out = model(data.x, data.edge_index, data.batch)
                                        probs = F.softmax(out, dim=1)
                                        results.extend(probs.cpu().numpy())
                                return results

                            predictions = predict_model(model, test_loader)

                            result_df = pd.DataFrame(predictions, columns=["Confidence for Class 0", "PCScoreInh"])
                            result_df["PCScoreInh"] = result_df["PCScoreInh"] * 100
                            result_df["Inhibitor"] = result_df["PCScoreInh"].apply(
                                lambda x: "✅ Inhibitor" if x > 50 else "❌ Non-Inhibitor"
                            )
                            result_df = result_df[["PCScoreInh", "Inhibitor"]]
                            result_df.index += 1 
                            st.write("**Prediction Results:**")
                            st.dataframe(result_df)
                            
                            st.markdown("<p style='font-size:12px; font-style:italic;'>PCScoreInh : GetPARPi model's Prediction confidence for molecule to be an inhibitor (%).</p>",
                                        unsafe_allow_html=True,
                                        )
                            result_df["Inhibitor"] = result_df["PCScoreInh"].apply(
                                lambda x: "inhibitor" if x > 50 else "non-inhibitor"
                            )
                            
                            csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download",
                                data=csv,
                                file_name="results.csv",
                                mime="text/csv"
                            )

# Single Prediction page
elif page == "Single Prediction🧬":
    st.subheader("🧬Single Prediction")
    tabs = st.tabs(["PARP-1", "PARP-2", "PARP-5A", "PARP-5B"])    

    for i, (tab, model_key) in enumerate(zip(tabs, models.keys())):
        with tab:
            st.write(f"**{model_key}**")
            smiles_input = st.text_input(f"Enter a SMILES string for {model_key}:", "", key=f"smiles_input_{i}")

            if st.button(f"Predict for {model_key}"):
                if not smiles_input:
                    st.error("Please enter a valid SMILES string.")
                else:
                    # Load the selected model
                    num_chemberta_features = 768  
                    num_fingerprint_features = 2048
                    num_combined_features = num_chemberta_features + num_fingerprint_features
                    num_classes = 2

                    model = GNN(num_features=num_combined_features, num_classes=num_classes, hidden_dim=128, dropout=0.3)
                    model.load_state_dict(torch.load(models[model_key]))
                    model.eval()

                    # Convert SMILES to data object
                    data_obj = create_data_object(smiles_input)

                    if data_obj is None:
                        st.error("Invalid SMILES string. Please check your input.")
                    else:
                        with torch.no_grad():
                            data_obj = data_obj.to("cpu") 
                            out = model(data_obj.x.unsqueeze(0), data_obj.edge_index, torch.tensor([0]))
                            probs = F.softmax(out, dim=1).squeeze()

                        # Display results
                        pc_score_inh = probs[1].item() * 100
                        is_inhibitor = "✅ Inhibitor" if pc_score_inh > 50 else "❌ Non-Inhibitor"

                        st.write(f"**Prediction Results for {model_key}:**")
                        st.markdown(f"- **PCScoreInh:** {pc_score_inh:.2f}%")
                        st.markdown(f"- **Class:** {is_inhibitor}")

                        # Display molecular structure
                        mol = Chem.MolFromSmiles(smiles_input)
                        if mol:
                            drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
                            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                            drawer.FinishDrawing()
                            img = Image.open(io.BytesIO(drawer.GetDrawingText()))
                            st.image(img, caption="Molecular Structure", use_column_width=True)

                        st.markdown("<p style='font-size:12px; font-style:italic;'>PCScoreInh: GetPARPi model's Prediction confidence for molecule to be an inhibitor (%).</p>",
                                    unsafe_allow_html=True)