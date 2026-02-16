import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Spaceship Titanic AI Ultimate",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sci-Fi Look
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stButton>button {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Session State (History)
# ---------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Age', 'HomePlanet', 'Role', 'Probability', 'Prediction'])

# ---------------------------------------------------------
# 3. Load Models & Tools
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model.pkl')
        cluster_model = joblib.load('cluster_model.pkl')
        
        # Smart Loading for Scalers
        try: scaler_cluster = joblib.load('scaler_cluster.pkl')
        except: scaler_cluster = joblib.load('scaler.pkl')

        try: scaler_model = joblib.load('scaler_model.pkl')
        except: scaler_model = None 

        try: model_columns = joblib.load('model_columns.pkl')
        except: model_columns = getattr(model, 'feature_names_in_', None)

        return model, cluster_model, scaler_cluster, scaler_model, model_columns
    except Exception as e:
        return None, None, None, None, None

model, cluster_model, scaler_cluster, scaler_model, model_columns = load_artifacts()

if model is None:
    st.error("⚠️ Critical Error: Model files not found. Please ensure .pkl files are in the directory.")
    st.stop()

# ---------------------------------------------------------
# 4. Sidebar Control
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3211/3211364.png", width=120)
    st.title("Command Center")
    st.markdown("---")
    
    # Global Filters for Dashboard
    st.header("📊 Dashboard Filters")
    filter_planet = st.multiselect("Filter by Planet", ['Earth', 'Europa', 'Mars'], default=['Earth', 'Europa', 'Mars'])
    filter_vip = st.radio("Filter by VIP", ["All", "VIP Only", "Non-VIP"], index=0)
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the sidebar to filter the data in the Dashboard tab.")
    st.caption("Version 3.0 | Ultimate Edition")

# ---------------------------------------------------------
# 5. Main Logic
# ---------------------------------------------------------
st.title("🚀 Spaceship Titanic: AI Analytics System")

# Tabs
tab_pred, tab_viz, tab_data = st.tabs(["🔮 AI Prediction", "📈 Interactive Dashboard", "🗃️ Raw Data"])

# =========================================================
# TAB 1: PREDICTION ENGINE
# =========================================================
with tab_pred:
    st.markdown("### 🧬 Passenger Survival Analysis")
    
    with st.container():
        col_form, col_res = st.columns([2, 1])
        
        with col_form:
            with st.form("main_form"):
                st.subheader("Passenger Profile")
                c1, c2, c3 = st.columns(3)
                age = c1.number_input("Age", 0, 100, 24)
                home_planet = c2.selectbox("Home Planet", ['Earth', 'Europa', 'Mars'])
                destination = c3.selectbox("Destination", ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'])
                
                c4, c5, c6 = st.columns(3)
                vip = c4.selectbox("VIP Status", [False, True], format_func=lambda x: "Yes" if x else "No")
                cryo_sleep = c5.selectbox("CryoSleep", [False, True], format_func=lambda x: "Yes" if x else "No")
                group_size = c6.slider("Group Size", 1, 15, 1)

                st.subheader("Services & Spending")
                s1, s2, s3, s4, s5 = st.columns(5)
                room_service = s1.number_input("RoomSvc", 0, 10000, 0)
                food_court = s2.number_input("FoodCt", 0, 10000, 0)
                shopping_mall = s3.number_input("ShopMall", 0, 10000, 0)
                spa = s4.number_input("Spa", 0, 10000, 0)
                vr_deck = s5.number_input("VRDeck", 0, 10000, 0)

                st.subheader("Logistics")
                l1, l2 = st.columns(2)
                deck = l1.selectbox("Cabin Deck", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
                side = l2.radio("Cabin Side", ['P (Port)', 'S (Starboard)'], horizontal=True)

                submitted = st.form_submit_button("🚀 Analyze Survival Probability")

        # Result Section (Right Column)
        with col_res:
            if submitted:
                # --- PROCESSING ---
                total_spend = room_service + food_court + shopping_mall + spa + vr_deck
                log_total_spend = np.log1p(total_spend)
                
                raw_input = pd.DataFrame({
                    'HomePlanet': [home_planet], 'CryoSleep': [int(cryo_sleep)], 'Destination': [destination],
                    'Age': [age], 'VIP': [int(vip)],
                    'RoomService': [room_service], 'FoodCourt': [food_court], 'ShoppingMall': [shopping_mall],
                    'Spa': [spa], 'VRDeck': [vr_deck],
                    'TotalSpend': [total_spend], 'LogTotalSpend': [log_total_spend],
                    'GroupSize': [group_size], 'IsSolo': [1 if group_size == 1 else 0],
                    'Deck': [deck], 'Side': [side[0]]
                })

                # Clustering
                cluster_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'Age']
                if scaler_cluster:
                    cluster_input = scaler_cluster.transform(raw_input[cluster_cols])
                else:
                    cluster_input = raw_input[cluster_cols]
                cluster_pred = cluster_model.predict(cluster_input)[0]
                raw_input['Cluster'] = cluster_pred

                # Prediction
                df_encoded = pd.get_dummies(raw_input, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
                final_df = df_encoded.reindex(columns=model_columns, fill_value=0)
                
                if scaler_model:
                    final_df_scaled = scaler_model.transform(final_df)
                else:
                    final_df_scaled = final_df

                prediction = model.predict(final_df_scaled)[0]
                try: proba = model.predict_proba(final_df_scaled)[0][1]
                except: proba = 0.5

                # --- DISPLAY ---
                st.markdown("### 📊 Report")
                
                if prediction == 1:
                    st.success("## Transported")
                    st.metric("Survival Confidence", f"{proba*100:.1f}%", delta="Safe")
                else:
                    st.error("## Not Transported")
                    st.metric("Risk Factor", f"{(1-proba)*100:.1f}%", delta="-High Risk")
                
                # Persona Badge
                badges = {
                    0: "👴 Frugal / Elderly", 
                    1: "👶 Young / Minimalist", 
                    2: "💎 VIP / Luxury", 
                    3: "🛍️ Average Spender"
                }
                st.info(f"**Persona:** {badges.get(cluster_pred)}")
                
                # Radar Chart for Spending
                spend_data = pd.DataFrame({
                    'Category': ['RoomSvc', 'FoodCt', 'Mall', 'Spa', 'VR'],
                    'Amount': [room_service, food_court, shopping_mall, spa, vr_deck]
                })
                fig_radar = px.line_polar(spend_data, r='Amount', theta='Category', line_close=True, title="Spending Profile")
                fig_radar.update_traces(fill='toself')
                st.plotly_chart(fig_radar, use_container_width=True)

                # Add to History
                new_row = pd.DataFrame({
                    'Age': [age], 'HomePlanet': [home_planet], 
                    'Role': [badges.get(cluster_pred)], 
                    'Probability': [f"{proba:.2f}"], 
                    'Prediction': ["✅" if prediction==1 else "❌"]
                })
                st.session_state.history = pd.concat([new_row, st.session_state.history]).reset_index(drop=True)

    # History Table
    if not st.session_state.history.empty:
        st.markdown("---")
        st.subheader("🕒 Prediction History (Session)")
        st.dataframe(st.session_state.history, use_container_width=True)

# =========================================================
# TAB 2: INTERACTIVE DASHBOARD
# =========================================================
with tab_viz:
    st.header("📈 Mission Analytics Dashboard")
    
    # Load Data
    try:
        df_viz = pd.read_csv("spaceship_titanic_dataset.csv")
        # Preprocessing on the fly for visualization
        if 'Deck' not in df_viz.columns:
             df_viz[['Deck', 'Num', 'Side']] = df_viz['Cabin'].str.split('/', expand=True)
        
        # Apply Sidebar Filters
        df_filtered = df_viz[df_viz['HomePlanet'].isin(filter_planet)]
        if filter_vip == "VIP Only":
            df_filtered = df_filtered[df_filtered['VIP'] == True]
        elif filter_vip == "Non-VIP":
            df_filtered = df_filtered[df_filtered['VIP'] == False]
            
    except:
        st.warning("⚠️ Dataset not found. Please upload 'spaceship_titanic_dataset.csv'.")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df_viz = pd.read_csv(uploaded_file)
            if 'Deck' not in df_viz.columns:
                 df_viz[['Deck', 'Num', 'Side']] = df_viz['Cabin'].str.split('/', expand=True)
            df_filtered = df_viz
        else:
            df_filtered = None

    if df_filtered is not None:
        # Row 1: KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Passengers", len(df_filtered))
        k2.metric("Transported", len(df_filtered[df_filtered['Transported']==True]))
        k3.metric("Survival Rate", f"{df_filtered['Transported'].mean()*100:.1f}%")
        k4.metric("Avg Spend", f"${df_filtered[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1).mean():.0f}")

        st.markdown("---")

        # Row 2: Advanced Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🪐 Sunburst: Planet → Dest → Status")
            # Cleaning for chart
            sun_data = df_filtered.dropna(subset=['HomePlanet', 'Destination', 'Transported'])
            fig_sun = px.sunburst(sun_data, path=['HomePlanet', 'Destination', 'Transported'], 
                                  color='Transported', color_discrete_map={True:'#00CC96', False:'#EF553B'})
            st.plotly_chart(fig_sun, use_container_width=True)

        with c2:
            st.subheader("💰 Spending Distribution by Deck")
            # Create TotalSpend for Viz
            df_filtered['TotalSpend_Viz'] = df_filtered[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
            fig_box = px.box(df_filtered.dropna(subset=['Deck']), x='Deck', y='TotalSpend_Viz', color='Transported',
                             category_orders={"Deck": ["A", "B", "C", "D", "E", "F", "G", "T"]},
                             title="Who Spends More? (By Deck)")
            st.plotly_chart(fig_box, use_container_width=True)

        # Row 3: Age vs Status
        st.subheader("📊 Age Demographics & Survival")
        fig_hist = px.histogram(df_filtered, x="Age", color="Transported", barmode="overlay", 
                                title="Age Distribution (Survivors vs Lost)",
                                color_discrete_map={True:'#00CC96', False:'#EF553B'})
        st.plotly_chart(fig_hist, use_container_width=True)

        # Row 4: 3D Scatter (The Wow Factor)
        st.subheader("🌌 3D Analysis: Spend vs Age vs Spa")
        fig_3d = px.scatter_3d(df_filtered.fillna(0).head(1000), x='Age', y='TotalSpend_Viz', z='Spa',
                               color='Transported', size_max=10, opacity=0.7,
                               color_discrete_map={True:'#00CC96', False:'#EF553B'})
        st.plotly_chart(fig_3d, use_container_width=True)

# =========================================================
# TAB 3: DATA VIEWER
# =========================================================
with tab_data:
    st.header("🗃️ Raw Dataset Explorer")
    if df_filtered is not None:
        st.dataframe(df_filtered.head(100), use_container_width=True)
        
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_spaceship_data.csv',
            mime='text/csv',
        )
        
        
# python -m streamlit run app.py