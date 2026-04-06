import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# Configuration générale
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motors.tn — Prévision des Ventes",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS personnalisé (UI améliorée)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0b1020 0%, #060b16 100%);
        color: #e8eefc;
    }

    [data-testid="stSidebar"] {
        background: #151c33;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #e8eefc;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .sub-title {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #c7d2fe;
        border-left: 4px solid #4f7cff;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        min-height: 120px;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }

    .result-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid rgba(79,124,255,0.35);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
        min-height: 260px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .result-title {
        color: #a5b4fc;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-align: center;
    }

    .result-value {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 12px;
        text-align: center;
    }

    .result-meta {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.9;
        text-align: center;
    }

    .empty-result {
        border: 2px dashed rgba(255,255,255,0.35);
        border-radius: 20px;
        padding: 36px 20px;
        min-height: 260px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        text-align: center;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4f7cff, #7c4dff);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.2rem;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(79,124,255,0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        transition: 0.2s ease;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_raw = pd.read_csv("Répartition_par_ville_-_Motors_tn (7).csv")
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], format="mixed", dayfirst=False, errors="coerce")
    df_raw["Année"] = df_raw["Date"].dt.year
    df_raw["Mois"] = df_raw["Date"].dt.month
    return df_raw

@st.cache_data
def load_predictions():
    par_marque = pd.read_csv("prevision_2026_par_marque.csv")
    par_ville = pd.read_csv("prevision_2026_par_ville.csv")
    par_marque_ville = pd.read_csv("prevision_2026_par_marque_ville.csv")
    comparaison = pd.read_csv("comparaison_modeles.csv")
    return par_marque, par_ville, par_marque_ville, comparaison

@st.cache_resource
def load_model():
    model = joblib.load("model_final.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

df_raw = load_data()
prev_marque, prev_ville, prev_marque_ville, comparaison = load_predictions()
model, encoders = load_model()

VILLES = sorted(df_raw["Ville"].dropna().unique())
MARQUES = sorted(df_raw["Marque"].dropna().unique())

# Palette cohérente
COLORS = px.colors.qualitative.Set2

# ─────────────────────────────────────────────────────────────────────────────
# Helpers UI
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def result_card(prediction_arrondie, ville, marque, annee, puissance_moy, ptac_moy):
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">Résultat de la prédiction</div>
        <div class="result-value">{prediction_arrondie} unités</div>
        <div class="result-meta">
            Ville: {ville}<br>
            Marque: {marque}<br>
            Année: {annee}<br>
            Puissance moyenne: {puissance_moy:.2f}<br>
            PTAC moyen: {ptac_moy:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

def empty_result_card():
    st.markdown("""
    <div class="empty-result">
        <div style="font-size: 3rem; margin-bottom: 8px;">🎯</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #cbd5e1; margin-bottom: 8px;">
            Résultat
        </div>
        <div style="font-size: 1rem;">
            Sélectionnez une ville et une marque,<br>
            puis cliquez sur <b>Lancer la prédiction</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚜 Motors.tn")
    st.markdown("Plateforme d'analyse et de prévision des ventes d'engins")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Exploration des données",
            "🔮 Prévisions 2026",
            "🎯 Simulateur de prédiction",
            "📈 Power BI"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"**Dataset :** {len(df_raw)} enregistrements")
    st.markdown("**Période :** 2021 → 2026")
    st.markdown(f"**Villes :** {df_raw['Ville'].nunique()}")
    st.markdown(f"**Marques :** {df_raw['Marque'].nunique()}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Exploration des données
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Exploration des données":

    st.markdown('<div class="main-title">📊 Exploration des données</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyse des immatriculations d\'engins en Tunisie (2021–2026)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total engins", f"{len(df_raw):,}")
    with col2:
        metric_card("Villes couvertes", df_raw["Ville"].nunique())
    with col3:
        metric_card("Marques", df_raw["Marque"].nunique())
    with col4:
        metric_card("Années", df_raw["Année"].nunique())

    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Filtres</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filtre_villes = st.multiselect("Villes", VILLES, placeholder="Toutes les villes")
    with col_f2:
        filtre_marques = st.multiselect("Marques", MARQUES, placeholder="Toutes les marques")
    with col_f3:
        filtre_annees = st.multiselect("Années", sorted(df_raw["Année"].dropna().unique()), placeholder="Toutes les années")

    df_filtered = df_raw.copy()
    if filtre_villes:
        df_filtered = df_filtered[df_filtered["Ville"].isin(filtre_villes)]
    if filtre_marques:
        df_filtered = df_filtered[df_filtered["Marque"].isin(filtre_marques)]
    if filtre_annees:
        df_filtered = df_filtered[df_filtered["Année"].isin(filtre_annees)]

    st.info(f"**{len(df_filtered)}** enregistrements correspondent aux filtres sélectionnés.")

    st.markdown('<div class="section-header">📋 Données filtrées</div>', unsafe_allow_html=True)
    cols_affichage = ["Ville", "Marque", "Date", "PAYS", "PUISSANCE", "PTAC", "Année", "Mois"]
    st.dataframe(
        df_filtered[cols_affichage].reset_index(drop=True),
        width="stretch",
        height=320
    )

    st.markdown("---")
    st.markdown('<div class="section-header">📈 Visualisations</div>', unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        ventes_marque = df_filtered.groupby("Marque").size().reset_index(name="Ventes").sort_values("Ventes", ascending=True).copy()
        ventes_marque["Ventes"] = pd.to_numeric(ventes_marque["Ventes"], errors="coerce")
        ventes_marque = ventes_marque.dropna(subset=["Ventes"])

        if not ventes_marque.empty:
            fig1 = px.bar(
                ventes_marque,
                x="Ventes",
                y="Marque",
                orientation="h",
                title="Ventes par marque",
                template="plotly_white"
            )
            fig1.update_layout(
                showlegend=False,
                title_font_size=15,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig1, width="stretch")
        else:
            st.warning("Aucune donnée disponible pour le graphique des marques.")

    with col_g2:
        ventes_ville = df_filtered.groupby("Ville").size().reset_index(name="Ventes").sort_values("Ventes", ascending=True).copy()
        ventes_ville["Ventes"] = pd.to_numeric(ventes_ville["Ventes"], errors="coerce")
        ventes_ville = ventes_ville.dropna(subset=["Ventes"])

        if not ventes_ville.empty:
            fig2 = px.bar(
                ventes_ville,
                x="Ventes",
                y="Ville",
                orientation="h",
                title="Ventes par ville",
                template="plotly_white"
            )
            fig2.update_layout(
                showlegend=False,
                title_font_size=15,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.warning("Aucune donnée disponible pour le graphique des villes.")

    st.markdown('<div class="section-header">📅 Évolution annuelle des ventes</div>', unsafe_allow_html=True)
    evolution = df_filtered.groupby(["Année", "Marque"]).size().reset_index(name="Ventes")
    top_marques = evolution.groupby("Marque")["Ventes"].sum().nlargest(6).index
    evolution_top = evolution[evolution["Marque"].isin(top_marques)]

    if not evolution_top.empty:
        fig3 = px.line(
            evolution_top,
            x="Année",
            y="Ventes",
            color="Marque",
            markers=True,
            title="Évolution des ventes par année (Top 6 marques)",
            template="plotly_white",
            color_discrete_sequence=COLORS
        )
        fig3.update_layout(
            title_font_size=15,
            legend_title="Marque",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(tickmode="linear", dtick=1)
        )
        st.plotly_chart(fig3, width="stretch")
    else:
        st.warning("Aucune donnée disponible pour l'évolution annuelle.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prévisions 2026
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prévisions 2026":

    st.markdown('<div class="main-title">🔮 Prévisions 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Résultats du modèle Random Forest — prévisions annuelles par ville et marque</div>', unsafe_allow_html=True)

    total_prevu = prev_marque_ville["Prévision_totale"].sum()
    top_marque = prev_marque.iloc[0]["Marque"]
    top_ville = prev_ville.iloc[0]["Ville"]
    nb_combos = len(prev_marque_ville)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total engins prévus", f"{total_prevu:.0f}")
    with col2:
        metric_card("Top marque", top_marque)
    with col3:
        metric_card("Top ville", top_ville)
    with col4:
        metric_card("Combinaisons prévues", nb_combos)

    st.markdown("---")

    with st.expander("📐 Performance du modèle Random Forest", expanded=False):
        rf_row = comparaison[comparaison["Modèle"] == "Random Forest"].iloc[0]
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            metric_card("MAE", f"{rf_row['MAE']:.4f}")
        with cm2:
            metric_card("RMSE", f"{rf_row['RMSE']:.4f}")
        with cm3:
            metric_card("R²", f"{rf_row['R²']:.4f}")

        st.markdown("**Comparaison des 3 modèles :**")
        st.dataframe(comparaison, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Top marques et villes prévues</div>', unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        top10_marques = prev_marque.head(10).sort_values("Prévision_totale", ascending=True).copy()
        top10_marques["Prévision_totale"] = pd.to_numeric(top10_marques["Prévision_totale"], errors="coerce")
        top10_marques = top10_marques.dropna(subset=["Prévision_totale"])

        if not top10_marques.empty:
            fig4 = px.bar(
                top10_marques,
                x="Prévision_totale",
                y="Marque",
                orientation="h",
                title="Top 10 marques — Prévisions 2026",
                template="plotly_white"
            )
            fig4.update_layout(
                showlegend=False,
                title_font_size=15,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig4, width="stretch")
        else:
            st.warning("Aucune donnée disponible pour le graphique des marques.")

    with col_g2:
        top10_villes = prev_ville.head(10).sort_values("Prévision_totale", ascending=True).copy()
        top10_villes["Prévision_totale"] = pd.to_numeric(top10_villes["Prévision_totale"], errors="coerce")
        top10_villes = top10_villes.dropna(subset=["Prévision_totale"])

        if not top10_villes.empty:
            fig5 = px.bar(
                top10_villes,
                x="Prévision_totale",
                y="Ville",
                orientation="h",
                title="Top 10 villes — Prévisions 2026",
                template="plotly_white"
            )
            fig5.update_layout(
                showlegend=False,
                title_font_size=15,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig5, width="stretch")
        else:
            st.warning("Aucune donnée disponible pour le graphique des villes.")

    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Tableau des prévisions — Ville × Marque</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filtre_ville_prev = st.multiselect("Filtrer par ville", sorted(prev_marque_ville["Ville"].unique()), placeholder="Toutes les villes")
    with col_f2:
        filtre_marque_prev = st.multiselect("Filtrer par marque", sorted(prev_marque_ville["Marque"].unique()), placeholder="Toutes les marques")

    df_prev_filtered = prev_marque_ville.copy()
    if filtre_ville_prev:
        df_prev_filtered = df_prev_filtered[df_prev_filtered["Ville"].isin(filtre_ville_prev)]
    if filtre_marque_prev:
        df_prev_filtered = df_prev_filtered[df_prev_filtered["Marque"].isin(filtre_marque_prev)]

    df_affichage = df_prev_filtered.rename(columns={
        "Prévision_totale": "Prévision totale",
        "Prévision_moyenne": "Prévision moyenne",
        "Nombre_combinaisons": "Nb combinaisons"
    })

    st.dataframe(df_affichage.reset_index(drop=True), width="stretch", height=420)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Simulateur de prédiction
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Simulateur de prédiction":

    st.markdown('<div class="main-title">🎯 Simulateur de prédiction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Estimez les ventes d\'une marque dans une ville pour une année donnée</div>', unsafe_allow_html=True)

    st.markdown("---")
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">⚙️ Paramètres de simulation</div>', unsafe_allow_html=True)

        ville_sim = st.selectbox("🏙️ Ville", VILLES)
        marque_sim = st.selectbox("🏭 Marque", MARQUES)
        annee_sim = st.slider("📅 Année de prévision", min_value=2026, max_value=2030, value=2026)

        mask = (df_raw["Ville"] == ville_sim) & (df_raw["Marque"] == marque_sim)
        df_combo = df_raw[mask]

        if len(df_combo) > 0:
            puissance_moy = float(df_combo["PUISSANCE"].mean())
            ptac_moy = float(df_combo["PTAC"].mean())
            historique_count = len(df_combo)
        else:
            puissance_moy = float(df_raw["PUISSANCE"].mean())
            ptac_moy = float(df_raw["PTAC"].mean())
            historique_count = 0

        st.markdown("---")
        st.markdown("**📜 Historique de cette combinaison :**")
        st.write(f"- Nombre d'enregistrements historiques : **{historique_count}**")
        st.write(f"- Puissance moyenne estimée : **{puissance_moy:.2f}**")
        st.write(f"- PTAC moyen estimé : **{ptac_moy:.2f}**")

        lancer = st.button("🚀 Lancer la prédiction")

    with col_result:
        st.markdown('<div class="section-header">📈 Résultat</div>', unsafe_allow_html=True)

        if lancer:
            ville_enc = int(encoders["Ville"].transform([ville_sim])[0])
            marque_enc = int(encoders["Marque"].transform([marque_sim])[0])

            X_sim = pd.DataFrame([{
                "Année": annee_sim,
                "PUISSANCE_MOY": puissance_moy,
                "PTAC_MOY": ptac_moy,
                "Ville_enc": ville_enc,
                "Marque_enc": marque_enc
            }])

            prediction = float(model.predict(X_sim)[0])
            prediction_arrondie = max(0, round(prediction))

            result_card(
                prediction_arrondie=prediction_arrondie,
                ville=ville_sim,
                marque=marque_sim,
                annee=annee_sim,
                puissance_moy=puissance_moy,
                ptac_moy=ptac_moy
            )

            st.markdown("<br>", unsafe_allow_html=True)

            hist = (
                df_raw[(df_raw["Ville"] == ville_sim) & (df_raw["Marque"] == marque_sim)]
                .groupby("Année")
                .size()
                .reset_index(name="Ventes")
            )

            if len(hist) > 0:
                fig_hist = px.line(
                    hist,
                    x="Année",
                    y="Ventes",
                    markers=True,
                    title="Historique des ventes de cette combinaison",
                    template="plotly_white"
                )
                fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_hist, width="stretch")
            else:
                st.warning("Aucun historique disponible pour cette combinaison.")
        else:
            empty_result_card()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Power BI
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Power BI":

    st.markdown('<div class="main-title">📈 Intégration Power BI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ajoutez ici votre dashboard Power BI embarqué ou un lien de consultation</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔗 Intégration</div>', unsafe_allow_html=True)

    powerbi_url = ""

    if powerbi_url.strip():
        components.iframe(powerbi_url, height=700, scrolling=True)
    else:
        st.info("Ajoutez votre lien Power BI dans la variable `powerbi_url` داخل app.py.")
        st.code('powerbi_url = "https://app.powerbi.com/view?r=..."', language="python")