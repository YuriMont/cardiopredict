"""
Dashboard Clínico e Analítico de ECG Multilabel — CardioPredict AI Server.

Conecta-se ao banco colunar DuckDB (`cardiopredict.duckdb`) e aos contêineres HDF5,
fornecendo visualização interativa das 5 Superclasses, 25 Subclasses Clínicas,
prevenção de data leakage por paciente, medições fisiológicas e traçados em mV.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.constants import (
    DEFAULT_DB_PATH,
    DERIVED_INTERVAL_LIMITS,
    MIMIC_DIR,
    PHYSIOLOGICAL_LIMITS,
    PROCESSED_DATA_DIR,
    STANDARD_LEADS,
    SUBCLASSES,
    SUPERCLASSES,
    TARGET_CLASSES,
)
from processing.signal import load_raw_record, preprocess_ecg_for_cnn

st.set_page_config(
    page_title="CardioPredict AI — Dashboard Multilabel & Biossinais",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_duckdb_conn() -> duckdb.DuckDBPyConnection:
    """Retorna uma conexão em modo somente leitura com o banco DuckDB."""
    return duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)


def _fetch_scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is not None and row[0] is not None:
        return int(row[0])
    return 0


@st.cache_data(ttl=3600)
def load_global_kpis() -> dict[str, int]:
    """Carrega os indicadores globais do banco DuckDB de forma instantânea."""
    con = get_duckdb_conn()
    return {
        "total_records": _fetch_scalar(con, "SELECT count(*) FROM ecg_records"),
        "total_patients": _fetch_scalar(con, "SELECT count(*) FROM patients"),
        "total_supervised": _fetch_scalar(
            con,
            "SELECT count(*) FROM ecg_processed_labels WHERE is_supervised_target = TRUE",
        ),
        "mimic_count": _fetch_scalar(
            con, "SELECT count(*) FROM ecg_records WHERE source_dataset = 'mimic'"
        ),
        "ptbxl_count": _fetch_scalar(
            con, "SELECT count(*) FROM ecg_records WHERE source_dataset = 'ptbxl'"
        ),
        "mi_count": _fetch_scalar(
            con, "SELECT sum(label_mi) FROM ecg_processed_labels"
        ),
        "afib_count": _fetch_scalar(
            con, "SELECT sum(label_afib) FROM ecg_processed_labels"
        ),
    }


@st.cache_data(ttl=1800)
def load_superclasses_summary(split_filter: str, dataset_filter: str) -> pd.DataFrame:
    """Consulta a distribuição das 5 superclasses no DuckDB com filtros."""
    con = get_duckdb_conn()
    where_clauses: list[str] = ["is_supervised_target = TRUE"]

    if split_filter != "Todos":
        where_clauses.append(f"split = '{split_filter.lower()}'")
    if dataset_filter != "Todos":
        where_clauses.append(f"source_dataset = '{dataset_filter.lower()}'")

    where_sql = " AND ".join(where_clauses)

    sum_cols = [f"sum(label_{cls.lower()}) as {cls.upper()}" for cls in SUPERCLASSES]
    query = f"""
        SELECT 
            split,
            source_dataset,
            count(*) as total_exames,
            {", ".join(sum_cols)}
        FROM ecg_processed_labels
        WHERE {where_sql}
        GROUP BY split, source_dataset
        ORDER BY split, source_dataset
    """
    return con.execute(query).fetchdf()


@st.cache_data(ttl=1800)
def load_subclasses_distribution(
    split_filter: str, dataset_filter: str
) -> pd.DataFrame:
    """Calcula a prevalência de cada uma das 25 subclasses no DuckDB."""
    con = get_duckdb_conn()
    where_clauses: list[str] = ["is_supervised_target = TRUE"]

    if split_filter != "Todos":
        where_clauses.append(f"split = '{split_filter.lower()}'")
    if dataset_filter != "Todos":
        where_clauses.append(f"source_dataset = '{dataset_filter.lower()}'")

    where_sql = " AND ".join(where_clauses)

    results: list[dict[str, Any]] = []
    for sub in SUBCLASSES:
        q = f"""
            SELECT 
                sum(label_{sub.lower()}) as total_positives,
                sum(CASE WHEN label_{sub.lower()} = 1 AND weight_{sub.lower()} = 1.0 THEN 1 ELSE 0 END) as certas,
                sum(CASE WHEN label_{sub.lower()} = 1 AND weight_{sub.lower()} < 1.0 THEN 1 ELSE 0 END) as incertas
            FROM ecg_processed_labels
            WHERE {where_sql}
        """
        row = con.execute(q).fetchone()
        if row and row[0] is not None:
            results.append(
                {
                    "Subclasse": sub,
                    "Total": int(row[0]),
                    "Categórico (1.0)": int(row[1] or 0),
                    "Com Incerteza (0.5)": int(row[2] or 0),
                }
            )

    df = pd.DataFrame(results).sort_values("Total", ascending=False)
    return df


@st.cache_data(ttl=1800)
def load_measurements_sample(sample_size: int = 15000) -> pd.DataFrame:
    """Carrega uma amostra com medições fisiológicas e rótulos para boxplots."""
    con = get_duckdb_conn()
    query = f"""
        SELECT 
            m.rr_interval,
            m.heart_rate,
            m.pr_interval,
            m.qrs_duration,
            m.qt_interval,
            m.p_axis,
            m.qrs_axis,
            m.t_axis,
            p.label_norm,
            p.label_mi,
            p.label_afib,
            p.label_clbbb,
            p.label_crbbb,
            p.label_lvh,
            p.split,
            p.source_dataset
        FROM ecg_measurements m
        JOIN ecg_processed_labels p ON m.ecg_uid = p.ecg_uid
        WHERE m.rr_interval IS NOT NULL
        USING SAMPLE {sample_size}
    """
    return con.execute(query).fetchdf()


@st.cache_data(ttl=600)
def list_available_exams_with_signals(
    limit_per_dataset: int = 50,
    limit: int | None = None,
) -> pd.DataFrame:
    """Lista exames com traçados físicos disponíveis localmente (MIMIC e PTB-XL)."""
    if limit is not None:
        limit_per_dataset = max(10, limit // 2)

    con = get_duckdb_conn()

    # 1. Exames do PTB-XL
    q_ptb = f"""
        SELECT 
            r.ecg_uid,
            r.source_dataset,
            r.split,
            r.waveform_relative_path,
            p.label_norm,
            p.label_mi,
            p.label_afib,
            p.label_clbbb,
            p.label_ami,
            p.label_imi,
            p.is_signal_valid,
            p.hdf5_index
        FROM ecg_records r
        JOIN ecg_processed_labels p ON r.ecg_uid = p.ecg_uid
        WHERE r.source_dataset = 'ptbxl'
        LIMIT {limit_per_dataset}
    """
    df_ptb = con.execute(q_ptb).fetchdf()

    # 2. Exames do MIMIC que possuem arquivos físicos .hea/.dat presentes localmente
    mimic_files_dir = MIMIC_DIR / "files"
    mimic_uids: list[str] = []

    if mimic_files_dir.exists():
        for p_sub in sorted(mimic_files_dir.iterdir()):
            if not p_sub.is_dir():
                continue
            for hea_path in p_sub.glob("*/*/*.hea"):
                if hea_path.with_suffix(".dat").exists():
                    mimic_uids.append(f"mimic_{hea_path.stem}")
                    if len(mimic_uids) >= limit_per_dataset:
                        break
            if len(mimic_uids) >= limit_per_dataset:
                break

    if mimic_uids:
        q_mimic = f"""
            SELECT 
                r.ecg_uid,
                r.source_dataset,
                r.split,
                r.waveform_relative_path,
                p.label_norm,
                p.label_mi,
                p.label_afib,
                p.label_clbbb,
                p.label_ami,
                p.label_imi,
                p.is_signal_valid,
                p.hdf5_index
            FROM ecg_records r
            JOIN ecg_processed_labels p ON r.ecg_uid = p.ecg_uid
            WHERE r.ecg_uid IN {tuple(mimic_uids)}
            LIMIT {limit_per_dataset}
        """
        df_mimic = con.execute(q_mimic).fetchdf()
        combined = pd.concat([df_ptb, df_mimic], ignore_index=True)
    else:
        combined = df_ptb

    return combined


# =============================================================================
# Componentes Visuais do Dashboard
# =============================================================================
def render_header_kpis(kpis: dict[str, int]) -> None:
    """Exibe 6 cartões de métricas consolidadas em tempo real."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total de Exames", f"{kpis['total_records']:,}")
    c2.metric("Pacientes Únicos", f"{kpis['total_patients']:,}")
    c3.metric(
        "Alvos Supervisionados",
        f"{kpis['total_supervised']:,}",
        f"{kpis['total_supervised'] / kpis['total_records'] * 100:.1f}% da base",
    )
    c4.metric("Infarto do Miocárdio (MI)", f"{kpis['mi_count']:,}")
    c5.metric("Fibrilação Atrial (AFIB)", f"{kpis['afib_count']:,}")
    c6.metric(
        "Divisão Bi-Hospitalar",
        f"{kpis['ptbxl_count']:,} PTB-XL",
        f"{kpis['mimic_count']:,} MIMIC",
    )


def render_tab_multilabel_taxonomy(
    df_super: pd.DataFrame, df_sub: pd.DataFrame
) -> None:
    """Aba 1: Distribuição das 5 Superclasses e 25 Subclasses Clínicas."""
    st.subheader(
        "Taxonomia Multilabel: 5 Superclasses SCP-ECG e 25 Patologias Críticas"
    )

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### 1. Volume das 5 Grandes Superclasses Clínicas")
        totals_super = {
            cls: int(np.sum(df_super[cls].to_numpy())) for cls in SUPERCLASSES
        }
        df_super_plot = pd.DataFrame(
            list(totals_super.items()), columns=["Superclasse", "Total Exames"]
        )

        fig_super = px.bar(
            df_super_plot,
            x="Superclasse",
            y="Total Exames",
            color="Superclasse",
            text_auto=True,
            title="Prevalência das 5 Superclasses (Multi-label)",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(fig_super, use_container_width=True)

    with col_r:
        st.markdown("#### 2. Distribuição por Partição (Zero Data Leakage)")
        df_split_melt = df_super.melt(
            id_vars=["split"],
            value_vars=list(SUPERCLASSES),
            var_name="Superclasse",
            value_name="Exames",
        )
        fig_split = px.bar(
            df_split_melt,
            x="Superclasse",
            y="Exames",
            color="split",
            barmode="group",
            category_orders={"split": ["train", "val", "test"]},
            title="Consistência de Distribuição em Treino (80%), Val (10%) e Teste (10%)",
        )
        st.plotly_chart(fig_split, use_container_width=True)

    st.divider()

    st.markdown("#### 3. As 25 Subclasses Específicas de Emergência Cardiológica")
    fig_sub = px.bar(
        df_sub,
        x="Total",
        y="Subclasse",
        orientation="h",
        color="Total",
        color_continuous_scale="Viridis",
        title="Ranking de Prevalência das 25 Subclasses Clínicas (MIMIC-IV-ECG & PTB-XL)",
        height=680,
    )
    fig_sub.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_sub, use_container_width=True)


def render_tab_uncertainty_weights(df_sub: pd.DataFrame) -> None:
    """Aba 2: Análise de Incerteza Diagnóstica e Pesos Ponderados."""
    st.subheader(
        "Ponderação de Incerteza Clínica (Pesos 1.0 Categóricos vs. 0.5 Incerteza)"
    )

    st.info(
        "**Diretriz Clínica de Modelagem:** Em laudos médicos reais, frases como *'possible inferior infarct'* "
        "ou *'borderline LVH'* expressam dúvida diagnóstica. O pipeline preserva essas amostras com o rótulo positivo (`1`), "
        "mas atribui `confidence_weight = 0.5` na função `BCEWithLogitsLoss`. Isso impede que o modelo ignore "
        "achados sutis sem sofrer penalização de gradiente desproporcional."
    )

    df_melt = df_sub.melt(
        id_vars=["Subclasse"],
        value_vars=["Categórico (1.0)", "Com Incerteza (0.5)"],
        var_name="Tipo de Afirmação",
        value_name="Ocorrências",
    )

    fig_unc = px.bar(
        df_melt,
        x="Subclasse",
        y="Ocorrências",
        color="Tipo de Afirmação",
        barmode="stack",
        color_discrete_map={
            "Categórico (1.0)": "#2ca02c",
            "Com Incerteza (0.5)": "#ff7f0e",
        },
        title="Proporção de Certeza vs. Incerteza por Diagnóstico",
        height=520,
    )
    st.plotly_chart(fig_unc, use_container_width=True)

    st.markdown("#### Tabela Comparativa de Incerteza por Subclasse")
    st.dataframe(df_sub, use_container_width=True)


def render_tab_measurements(df_meas: pd.DataFrame) -> None:
    """Aba 3: Distribuição das Medições Fisiológicas e Detecção de Anomalias."""
    st.subheader("Medições Eletrofisiológicas Validadas e Limites Humanos (AHA)")

    st.info(
        "💡 **Origem dos Dados:** As variáveis fisiológicas de intervalo (`heart_rate`, `pr_interval`, `qrs_duration`, "
        "`qt_interval`, `rr_interval`) são provenientes exclusivamente do **MIMIC-IV-ECG** "
        "(geradas pelo algoritmo proprietário GE Marquette 12SL durante a aquisição clínica). "
        "O dataset **PTB-XL** fornece parâmetros antropométricos e laudos estruturados SCP-ECG, mas **não** inclui "
        "tabelas com intervalos milissegundos pré-calculados pelo hardware."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        metric = st.selectbox(
            "Selecione a Variável Fisiológica:",
            options=[
                "heart_rate",
                "pr_interval",
                "qrs_duration",
                "qt_interval",
                "rr_interval",
            ],
            index=2,  # qrs_duration
        )

    all_limits = {**PHYSIOLOGICAL_LIMITS, **DERIVED_INTERVAL_LIMITS}
    limit_str = f"{all_limits[metric]}" if metric in all_limits else "[20.0, 300.0] BPM"

    with c2:
        st.info(
            f"**Variável Selecionada:** `{metric}` | **Faixa de Limite Fisiológico Válido:** `{limit_str}`"
        )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_hist = px.histogram(
            df_meas,
            x=metric,
            nbins=50,
            title=f"Distribuição Fisiológica de `{metric}` (Após Limpeza)",
            color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_g2:
        # Boxplot comparando QRS em Bloqueio de Ramo Esquerdo (CLBBB) vs Normal
        df_meas_clean = df_meas.dropna(subset=[metric]).copy()
        df_meas_clean["Grupo"] = "Geral"
        if "label_clbbb" in df_meas_clean.columns:
            df_meas_clean.loc[df_meas_clean["label_clbbb"] == 1, "Grupo"] = (
                "Com BRE (CLBBB)"
            )
        if "label_norm" in df_meas_clean.columns:
            df_meas_clean.loc[df_meas_clean["label_norm"] == 1, "Grupo"] = "Normal"

        fig_box = px.box(
            df_meas_clean,
            x="Grupo",
            y=metric,
            color="Grupo",
            title=f"Comparação de `{metric}` por Condição Clínica",
            color_discrete_sequence=["#2ca02c", "#d62728", "#7f7f7f"],
        )
        st.plotly_chart(fig_box, use_container_width=True)


def render_tab_signals(df_exams: pd.DataFrame) -> None:
    """Aba 4: Visualizador de Ondas de ECG de 12 Derivações (WFDB e HDF5)."""
    st.subheader("Visualizador de Biossinais: 12 Derivações a 500 Hz em mV")

    if df_exams.empty:
        st.warning("Nenhum exame local encontrado com arquivos .hea/.dat válidos.")
        return

    c_filter, c_dur = st.columns([2, 1])
    with c_filter:
        base_choice = st.radio(
            "Filtrar Base Hospitalar do Exame:",
            options=["Todas as Bases", "MIMIC-IV-ECG (EUA)", "PTB-XL (Alemanha)"],
            horizontal=True,
        )

    with c_dur:
        duration_s = st.slider(
            "Duração exibida (segundos):",
            min_value=2.0,
            max_value=10.0,
            value=4.0,
            step=1.0,
        )

    df_filtered = df_exams.copy()
    if base_choice == "MIMIC-IV-ECG (EUA)":
        df_filtered = df_filtered.loc[df_filtered["source_dataset"] == "mimic"]
    elif base_choice == "PTB-XL (Alemanha)":
        df_filtered = df_filtered.loc[df_filtered["source_dataset"] == "ptbxl"]

    if df_filtered.empty:
        st.warning(
            f"Nenhum exame local encontrado para a base selecionada: {base_choice}"
        )
        return

    exam_options = df_filtered["ecg_uid"].tolist()

    def format_exam_label(uid: str) -> str:
        row = df_filtered.loc[df_filtered["ecg_uid"] == uid].iloc[0]
        diags: list[str] = []
        if row.get("label_norm") == 1:
            diags.append("Normal")
        if row.get("label_mi") == 1:
            diags.append("Infarto (MI)")
        if row.get("label_afib") == 1:
            diags.append("AFIB")
        if row.get("label_clbbb") == 1:
            diags.append("BRE (CLBBB)")
        if row.get("label_ami") == 1:
            diags.append("Infarto Anterior")
        if row.get("label_imi") == 1:
            diags.append("Infarto Inferior")
        diag_str = ", ".join(diags) if diags else "Outros Achados"
        return f"{uid} ({row['source_dataset'].upper()} - {row['split'].upper()}) | {diag_str}"

    selected_uid = st.selectbox(
        "Selecione o Exame (`ecg_uid`):",
        options=exam_options,
        format_func=format_exam_label,
    )

    row_exam = df_filtered.loc[df_filtered["ecg_uid"] == selected_uid].iloc[0]

    try:
        # Carrega e filtra o sinal físico usando o pipeline validado
        record = load_raw_record(row_exam["waveform_relative_path"])
        cnn_tensor, _, is_valid = preprocess_ecg_for_cnn(record)

        fs = float(record.fs or 500.0)
        n_points = int(duration_s * fs)
        time_axis = np.arange(n_points) / fs

        fig = make_subplots(
            rows=6,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=[f"Derivação {lead}" for lead in STANDARD_LEADS[:12]],
        )

        for idx, lead_name in enumerate(STANDARD_LEADS[:12]):
            r = (idx % 6) + 1
            c = (idx // 6) + 1
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=cnn_tensor[:n_points, idx],
                    mode="lines",
                    name=lead_name,
                    line={"color": "#1f77b4", "width": 1.5},
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            fig.update_yaxes(title_text="mV", row=r, col=c)

        fig.update_layout(
            height=850,
            title_text=f"Traçado de 12 Derivações — Exame {selected_uid} ({row_exam['source_dataset'].upper()})",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"Exame: `{selected_uid}` | Base: `{row_exam['source_dataset']}` | Split: `{row_exam['split']}` | "
            f"Sinal Físico Válido: `{is_valid}`"
        )

    except (OSError, ValueError, RuntimeError, KeyError) as e:
        st.error(f"Erro ao carregar traçado WFDB: {e}")


def render_tab_duckdb_explorer() -> None:
    """Aba 5: Consultas Analíticas Diretas e Tabela de Dados no DuckDB."""
    st.subheader("Consultas Analíticas Diretas no DuckDB (`cardiopredict.duckdb`)")

    con = get_duckdb_conn()

    st.markdown("#### Filtro Dinâmico de Exames")
    c1, c2, c3 = st.columns(3)
    with c1:
        f_split = st.selectbox(
            "Filtrar Split:", options=["Todos", "train", "val", "test"]
        )
    with c2:
        f_dataset = st.selectbox("Filtrar Base:", options=["Todos", "mimic", "ptbxl"])
    with c3:
        f_class = st.selectbox(
            "Filtrar Patologia Positiva:",
            options=["Todas"] + list(TARGET_CLASSES),
        )

    where_clauses: list[str] = ["p.is_supervised_target = TRUE"]
    if f_split != "Todos":
        where_clauses.append(f"p.split = '{f_split}'")
    if f_dataset != "Todos":
        where_clauses.append(f"p.source_dataset = '{f_dataset}'")
    if f_class != "Todas":
        where_clauses.append(f"p.label_{f_class.lower()} = 1")

    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT 
            p.ecg_uid,
            p.patient_uid,
            p.source_dataset,
            p.split,
            p.label_norm,
            p.label_mi,
            p.label_sttc,
            p.label_cd,
            p.label_hyp,
            p.label_afib,
            p.label_ami,
            p.label_imi,
            p.label_clbbb,
            p.is_signal_valid,
            p.hdf5_index
        FROM ecg_processed_labels p
        WHERE {where_sql}
        LIMIT 500
    """
    df_result = con.execute(query).fetchdf()

    st.metric("Registros Encontrados (Limite 500 para exibição)", len(df_result))
    st.dataframe(df_result, use_container_width=True, height=450)


# =============================================================================
# Função Principal (Main)
# =============================================================================
def main() -> None:
    st.title("🫀 CardioPredict AI — Núcleo de Biossinais e Multilabel")
    st.caption(
        "Painel analítico integrado: 821 mil exames, consolidação DuckDB, taxonomia 5+25 e contêineres HDF5."
    )

    # 1. Carrega KPIs Globais
    kpis = load_global_kpis()
    render_header_kpis(kpis)

    st.divider()

    # 2. Barra Lateral de Filtros Globais
    st.sidebar.header("Filtros do Painel")
    f_split = st.sidebar.selectbox(
        "Partição de Pacientes:", ["Todos", "train", "val", "test"]
    )
    f_dataset = st.sidebar.selectbox("Base Hospitalar:", ["Todos", "mimic", "ptbxl"])

    # Verifica status dos contêineres HDF5 em disco
    train_h5 = PROCESSED_DATA_DIR / "train.h5"
    val_h5 = PROCESSED_DATA_DIR / "val.h5"
    test_h5 = PROCESSED_DATA_DIR / "test.h5"

    st.sidebar.subheader("Armazenamento HDF5")
    st.sidebar.text(f"train.h5: {'Ativo' if train_h5.exists() else 'Pendente'}")
    st.sidebar.text(f"val.h5: {'Ativo' if val_h5.exists() else 'Pendente'}")
    st.sidebar.text(f"test.h5: {'Ativo' if test_h5.exists() else 'Pendente'}")

    # 3. Carrega dados filtrados para as abas
    df_super = load_superclasses_summary(f_split, f_dataset)
    df_sub = load_subclasses_distribution(f_split, f_dataset)
    df_meas = load_measurements_sample(sample_size=15000)
    df_exams = list_available_exams_with_signals(limit=100)

    # 4. Renderiza Abas Analíticas
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 1. Taxonomia (5+25 Classes)",
            "⚖️ 2. Incerteza Diagnóstica",
            "📈 3. Fisiologia & Limites AHA",
            "⚡ 4. Ondas 12 Derivações (mV)",
            "📋 5. Consultas DuckDB",
        ]
    )

    with tab1:
        render_tab_multilabel_taxonomy(df_super, df_sub)
    with tab2:
        render_tab_uncertainty_weights(df_sub)
    with tab3:
        render_tab_measurements(df_meas)
    with tab4:
        render_tab_signals(df_exams)
    with tab5:
        render_tab_duckdb_explorer()


if __name__ == "__main__":
    main()
