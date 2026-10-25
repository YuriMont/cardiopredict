from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from clean_measurements import DERIVED_INTERVAL_LIMITS, clean_machine_measurements
from inspect_database import PHYSIOLOGICAL_LIMITS
from normalize_report import (
    classify_report_line,
    extract_core_diagnosis,
    normalize_report,
)
from preprocess_ecg import (
    STANDARD_LEADS,
    load_raw_record,
    preprocess_ecg_for_cnn,
)
from read_database import DATA_DIR
from split_patients import build_study_diagnostic_status, split_by_patient

ALL_NUMERIC_METRICS = [
    "rr_interval",
    "heart_rate_bpm",
    "pr_interval",
    "qrs_duration",
    "qt_interval",
    "p_duration",
    "p_onset",
    "p_end",
    "qrs_onset",
    "qrs_end",
    "t_end",
    "p_axis",
    "qrs_axis",
    "t_axis",
]


@st.cache_data(show_spinner="Carregando e processando amostra da base MIMIC-IV-ECG...")
def load_dashboard_data(max_rows: int = 100_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega uma amostra representativa de machine_measurements.csv e record_list.csv,
    retornando os dados brutos e os dados limpos enriquecidos com:
    - `ecg_status` ('normal', 'borderline', 'abnormal')
    - `findings_str`, `primary_finding`, `n_findings` e `is_multilabel` (para CNN Multilabel)
    - `split` agrupado por paciente ('train', 'val', 'test')
    """
    raw_df = pd.read_csv(
        DATA_DIR / "machine_measurements.csv",
        nrows=max_rows,
        low_memory=False,
    )
    records_df = pd.read_csv(
        DATA_DIR / "record_list.csv",
        nrows=max_rows,
        low_memory=False,
    )

    cleaned_df = clean_machine_measurements(raw_df)
    status_df = build_study_diagnostic_status(raw_df)

    merged_clean = cleaned_df.merge(
        status_df.drop(columns=["subject_id"]),
        on="study_id",
        how="inner",
    ).merge(
        records_df[["study_id", "path"]],
        on="study_id",
        how="left",
    )

    merged_clean = split_by_patient(merged_clean, random_state=42)
    return raw_df, merged_clean


@st.cache_data(show_spinner="Extraindo e normalizando todos os reports da amostra...")
def load_normalized_reports_sample(
    raw_df: pd.DataFrame, max_exams: int = 50_000
) -> pd.DataFrame:
    """Extrai e normaliza todas as colunas report_0..report_17 da amostra."""
    sub = raw_df.head(max_exams)
    report_cols = sorted(
        (c for c in sub.columns if c.startswith("report_")),
        key=lambda c: int(c.split("_")[1]),
    )
    rows = []
    matrix = sub[report_cols].to_numpy()
    for col_idx, col_name in enumerate(report_cols):
        col_vals = matrix[:, col_idx]
        for val in col_vals:
            norm_text = normalize_report(val)
            if norm_text is not None:
                raw_text = str(val).strip()
                cat, is_unc, is_bord, weight = classify_report_line(norm_text)
                rows.append(
                    {
                        "report_column": col_name,
                        "raw_report_text": raw_text,
                        "report_text": norm_text,
                        "core_diagnosis": extract_core_diagnosis(norm_text),
                        "diagnostic_category": cat,
                        "Incerteza": "Com Incerteza (possible/probable)"
                        if is_unc
                        else "Definitivo",
                        "is_uncertain": is_unc,
                        "is_borderline": is_bord,
                        "confidence_weight": weight,
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Desmembrando findings multilabel por exame...")
def explode_exam_findings(
    clean_df: pd.DataFrame, max_exams: int = 25_000
) -> pd.DataFrame:
    """
    DECISÃO MÉDICA E DE IA (PROJEÇÃO MULTILABEL COM TEXTO EXATO DA BASE):
    Um mesmo ECG (`study_id`) frequentemente apresenta múltiplos achados simultâneos em
    `findings_str` preservando o texto exato gravado na base (ex.: 'Sinus rhythm | Left axis deviation').
    Esta função utiliza `.explode()` para projetar cada achado individualmente sem alterar os
    KPIs globais, permitindo comparar os Box Plots fisiológicos de cada patologia.
    """
    sub = clean_df.head(max_exams).copy()
    sub["finding_item"] = sub["findings_str"].str.split(r" \| ")
    exploded = sub.explode("finding_item")
    return pd.DataFrame(exploded.loc[exploded["finding_item"] != "none"])


@st.cache_data(show_spinner="Calculando ranking dos Findings exatos da base...")
def get_top_findings_ranking(clean_df: pd.DataFrame) -> pd.Series:
    """
    Retorna uma Series indexada pelo texto exato do Finding na base (`machine_measurements.csv`)
    com a contagem de exames em que cada Finding ocorre, ordenada da maior para a menor frequência.
    """
    exploded = explode_exam_findings(clean_df, max_exams=len(clean_df))
    if exploded.empty:
        return pd.Series(dtype=int)
    return exploded["finding_item"].value_counts()


@st.cache_data(show_spinner="Buscando arquivos locais de ondas ECG (.hea/.dat)...")
def list_available_local_waveforms() -> list[tuple[str, int, Path]]:
    """Lista os arquivos .hea realmente presentes na pasta data/files/ para visualização."""
    files_dir = DATA_DIR / "files"
    if not files_dir.exists():
        return []
    hea_files = sorted(files_dir.rglob("*.hea"))[:100]
    return [
        (
            f"Study {p.stem} ({p.parent.parent.name})",
            int(p.stem),
            p.relative_to(DATA_DIR),
        )
        for p in hea_files
    ]


def render_kpi_cards(df: pd.DataFrame) -> None:
    """Renderiza a faixa superior de 6 cartões KPI estilo Power BI (incluindo Multilabel)."""
    total_exams = len(df)
    unique_patients = len(set(df["subject_id"])) if total_exams > 0 else 0
    mean_bpm = float(df["heart_rate_bpm"].to_numpy().mean()) if total_exams > 0 else 0.0
    multilabel_pct = (
        float((df["is_multilabel"] == 1).mean() * 100) if total_exams > 0 else 0.0
    )
    mean_findings = (
        float(df["n_findings"].to_numpy().mean()) if total_exams > 0 else 0.0
    )
    borderline_pct = (
        float((df["ecg_status"] == "borderline").mean() * 100)
        if total_exams > 0
        else 0.0
    )
    p_missing_pct = (
        float((df["p_wave_missing"] == 1).mean() * 100) if total_exams > 0 else 0.0
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Exames Filtrados", f"{total_exams:,}")
    c2.metric("Pacientes Únicos", f"{unique_patients:,}")
    c3.metric("Média BPM", f"{mean_bpm:.1f}")
    c4.metric(
        "Multilabel (>=2 Findings)",
        f"{multilabel_pct:.1f}%",
        f"Média: {mean_findings:.2f} / exame",
    )
    c5.metric("Borderline ECG", f"{borderline_pct:.1f}%")
    c6.metric("Onda P Ausente (FA)", f"{p_missing_pct:.1f}%")


def render_tab_boxplots(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    top_k_findings: int = 15,
) -> None:
    """Aba 1: Box Plots por Status Geral e por Finding Multilabel (com texto exato da base)."""
    st.subheader(
        "Análise de Outliers, Limites Humanos e Box Plots por Finding (Texto Exato da Base)"
    )

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_metric = st.selectbox(
            "Selecione a Variável do ECG:",
            options=ALL_NUMERIC_METRICS,
            index=3,  # qrs_duration por padrão para evidenciar bloqueios de ramo
        )
        max_plot_points = st.slider(
            "Amostra para plotagem rápida:",
            min_value=5_000,
            max_value=50_000,
            value=15_000,
            step=5_000,
        )

    all_limits = {**PHYSIOLOGICAL_LIMITS, **DERIVED_INTERVAL_LIMITS}
    limit_str = (
        f"{all_limits[selected_metric]}"
        if selected_metric in all_limits
        else "[20.0, 300.0] BPM"
    )
    with col_info:
        st.info(
            f"**Variável selecionada:** `{selected_metric}` | "
            f"**Limite Fisiológico Humano:** `{limit_str}`\n\n"
            "**Como tratamos exames com mais de um diagnóstico (`Finding`) nos Box Plots abaixo:** "
            "Um paciente que possui simultaneamente *Fibrilação Atrial* e *Bloqueio de Ramo Direito* "
            "tem suas medições projetadas em ambos os grupos via desmembramento multilabel (`.explode()`), "
            "permitindo enxergar a assinatura fisiológica exata de cada patologia!"
        )

    clean_sample = clean_df.head(max_plot_points)

    c1, c2 = st.columns(2)
    with c1:
        if selected_metric in raw_df.columns:
            raw_sample = raw_df[[selected_metric]].head(max_plot_points)
            fig_raw = px.box(
                raw_sample,
                y=selected_metric,
                title=f"1. ANTES da Limpeza: {selected_metric} (Com Erros 29999 / -32768)",
                color_discrete_sequence=["#d62728"],
            )
            st.plotly_chart(fig_raw, use_container_width=True)
        else:
            st.warning(
                f"A variável `{selected_metric}` é calculada após a limpeza das ondas brutas."
            )

    with c2:
        fig_clean = px.box(
            clean_sample,
            x="ecg_status",
            y=selected_metric,
            color="ecg_status",
            category_orders={"ecg_status": ["normal", "borderline", "abnormal"]},
            color_discrete_map={
                "normal": "#2ca02c",
                "borderline": "#ff7f0e",
                "abnormal": "#d62728",
            },
            title=f"2. DEPOIS da Limpeza: {selected_metric} por Status Geral (ecg_status)",
        )
        st.plotly_chart(fig_clean, use_container_width=True)

    # Box Plot agrupado pelos Top N Findings (com texto exato da base)
    exploded_df = explode_exam_findings(clean_df, max_exams=max_plot_points)
    if not exploded_df.empty:
        top_n_findings = (
            exploded_df["finding_item"]
            .value_counts()
            .head(top_k_findings)
            .index.tolist()
        )
        exploded_top = exploded_df.loc[exploded_df["finding_item"].isin(top_n_findings)]

        fig_find_box = px.box(
            exploded_top,
            x=selected_metric,
            y="finding_item",
            color="ecg_status",
            orientation="h",
            category_orders={"finding_item": top_n_findings},
            color_discrete_map={
                "normal": "#2ca02c",
                "borderline": "#ff7f0e",
                "abnormal": "#d62728",
            },
            title=(
                f"3. Box Plot de `{selected_metric}` por Finding "
                f"(Top {len(top_n_findings)} Textos Exatos da Base)"
            ),
            height=max(480, 28 * len(top_n_findings) + 140),
        )
        st.plotly_chart(fig_find_box, use_container_width=True)

    st.markdown("#### Tabela Estatística `.describe()` — Dados Limpos e Validados")
    desc_df = (
        clean_df[ALL_NUMERIC_METRICS]
        .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
        .T.round(2)
    )
    st.dataframe(desc_df, use_container_width=True)


def render_tab_diagnostics_and_splits(
    clean_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    top_k_findings: int = 15,
) -> None:
    """Aba 2: Cardinalidade Multilabel, Top Findings por Split e Explicação do Tratamento."""
    st.subheader(
        "Tratamento de Múltiplos Diagnósticos (Multilabel), Findings e Divisão por Pacientes"
    )

    st.success(
        "**Como os casos com mais de um diagnóstico (`Finding`) são tratados para a CNN (Keras / sklearn):**\n"
        "1. **Não descartamos diagnósticos secundários:** Em vez de pegar apenas `report_0`, extraímos todos os "
        "achados do exame em `findings_str` (ex.: `'sinus bradycardia | first degree av block | left axis deviation'`).\n"
        "2. **Padronização para `core_diagnosis`:** Unificamos prefixos de incerteza (ex.: `'possible left atrial abnormality'` "
        "$\\rightarrow$ `'left atrial abnormality'`) mantendo o peso `sample_weight = 0.5` quando há incerteza.\n"
        "3. **Codificação Multilabel (`MultiLabelBinarizer` do `sklearn`):** Na CNN do Keras, um exame com 3 diagnósticos "
        "recebe um vetor $y$ com três posições ativas (`1`), treinado com `Dense(n_classes, activation='sigmoid')` "
        "e `loss='binary_crossentropy'` (pois `softmax` só serve quando há 1 único diagnóstico por exame!)."
    )

    col1, col2 = st.columns(2)
    with col1:
        # Gráfico de Cardinalidade Multilabel (Quantos findings simultâneos cada exame tem)
        card_df = clean_df.copy()
        card_df["Qtd Findings no Exame"] = np.where(
            card_df["n_findings"] >= 5,
            "5+ findings",
            card_df["n_findings"].astype(str) + " finding(s)",
        )
        card_counts = card_df.value_counts(
            ["Qtd Findings no Exame", "ecg_status"]
        ).reset_index(name="Exames")

        fig_card = px.bar(
            card_counts,
            x="Qtd Findings no Exame",
            y="Exames",
            color="ecg_status",
            barmode="stack",
            category_orders={
                "Qtd Findings no Exame": [
                    "0 finding(s)",
                    "1 finding(s)",
                    "2 finding(s)",
                    "3 finding(s)",
                    "4 finding(s)",
                    "5+ findings",
                ],
                "ecg_status": ["normal", "borderline", "abnormal"],
            },
            color_discrete_map={
                "normal": "#2ca02c",
                "borderline": "#ff7f0e",
                "abnormal": "#d62728",
            },
            title="Cardinalidade Multilabel: Quantos Diagnósticos (Findings) um Mesmo Exame Possui?",
        )
        st.plotly_chart(fig_card, use_container_width=True)

    with col2:
        split_summary = clean_df.value_counts(["split", "ecg_status"]).reset_index(
            name="Exames"
        )
        fig_split = px.bar(
            split_summary,
            x="split",
            y="Exames",
            color="ecg_status",
            barmode="group",
            category_orders={
                "split": ["train", "val", "test"],
                "ecg_status": ["normal", "borderline", "abnormal"],
            },
            color_discrete_map={
                "normal": "#2ca02c",
                "borderline": "#ff7f0e",
                "abnormal": "#d62728",
            },
            title="Divisão por Pacientes (70% Train / 15% Val / 15% Test — Zero Data Leakage)",
        )
        st.plotly_chart(fig_split, use_container_width=True)

    # Distribuição dos Top N Findings (texto exato da base) nos conjuntos Train / Val / Test
    exploded_df = explode_exam_findings(clean_df, max_exams=30_000)
    if not exploded_df.empty:
        top_findings_names = (
            exploded_df["finding_item"]
            .value_counts()
            .head(top_k_findings)
            .index.tolist()
        )
        find_split_counts = (
            exploded_df.loc[exploded_df["finding_item"].isin(top_findings_names)]
            .value_counts(["finding_item", "split"])
            .reset_index(name="Ocorrências")
        )
        fig_find_split = px.bar(
            find_split_counts,
            x="Ocorrências",
            y="finding_item",
            color="split",
            orientation="h",
            barmode="stack",
            category_orders={
                "finding_item": top_findings_names,
                "split": ["train", "val", "test"],
            },
            title=(
                f"Top {len(top_findings_names)} Findings (Texto Exato da Base) "
                "Distribuídos em Train / Val / Test"
            ),
            height=max(480, 26 * len(top_findings_names) + 140),
        )
        st.plotly_chart(fig_find_split, use_container_width=True)

    _ = reports_df


def render_tab_all_reports_explorer(reports_df: pd.DataFrame) -> None:
    """Aba 3: Explorador completo de todas as colunas report_0..report_17 e categorias."""
    st.subheader("Explorador Completo de Todos os Reports (`report_0` até `report_17`)")

    if reports_df.empty:
        st.warning("Nenhum report carregado.")
        return

    col_counts = (
        reports_df["report_column"]
        .value_counts()
        .rename_axis("Coluna")
        .reset_index(name="Total Preenchido")
    )
    col_counts["ordem"] = col_counts["Coluna"].map(lambda c: int(str(c).split("_")[1]))
    col_counts = col_counts.sort_values("ordem")

    fig_cols = px.bar(
        col_counts,
        x="Coluna",
        y="Total Preenchido",
        text_auto=True,
        title="Volume de Laudos Preenchidos em Cada Coluna (report_0 até report_17)",
        color="Total Preenchido",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_cols, use_container_width=True)

    st.markdown("#### Filtrar e Pesquisar Todos os Textos de Reports")
    fcol1, fcol2, fcol3, fcol4 = st.columns([1.2, 1.2, 1.2, 1.8])

    all_rep_cols = col_counts["Coluna"].tolist()
    with fcol1:
        sel_rep_cols = st.multiselect(
            "Filtrar por Coluna (report_#):",
            options=all_rep_cols,
            default=all_rep_cols,
        )
    with fcol2:
        sel_cats = st.multiselect(
            "Categoria Diagnóstica:",
            options=["finding", "normal", "borderline", "abnormal"],
            default=["finding", "normal", "borderline", "abnormal"],
        )
    with fcol3:
        sel_unc = st.selectbox(
            "Presença de Incerteza:",
            options=[
                "Todos",
                "Definitivo",
                "Com Incerteza (possible/probable)",
            ],
            index=0,
        )
    with fcol4:
        search_query = st.text_input(
            "Pesquisar palavra no laudo (ex: infarct, fibrillation, block, borderline):",
            value="",
        )

    rep_mask = reports_df["report_column"].isin(sel_rep_cols) & reports_df[
        "diagnostic_category"
    ].isin(sel_cats)
    if sel_unc != "Todos":
        rep_mask = rep_mask & (reports_df["Incerteza"] == sel_unc)
    if search_query.strip():
        rep_mask = rep_mask & reports_df["report_text"].str.contains(
            search_query.strip().lower(), na=False, regex=False
        )

    sub_reports = pd.DataFrame(reports_df.loc[rep_mask])

    catalog = (
        sub_reports.value_counts(
            [
                "raw_report_text",
                "report_text",
                "core_diagnosis",
                "diagnostic_category",
                "Incerteza",
                "is_borderline",
                "confidence_weight",
            ]
        )
        .reset_index(name="Ocorrências")
        .sort_values("Ocorrências", ascending=False)
    )
    total_sub = max(1, len(sub_reports))
    catalog["Percentual (%)"] = (catalog["Ocorrências"] / total_sub * 100).round(3)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total de Ocorrências Filtradas", f"{len(sub_reports):,}")
    m2.metric("Textos Únicos de Laudos Encontrados", f"{len(catalog):,}")
    m3.metric(
        "Com Incerteza Preservada",
        f"{int((sub_reports['is_uncertain'] == 1).sum()):,}",
    )

    if not catalog.empty:
        top_n = catalog.head(25).sort_values("Ocorrências", ascending=True)
        fig_top = px.bar(
            top_n,
            x="Ocorrências",
            y="raw_report_text",
            color="diagnostic_category",
            orientation="h",
            color_discrete_map={
                "normal": "#2ca02c",
                "borderline": "#ff7f0e",
                "abnormal": "#d62728",
                "finding": "#1f77b4",
            },
            title="Top 25 Reports Mais Frequentes no Filtro Atual (Texto Exato da Base)",
            height=620,
        )
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("#### Catálogo Completo de Todos os Reports Únicos Filtrados")
    st.dataframe(catalog, use_container_width=True, height=500)


def render_tab_ecg_signals(clean_df: pd.DataFrame) -> None:
    """Aba 4: Visualizador dos 12 canais em mV exibindo os Findings Multilabel do exame."""
    st.subheader(
        "Explorador de Ondas ECG (12 Derivações na Mesma Escala mV + Findings Multilabel do Exame)"
    )

    available_files = list_available_local_waveforms()
    if not available_files:
        st.warning("Nenhum arquivo .hea encontrado em data/files/.")
        return

    col_select, col_opts = st.columns([2, 2])
    with col_select:
        labels = [item[0] for item in available_files]
        selected_idx = st.selectbox(
            "Selecione um Exame Local (.hea/.dat):",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
        )
    with col_opts:
        time_window = st.slider(
            "Janela de Tempo Exibida (segundos):",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=1.0,
        )

    _, study_id, rel_path = available_files[selected_idx]

    # Exibe os Findings Multilabel e o Status Geral desse exame específico
    exam_row = clean_df.loc[clean_df["study_id"] == study_id]
    if not exam_row.empty:
        row_info = exam_row.iloc[0]
        st.info(
            f"**Exame (`study_id`):** `{study_id}` | "
            f"**Status Geral (`ecg_status`):** `{row_info['ecg_status']}` | "
            f"**Qtd Findings (`n_findings`):** `{row_info['n_findings']}` | "
            f"**Conjunto (`split`):** `{row_info['split']}`\n\n"
            f"**Findings Multilabel do Exame:** `{row_info['findings_str']}`"
        )

    record = load_raw_record(rel_path)
    _, steps, is_valid = preprocess_ecg_for_cnn(record)

    if not is_valid or not steps:
        st.error(
            "Este exame reprovou na validação de qualidade (excesso de NaN ou Flatline)."
        )
        return

    fs = float(record.fs or 500.0)
    max_samples = int(min(steps["raw_mv"].shape[0], time_window * fs))
    time_sec = np.arange(max_samples) / fs

    global_max_mv = float(
        max(
            np.abs(steps["raw_mv"][:max_samples]).max(),
            np.abs(steps["clean_mv"][:max_samples]).max(),
        )
    )
    y_limit = max(1.5, round(global_max_mv * 1.1, 1))

    fig = make_subplots(
        rows=6,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[f"Derivação {lead}" for lead in STANDARD_LEADS],
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    for idx, _ in enumerate(STANDARD_LEADS):
        r = (idx % 6) + 1
        c = (idx // 6) + 1
        show_leg = idx == 0

        fig.add_trace(
            go.Scatter(
                x=time_sec,
                y=steps["raw_mv"][:max_samples, idx],
                name="Sinal Bruto (mV)",
                line={"color": "#ff7f0e", "width": 1.0},
                opacity=0.65,
                showlegend=show_leg,
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=time_sec,
                y=steps["clean_mv"][:max_samples, idx],
                name="Limpo (Passa-Alta 0.5Hz + Notch 50/60Hz + Passa-Baixa 40Hz)",
                line={"color": "#1f77b4", "width": 1.3},
                showlegend=show_leg,
            ),
            row=r,
            col=c,
        )

    fig.update_yaxes(range=[-y_limit, y_limit], title_text="mV")
    fig.update_layout(
        height=950,
        title_text=f"12 Derivações na Mesma Escala Física (±{y_limit} mV) — Unidade Original: {record.units[0] if record.units else 'mV'}",
        legend={"orientation": "h", "y": 1.03, "x": 0.0},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Auditoria de Ruído Removido e Escala por Canal (mV)")
    raw_s_df = pd.DataFrame(steps["raw_mv"], columns=STANDARD_LEADS)
    clean_s_df = pd.DataFrame(steps["clean_mv"], columns=STANDARD_LEADS)
    base_noise = pd.DataFrame(steps["baseline_noise_mv"], columns=STANDARD_LEADS)
    hf_noise = pd.DataFrame(steps["high_freq_noise_mv"], columns=STANDARD_LEADS)

    audit_table = pd.DataFrame(
        {
            "Média Bruta (mV)": raw_s_df.mean().round(4),
            "Média Limpa (mV)": clean_s_df.mean().round(4),
            "Std Sinal Limpo (mV)": clean_s_df.std().round(4),
            "Ruído Respiração Removido (<0.5Hz Std mV)": base_noise.std().round(4),
            "Ruído Elétrico 50/60Hz + EMG (>40Hz Std mV)": hf_noise.std().round(4),
        }
    )
    st.dataframe(audit_table, use_container_width=True)


def render_tab_table(clean_df: pd.DataFrame) -> None:
    """Aba 5: Tabela dinâmica destacando os Findings Multilabel de cada exame."""
    st.subheader(
        "Tabela de Dados Limpos com Findings Multilabel (`findings_str` e `n_findings`)"
    )

    priority_cols = [
        "subject_id",
        "study_id",
        "split",
        "ecg_status",
        "n_findings",
        "is_multilabel",
        "primary_finding",
        "findings_str",
        "sample_weight",
        "heart_rate_bpm",
        "pr_interval",
        "qrs_duration",
        "qt_interval",
        "p_wave_missing",
    ]
    other_cols = [c for c in clean_df.columns if c not in priority_cols]
    ordered_df = clean_df[
        [c for c in priority_cols if c in clean_df.columns] + other_cols
    ]

    st.dataframe(ordered_df.head(2_000), use_container_width=True, height=550)

    csv_bytes = ordered_df.head(10_000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Baixar Amostra Filtrada com Findings Multilabel (CSV)",
        data=csv_bytes,
        file_name="cardiopredict_multilabel_sample.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="CardioPredict 2.0 — BI Dashboard",
        page_icon="❤️",
        layout="wide",
    )

    st.title("❤️ CardioPredict 2.0 — Painel Analítico Multilabel de ECG (MIMIC-IV-ECG)")

    st.sidebar.header("Filtros Globais (Segmentação)")
    sample_limit = st.sidebar.selectbox(
        "Tamanho da Amostra Carregada:",
        options=[50_000, 100_000, 250_000],
        index=1,
        format_func=lambda x: f"{x:,} exames",
    )

    raw_df, clean_df = load_dashboard_data(max_rows=sample_limit)
    reports_df = load_normalized_reports_sample(raw_df)

    selected_splits = st.sidebar.multiselect(
        "Conjunto (Split por Paciente):",
        options=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    selected_status = st.sidebar.multiselect(
        "Status Diagnóstico Geral (ecg_status):",
        options=["normal", "borderline", "abnormal", "unclassified"],
        default=["normal", "borderline", "abnormal"],
    )

    # =========================================================================
    # FILTRO GLOBAL DINÂMICO DE TOP FINDINGS (TEXTO EXATO COLOCADO NA BASE)
    # Permite escolher a quantidade (Top N) de Findings mais frequentes na amostra
    # e exibe literalmente a string original gravada em `machine_measurements.csv`.
    # =========================================================================
    findings_ranking = get_top_findings_ranking(clean_df)
    max_available_findings = max(5, min(200, len(findings_ranking)))
    top_k_findings = st.sidebar.slider(
        "Quantidade de Top Findings (N):",
        min_value=5,
        max_value=max_available_findings,
        value=min(20, max_available_findings),
        step=5,
        help=(
            "Define quantos dos Findings mais frequentes (com o texto exato da base "
            "machine_measurements.csv) são listados no filtro abaixo e exibidos nos "
            "gráficos das Abas 1 e 2."
        ),
    )
    top_findings_options: list[str] = [
        str(x) for x in findings_ranking.head(top_k_findings).index.tolist()
    ]
    finding_counts_map: dict[str, int] = {
        str(k): int(v) for k, v in findings_ranking.head(top_k_findings).items()
    }
    selected_finding = st.sidebar.selectbox(
        f"Filtrar por Finding Exato da Base (Top {top_k_findings}):",
        options=["Todos os Findings", *top_findings_options],
        index=0,
        format_func=lambda opt: (
            opt
            if opt == "Todos os Findings"
            else f"{opt} ({finding_counts_map.get(opt, 0):,})"
        ),
    )

    multilabel_filter = st.sidebar.radio(
        "Multiplicidade de Diagnósticos por Exame:",
        options=[
            "Todos os Exames",
            "Diagnóstico Único (1 finding)",
            "Multilabel (>= 2 findings simultâneos)",
        ],
        index=0,
    )

    # =========================================================================
    # FILTROS FISIOLÓGICOS POR ONDA DO ECG (Onda P, Complexo QRS e Onda T/QT)
    # Justificativa Médica:
    # 1. Onda P (Despolarização Atrial): É a única onda que pode estar fisiologicamente
    #    AUSENTE em pacientes vivos (ex.: Fibrilação Atrial, Flutter ou Marcapasso).
    # 2. Complexo QRS (Despolarização Ventricular): Sempre presente em pacientes vivos;
    #    sua alteração clínica clássica é o alargamento >= 120 ms (Bloqueios de Ramo).
    # 3. Onda T / Intervalo QT (Repolarização Ventricular): Sempre presente em pacientes
    #    vivos; sua alteração crítica é o prolongamento >= 450 ms (risco de Torsades de Pointes).
    # =========================================================================
    st.sidebar.subheader("Filtros Fisiológicos das Ondas")
    p_wave_filter = st.sidebar.radio(
        "1. Onda P (Atividade Atrial):",
        options=["Todos", "Com Onda P Presente (0)", "Onda P Ausente / FA (1)"],
        index=0,
    )
    qrs_wave_filter = st.sidebar.radio(
        "2. Complexo QRS (Condução Ventricular):",
        options=[
            "Todos",
            "QRS Normal (< 120 ms)",
            "QRS Alargado / Bloqueio (>= 120 ms)",
        ],
        index=0,
    )
    qt_wave_filter = st.sidebar.radio(
        "3. Onda T / Intervalo QT (Repolarização):",
        options=[
            "Todos",
            "QT Normal (< 450 ms)",
            "QT Prolongado (>= 450 ms)",
        ],
        index=0,
    )

    mask = clean_df["split"].isin(selected_splits) & clean_df["ecg_status"].isin(
        selected_status
    )
    if selected_finding != "Todos os Findings":
        mask = mask & clean_df["findings_str"].str.contains(
            selected_finding, na=False, regex=False
        )
    if multilabel_filter == "Diagnóstico Único (1 finding)":
        mask = mask & (clean_df["n_findings"] == 1)
    elif multilabel_filter == "Multilabel (>= 2 findings simultâneos)":
        mask = mask & (clean_df["is_multilabel"] == 1)

    if p_wave_filter == "Com Onda P Presente (0)":
        mask = mask & (clean_df["p_wave_missing"] == 0)
    elif p_wave_filter == "Onda P Ausente / FA (1)":
        mask = mask & (clean_df["p_wave_missing"] == 1)

    if qrs_wave_filter == "QRS Normal (< 120 ms)":
        mask = mask & (clean_df["qrs_duration"] < 120)
    elif qrs_wave_filter == "QRS Alargado / Bloqueio (>= 120 ms)":
        mask = mask & (clean_df["qrs_duration"] >= 120)

    if qt_wave_filter == "QT Normal (< 450 ms)":
        mask = mask & (clean_df["qt_interval"] < 450)
    elif qt_wave_filter == "QT Prolongado (>= 450 ms)":
        mask = mask & (clean_df["qt_interval"] >= 450)

    filtered_df = pd.DataFrame(clean_df.loc[mask])

    render_kpi_cards(filtered_df)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 1. Box Plots (Status & Findings)",
            "🏥 2. Findings Multilabel, Incerteza & Splits",
            "📝 3. Todos os Reports (report_0..17)",
            "📈 4. Ondas ECG (12 Canais em mV)",
            "📋 5. Tabela de Dados Multilabel",
        ]
    )

    with tab1:
        render_tab_boxplots(raw_df, filtered_df, top_k_findings=top_k_findings)
    with tab2:
        render_tab_diagnostics_and_splits(
            filtered_df, reports_df, top_k_findings=top_k_findings
        )
    with tab3:
        render_tab_all_reports_explorer(reports_df)
    with tab4:
        render_tab_ecg_signals(clean_df)
    with tab5:
        render_tab_table(filtered_df)


if __name__ == "__main__":
    main()
