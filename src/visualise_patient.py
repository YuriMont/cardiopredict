import matplotlib.pyplot as plt
import pandas as pd
import wfdb

from read_database import DATA_DIR, patients_records, subject_id

# um paciente
exam = patients_records.iloc[0]

record_path = DATA_DIR / exam["path"]

# remove a extensão caso tenha uma
if record_path.suffix in {".hea", "dat"}:
    record_path = record_path.with_suffix("")

record = wfdb.rdrecord(str(record_path))

if isinstance(record, wfdb.MultiRecord):
    record = record.multi_to_single(physical=True)

if record.p_signal is not None:
    print(f"Indentificador do exame: {exam['study_id']}")
    print(f"Identificador do paciente: {exam['subject_id']}")
    print(f"Frequencia da amostragem: {record.fs} Hz")
    print(f"Formato do sinal: {record.p_signal.shape}")
    print(f"Derivações: {record.sig_name}")
else:
    print("Erro")

measurements = pd.read_csv(DATA_DIR / "machine_measurements.csv", low_memory=False)

print(measurements.describe())

patient_ecg = measurements.loc[measurements["subject_id"] == subject_id]

print(patient_ecg[["subject_id", "study_id", "ecg_time"]].to_string(index=False))

if not patient_ecg.empty:
    exam_reports = patient_ecg.iloc[0]

    print(f"ECG: {exam_reports['study_id']}")

    for col in patient_ecg.columns:
        if col.startswith("report_"):
            value = exam_reports[col]

            if pd.notna(value) and str(value).strip():
                print(f"{col}: {value}")
else:
    print("vazio")

wfdb.plot_wfdb(
    record=record,
    title=f"ECC - Study {exam['study_id']}",
    figsize=(16, 10),
    ecg_grids="all",
)

plt.show()
