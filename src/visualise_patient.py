"""Módulo de compatibilidade retroativa para src/visualization/plot_patient.py."""

from visualization.plot_patient import main, plot_single_patient_exam

__all__ = [
    "main",
    "plot_single_patient_exam",
]

if __name__ == "__main__":
    main()
