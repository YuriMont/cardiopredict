"""Módulo de compatibilidade retroativa para src/data/database.py."""

from data.database import (
    DEFAULT_DB_PATH,
    create_schema,
    get_connection,
    main,
    migrate_all,
    migrate_mimic,
    migrate_ptbxl,
    query_summary,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "create_schema",
    "get_connection",
    "main",
    "migrate_all",
    "migrate_mimic",
    "migrate_ptbxl",
    "query_summary",
]

if __name__ == "__main__":
    main()
