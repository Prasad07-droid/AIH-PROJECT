"""Small SQLite storage layer for patients and wound visits."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from woundtrack.config import DATABASE_PATH


def _connect(db_path: Path | str = DATABASE_PATH) -> sqlite3.Connection:
    """Open a foreign-key-enabled SQLite connection with dictionary-like rows."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _connection(db_path: Path | str = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a connection and close it after committing or rolling back."""
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_db(db_path: Path | str = DATABASE_PATH) -> None:
    """Create the patient/visit tables and their basic indexes if missing."""
    schema = """
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER CHECK (age IS NULL OR age >= 0),
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        date TEXT NOT NULL,
        image_path TEXT NOT NULL,
        mask_path TEXT NOT NULL,
        area_cm2 REAL,
        area_pixels INTEGER NOT NULL DEFAULT 0,
        perimeter REAL,
        perimeter_pixels REAL,
        perimeter_unit TEXT NOT NULL DEFAULT 'px',
        calibrated INTEGER NOT NULL DEFAULT 0 CHECK (calibrated IN (0, 1)),
        calibration_method TEXT NOT NULL DEFAULT 'uncalibrated / relative',
        change_from_baseline_pct REAL,
        change_from_previous_pct REAL,
        reduction_from_baseline_pct REAL,
        healing_rate_cm2_per_day REAL,
        status_label TEXT NOT NULL DEFAULT 'Stagnant',
        granulation_pct REAL NOT NULL DEFAULT 0,
        slough_pct REAL NOT NULL DEFAULT 0,
        necrotic_pct REAL NOT NULL DEFAULT 0,
        other_tissue_pct REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_visits_patient_date ON visits(patient_id, date, id);
    """
    with _connection(db_path) as connection:
        connection.executescript(schema)


def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert one optional SQLite row to a regular dictionary."""
    return dict(row) if row is not None else None


def add_patient(
    name: str,
    age: int | None = None,
    notes: str = "",
    db_path: Path | str = DATABASE_PATH,
) -> int:
    """Insert a patient and return the new patient id."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("patient name cannot be empty")
    if age is not None and age < 0:
        raise ValueError("patient age cannot be negative")
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO patients (name, age, notes) VALUES (?, ?, ?)",
            (clean_name, age, notes.strip()),
        )
        return int(cursor.lastrowid)


def list_patients(db_path: Path | str = DATABASE_PATH) -> list[dict[str, Any]]:
    """Return all patients ordered by name and id."""
    with _connection(db_path) as connection:
        rows = connection.execute("SELECT * FROM patients ORDER BY name, id").fetchall()
    return [dict(row) for row in rows]


def get_patient(patient_id: int, db_path: Path | str = DATABASE_PATH) -> dict[str, Any] | None:
    """Return one patient or ``None`` if the id does not exist."""
    with _connection(db_path) as connection:
        row = connection.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    return _as_dict(row)


def update_patient(
    patient_id: int,
    *,
    name: str,
    age: int | None = None,
    notes: str = "",
    db_path: Path | str = DATABASE_PATH,
) -> bool:
    """Update a patient's name, age, and notes; return whether a row changed."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("patient name cannot be empty")
    if age is not None and age < 0:
        raise ValueError("patient age cannot be negative")
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE patients SET name = ?, age = ?, notes = ? WHERE id = ?",
            (clean_name, age, notes.strip(), patient_id),
        )
        return cursor.rowcount > 0


def delete_patient(patient_id: int, db_path: Path | str = DATABASE_PATH) -> bool:
    """Delete a patient; related visits cascade, stored image files are retained."""
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        return cursor.rowcount > 0


def add_visit(
    patient_id: int,
    *,
    visit_date: str,
    image_path: str,
    mask_path: str,
    area_cm2: float | None,
    area_pixels: int,
    perimeter: float | None,
    perimeter_pixels: float,
    perimeter_unit: str,
    calibrated: bool,
    calibration_method: str,
    change_from_baseline_pct: float | None = None,
    change_from_previous_pct: float | None = None,
    reduction_from_baseline_pct: float | None = None,
    healing_rate_cm2_per_day: float | None = None,
    status_label: str = "Stagnant",
    granulation_pct: float = 0.0,
    slough_pct: float = 0.0,
    necrotic_pct: float = 0.0,
    other_tissue_pct: float = 0.0,
    db_path: Path | str = DATABASE_PATH,
) -> int:
    """Insert one measured visit and return its new visit id."""
    if area_pixels < 0 or perimeter_pixels < 0:
        raise ValueError("pixel area and perimeter cannot be negative")
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """INSERT INTO visits (
                patient_id, date, image_path, mask_path, area_cm2, area_pixels,
                perimeter, perimeter_pixels, perimeter_unit, calibrated,
                calibration_method, change_from_baseline_pct,
                change_from_previous_pct, reduction_from_baseline_pct,
                healing_rate_cm2_per_day, status_label, granulation_pct,
                slough_pct, necrotic_pct, other_tissue_pct
            ) VALUES (
                :patient_id, :visit_date, :image_path, :mask_path, :area_cm2,
                :area_pixels, :perimeter, :perimeter_pixels, :perimeter_unit,
                :calibrated, :calibration_method, :change_from_baseline_pct,
                :change_from_previous_pct, :reduction_from_baseline_pct,
                :healing_rate_cm2_per_day, :status_label, :granulation_pct,
                :slough_pct, :necrotic_pct, :other_tissue_pct
            )""",
            {
                "patient_id": patient_id,
                "visit_date": visit_date,
                "image_path": image_path,
                "mask_path": mask_path,
                "area_cm2": area_cm2,
                "area_pixels": area_pixels,
                "perimeter": perimeter,
                "perimeter_pixels": perimeter_pixels,
                "perimeter_unit": perimeter_unit,
                "calibrated": int(calibrated),
                "calibration_method": calibration_method,
                "change_from_baseline_pct": change_from_baseline_pct,
                "change_from_previous_pct": change_from_previous_pct,
                "reduction_from_baseline_pct": reduction_from_baseline_pct,
                "healing_rate_cm2_per_day": healing_rate_cm2_per_day,
                "status_label": status_label,
                "granulation_pct": granulation_pct,
                "slough_pct": slough_pct,
                "necrotic_pct": necrotic_pct,
                "other_tissue_pct": other_tissue_pct,
            },
        )
        return int(cursor.lastrowid)


def list_visits(
    patient_id: int,
    db_path: Path | str = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return all visits for a patient ordered chronologically."""
    with _connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM visits WHERE patient_id = ? ORDER BY date, id",
            (patient_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_visit(visit_id: int, db_path: Path | str = DATABASE_PATH) -> dict[str, Any] | None:
    """Return one visit or ``None`` if the id does not exist."""
    with _connection(db_path) as connection:
        row = connection.execute("SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
    return _as_dict(row)


def update_visit(
    visit_id: int,
    *,
    visit_date: str,
    area_cm2: float | None,
    perimeter: float | None,
    calibrated: bool,
    granulation_pct: float,
    slough_pct: float,
    necrotic_pct: float,
    db_path: Path | str = DATABASE_PATH,
) -> bool:
    """Update visit date and manually corrected quantitative metrics."""
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """UPDATE visits SET date = ?, area_cm2 = ?, perimeter = ?, calibrated = ?,
               granulation_pct = ?, slough_pct = ?, necrotic_pct = ? WHERE id = ?""",
            (
                visit_date,
                area_cm2,
                perimeter,
                int(calibrated),
                granulation_pct,
                slough_pct,
                necrotic_pct,
                visit_id,
            ),
        )
        return cursor.rowcount > 0


def delete_visit(visit_id: int, db_path: Path | str = DATABASE_PATH) -> bool:
    """Delete one visit record without removing its image files."""
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM visits WHERE id = ?", (visit_id,))
        return cursor.rowcount > 0
