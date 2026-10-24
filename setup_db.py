"""
setup_db.py – One-command database initialisation.

Usage:
    python setup_db.py [--csv path/to/file.csv] [--db path/to/file.db]

This script:
  1. Creates (or re-creates) the SQLite database schema.
  2. Ingests sales_data.csv using the validated ingestion pipeline.
  3. Prints an ingestion report including accepted, skipped, and rejected counts.
  4. Is idempotent: re-running will not duplicate existing records.

Run this before starting the application for the first time, or after
replacing the CSV file with updated data.
"""
import argparse
import sys
import os

# Ensure the project root is on the path when called from any directory
sys.path.insert(0, os.path.dirname(__file__))

from database import ingest_csv, database_status

def main():
    parser = argparse.ArgumentParser(description="Initialise the BI Assistant database.")
    parser.add_argument("--csv", default="sales_data.csv", help="Path to the CSV file.")
    parser.add_argument("--db",  default="sales.db",       help="Path to the SQLite database.")
    args = parser.parse_args()

    print(f"Importing '{args.csv}' -> '{args.db}'...")
    try:
        report = ingest_csv(args.csv, args.db)
        print(f"\n[OK] {report.summary()}\n")
        if report.warnings:
            print("Warnings / Notes:")
            for w in report.warnings:
                print(f"  {w}")
    except FileNotFoundError as exc:
        print(f"\n[ERROR] File not found: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    status = database_status(args.db)
    print(f"\nDatabase status: {status}")

if __name__ == "__main__":
    main()
