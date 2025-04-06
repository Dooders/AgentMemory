#!/usr/bin/env python
"""Utility to check database content for memory debugging."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("memory_debug")

def check_database():
    """Check the content of the memory database and log results."""
    try:
        # Find and connect to the database file (assuming it's in the current directory or parent)
        db_file = Path("simulation.db")
        if not db_file.exists():
            logger.error(f"Database file not found: {db_file}")
            return
            
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Database tables: {[t[0] for t in tables]}")
        
        # Check memory tables
        memory_tables = ["short_term_memory", "intermediate_memory", "long_term_memory"]
        for table in memory_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"  {table}: {count} records")
                
                # Sample data
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                    columns = [desc[0] for desc in cursor.description]
                    logger.info(f"  {table} columns: {columns}")
            except sqlite3.OperationalError as e:
                logger.warning(f"  Cannot query table {table}: {e}")
        
        conn.close()
        logger.info("Database check completed")
        
    except Exception as e:
        logger.error(f"Error checking database: {e}") 