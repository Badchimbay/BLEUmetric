import psycopg2


class PostgresLogger:
    def __init__(self, host, db, user, password, port=5432):
        self.conn = psycopg2.connect(host=host, dbname=db, user=user, password=password, port=port)
        self.cursor = self.conn.cursor()

    def log(self, start_time, end_time, ref_sources, machine_translation, results, error_details):
        duration = (end_time - start_time).total_seconds()
        status = "ERROR" if error_details else "INFO"
        self.cursor.execute("""
                INSERT INTO public.bleu_results
                (status, results, ref_sources, machine_translation, error_details, start_time, end_time, duration)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (status, results, ref_sources, machine_translation, error_details, start_time.isoformat(), end_time.isoformat(), duration))
        self.conn.commit()
