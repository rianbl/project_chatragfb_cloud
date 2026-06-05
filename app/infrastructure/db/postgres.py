from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostgresSettings:
    host: str
    port: str
    database: str
    user: str
    password: str


class PostgresConnectionFactory:
    def __init__(self, settings: PostgresSettings) -> None:
        self._settings = settings

    def create_connection(self):
        import psycopg2

        return psycopg2.connect(
            host=self._settings.host,
            port=self._settings.port,
            dbname=self._settings.database,
            user=self._settings.user,
            password=self._settings.password,
        )


def build_default_postgres_factory(settings: PostgresSettings) -> PostgresConnectionFactory:
    return PostgresConnectionFactory(settings)
