from __future__ import annotations

from dataclasses import dataclass
from logging import Logger

from application.use_cases import ChatService, ContextService, HealthService, QueryService, StartupService
from domain.models import AppLimits
from infrastructure.adapters import (
    DefaultChatAdapter,
    DefaultDatabaseHealthAdapter,
    DefaultIngestionAdapter,
    DefaultRetrievalAdapter,
)


@dataclass(frozen=True)
class ServiceContainer:
    limits: AppLimits
    context_service: ContextService
    query_service: QueryService
    chat_service: ChatService
    health_service: HealthService
    startup_service: StartupService


def build_container(*, limits: AppLimits, logger: Logger) -> ServiceContainer:
    ingestion_adapter = DefaultIngestionAdapter()
    retrieval_adapter = DefaultRetrievalAdapter()
    chat_adapter = DefaultChatAdapter()
    db_health_adapter = DefaultDatabaseHealthAdapter()

    context_service = ContextService(
        ingestion=ingestion_adapter,
        retrieval=retrieval_adapter,
        limits=limits,
        logger=logger,
    )
    query_service = QueryService(retrieval=retrieval_adapter, default_top_k=limits.retrieval_top_k)
    chat_service = ChatService(chat=chat_adapter)
    health_service = HealthService(
        db_health=db_health_adapter,
        retrieval=retrieval_adapter,
        chat=chat_adapter,
    )
    startup_service = StartupService(
        db_health=db_health_adapter,
        ingestion=ingestion_adapter,
        retrieval=retrieval_adapter,
        chat=chat_adapter,
        context=context_service,
        embedding_model_id=limits.embedding_model_id,
        logger=logger,
    )

    return ServiceContainer(
        limits=limits,
        context_service=context_service,
        query_service=query_service,
        chat_service=chat_service,
        health_service=health_service,
        startup_service=startup_service,
    )

