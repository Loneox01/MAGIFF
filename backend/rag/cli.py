import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_CONTEXT_PLANNER_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ESCALATION_MODEL,
    DEFAULT_INDEX_PATH,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REPORT_STORE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_MODEL,
)
from .planning.context_planner import ContextPlanResult, ContextPlanner
from .documents import load_reports
from .evaluation import CASES, evaluate_retrieval
from .planning.planner import QueryPlanResult, QueryPlanner
from .planning.enrichment import StructuredLookupExecutor
from .planning.resolver import (
    EntityResolver,
    ResolutionResult,
    ResolutionValidationError,
    validate_resolution_bounds,
)
from .planning.router import (
    MAX_ESCALATION_DATABASE_CANDIDATES,
    EscalationEvent,
    EscalationRouter,
)
from .retrieval.executor import QueryPlanExecutor
from .retrieval.factory import create_report_store
from .retrieval.reranker import (
    MAX_RERANK_CANDIDATES,
    ReportReranker,
    RerankResult,
)
from .retrieval.store import LocalRAGStore, SearchHit


def _add_index_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Local planner/cache path (and SQLite report index when selected)",
    )


def _add_store(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store",
        choices=("local", "supabase"),
        default=DEFAULT_REPORT_STORE,
        help="Report retrieval store (defaults to RAG_REPORT_STORE or supabase)",
    )


def _add_planner_options(
    parser: argparse.ArgumentParser,
    *,
    include_enable_flag: bool,
) -> None:
    if include_enable_flag:
        parser.add_argument(
            "--use-planner",
            action="store_true",
            help="Run staged direct and contextual planning before retrieval",
        )
        parser.add_argument(
            "--show-plan",
            action="store_true",
            help="Print the structured query plan before search results",
        )
        parser.add_argument(
            "--show-resolution",
            action="store_true",
            help="Print database-resolved players, teams, and unresolved constraints",
        )
    parser.add_argument(
        "--planner-model",
        default=DEFAULT_PLANNER_MODEL,
        help="Model used for direct target and constraint planning",
    )
    parser.add_argument(
        "--refresh-plan",
        action="store_true",
        help="Ignore a cached plan and call the planner again",
    )
    parser.add_argument(
        "--context-model",
        default=DEFAULT_CONTEXT_PLANNER_MODEL,
        help="Model used only for indirect contextual evidence planning",
    )
    parser.add_argument(
        "--refresh-context",
        action="store_true",
        help="Ignore a cached contextual plan and call its planner again",
    )
    parser.add_argument(
        "--no-context-planner",
        action="store_true",
        help="Run only direct report planning (useful for isolated tests)",
    )
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Keep the Luna plan even when the escalation policy triggers",
    )
    parser.add_argument(
        "--escalation-model",
        default=DEFAULT_ESCALATION_MODEL,
        help="Higher-capability model used by narrow fallback routes",
    )
    parser.add_argument(
        "--refresh-escalation",
        action="store_true",
        help="Ignore a cached escalation decision",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and test report retrieval stores."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a report snapshot")
    index_parser.add_argument(
        "--snapshot",
        help="Snapshot folder/date; defaults to the latest snapshot",
    )
    index_parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Generate missing OpenAI embeddings in addition to the keyword index",
    )
    index_parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    _add_index_path(index_parser)

    search_parser = subparsers.add_parser("search", help="Search indexed reports")
    search_parser.add_argument("query")
    search_parser.add_argument(
        "--mode",
        choices=("keyword", "vector", "hybrid"),
        default="keyword",
    )
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    search_parser.add_argument("--player")
    search_parser.add_argument("--team")
    search_parser.add_argument("--source")
    search_parser.add_argument("--season", type=int)
    search_parser.add_argument("--document-type")
    search_parser.add_argument("--storyline")
    search_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use one batched model call to rerank a larger candidate pool",
    )
    search_parser.add_argument(
        "--show-rerank",
        action="store_true",
        help="Print per-document reranker judgments (also enables reranking)",
    )
    search_parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
    )
    search_parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
        help="Number of retrieved reports sent to the batched reranker",
    )
    search_parser.add_argument(
        "--refresh-rerank",
        action="store_true",
        help="Ignore a cached reranker response",
    )
    _add_planner_options(search_parser, include_enable_flag=True)
    _add_index_path(search_parser)
    _add_store(search_parser)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Generate and inspect a structured retrieval plan",
    )
    plan_parser.add_argument("query")
    _add_planner_options(plan_parser, include_enable_flag=False)
    _add_index_path(plan_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run a small retrieval regression set",
    )
    evaluate_parser.add_argument(
        "--mode",
        choices=("keyword", "vector", "hybrid"),
        default="keyword",
    )
    evaluate_parser.add_argument("--top-k", type=int, default=3)
    _add_index_path(evaluate_parser)

    status_parser = subparsers.add_parser("status", help="Show report-store status")
    _add_index_path(status_parser)
    _add_store(status_parser)

    return parser


def _print_hit(index: int, hit: SearchHit) -> None:
    document = hit.document
    ranks = []
    if hit.keyword_rank is not None:
        ranks.append(f"keyword #{hit.keyword_rank}")
    if hit.vector_rank is not None:
        ranks.append(f"vector #{hit.vector_rank}")
    rank_text = f"; {', '.join(ranks)}" if ranks else ""
    scope_text = (
        f"; scopes {', '.join(hit.retrieval_scopes)}"
        if hit.retrieval_scopes
        else ""
    )

    print(f"\n{index}. {document.title}")
    print(
        f"   {document.source} | {document.published_at} | "
        f"{hit.method} {hit.score:.4f}{rank_text}{scope_text}"
    )
    print(f"   Players: {', '.join(document.players)}")
    print(f"   {document.url}")
    snippet = document.snippet
    print(f"   {snippet[:320]}{'...' if len(snippet) > 320 else ''}")


def _print_plan(result: QueryPlanResult, *, include_json: bool) -> None:
    cache_status = "cache hit" if result.cached else "cache miss"
    print(
        f"Planner: {result.model} | {cache_status} | "
        f"input tokens {result.input_tokens} "
        f"({result.cached_input_tokens} cached) | "
        f"output tokens {result.output_tokens} | "
        f"attempts {result.attempts} | "
        f"correction retry {'yes' if result.retried else 'no'}"
    )
    if result.retry_reason:
        print(f"  Correction reason: {result.retry_reason.splitlines()[0]}")
    if include_json:
        print(result.plan.model_dump_json(indent=2))


def _print_context_plan(
    result: ContextPlanResult,
    *,
    include_json: bool,
) -> None:
    cache_status = "cache hit" if result.cached else "cache miss"
    print(
        f"Context planner: {result.model} | {cache_status} | "
        f"needed {'yes' if result.context_plan.context_needed else 'no'} | "
        f"branches {len(result.context_plan.context_requests)} | "
        f"attempts {result.attempts} | "
        f"correction retry {'yes' if result.retried else 'no'} | "
        f"input tokens {result.input_tokens} "
        f"({result.cached_input_tokens} cached) | "
        f"output tokens {result.output_tokens}"
    )
    print(f"  Reason: {result.context_plan.rationale}")
    if result.retry_reason:
        print(f"  Correction reason: {result.retry_reason.splitlines()[0]}")
    if include_json:
        print(result.context_plan.model_dump_json(indent=2))


def _print_resolution(result: ResolutionResult) -> None:
    print("Resolution:")
    print(result.model_dump_json(indent=2))


def _print_escalation(event: EscalationEvent) -> None:
    if not event.triggered:
        print("Escalation: not triggered")
    else:
        request_status = "cache hit" if event.cache_hit else "API call"
        impact = "yes" if event.impactful else "no"
        reasons = ", ".join(event.reasons)
        print(
            f"Escalation: {event.model} | {request_status} | {reasons} | "
            f"input tokens {event.input_tokens} "
            f"({event.cached_input_tokens} cached) | "
            f"output tokens {event.output_tokens} | "
            f"estimated cost ${event.estimated_cost_usd:.6f} | "
            f"identity impact {impact}"
        )

    for signal in event.signals:
        candidate = ", ".join(signal.luna_candidates) or "(none)"
        database_matches = ", ".join(
            " ".join(
                part
                for part in (
                    match.display_name,
                    match.team,
                    match.position,
                    f"#{match.jersey_number}" if match.jersey_number else None,
                    match.roster_status,
                    (
                        f"rookie {match.rookie_season}"
                        if match.rookie_season is not None
                        else None
                    ),
                    f"[{match.player_id}]",
                )
                if part
            )
            for match in signal.database_matches
        )
        reasons = ", ".join(signal.reasons) or "accepted"
        print(
            f"  Luna identity: {signal.reference_text!r} -> {candidate} | "
            f"basis {signal.resolution_basis} | "
            f"confidence {signal.identity_confidence:.2f} | "
            f"database {signal.database_status} | {reasons}"
        )
        if signal.database_matches and (
            len(signal.database_matches) > 1
            or signal.database_match_method != "direct"
        ):
            method = signal.database_match_method.replace("_", " ")
            print(f"    Database candidates ({method}): {database_matches}")
        elif signal.database_matches_omitted:
            print(
                "    Database candidates omitted: "
                f"{signal.database_match_count} fuzzy matches exceeds limit "
                f"{MAX_ESCALATION_DATABASE_CANDIDATES}"
            )
    for decision in event.decisions:
        destination = decision.canonical_name or decision.status
        player_id = f" [{decision.player_id}]" if decision.player_id else ""
        grounded = "grounded" if decision.grounded else "not grounded"
        print(
            f"  {decision.reference_text!r} -> {destination}{player_id} "
            f"({decision.status}, {grounded})"
        )
        if decision.note:
            print(f"  Note: {decision.note}")
    if event.error:
        print(f"  Fallback failed; retaining Luna plan: {event.error}")


def _print_rerank(result: RerankResult, *, include_judgments: bool) -> None:
    request_status = "cache hit" if result.cached else (
        "API call" if result.api_called else "no call"
    )
    cost = (
        f"${result.estimated_cost_usd:.6f}"
        if result.estimated_cost_usd is not None
        else "not configured"
    )
    changed = "yes" if result.ranking_changed else "no"
    print(
        f"Reranker: {result.model} | {request_status} | "
        f"candidates {result.candidate_count} -> {len(result.hits)} | "
        f"input tokens {result.input_tokens} "
        f"({result.cached_input_tokens} cached) | "
        f"output tokens {result.output_tokens} | "
        f"estimated cost {cost} | latency {result.latency_ms}ms | "
        f"ranking changed {changed}"
    )
    print(
        f"  Evidence: {result.evidence_sufficiency} | "
        f"{result.sufficiency_reason}"
    )
    if result.error:
        print(f"  Reranking failed; original order retained: {result.error}")
    if include_judgments:
        for item in result.ranked_candidates:
            disposition = (
                f"selected #{item.final_rank}"
                if item.final_rank is not None
                else "excluded"
            )
            print(
                f"  {disposition} (retrieval #{item.original_rank}) "
                f"{item.hit.document.id} | "
                f"relevance {item.judgment.relevance_score}/100 | "
                f"{item.judgment.relationship} | "
                f"{item.judgment.temporal_role} | "
                f"condition {item.judgment.condition_alignment} | "
                f"{item.judgment.reason}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {"search", "status"}:
        store = create_report_store(args.store, index_path=args.index_path)
    else:
        store = LocalRAGStore(index_path=args.index_path)

    try:
        if args.command == "index":
            documents = load_reports(snapshot=args.snapshot)
            result = store.build_index(
                documents,
                with_embeddings=args.with_embeddings,
                embedding_model=args.embedding_model,
            )
            print(f"Indexed {result.document_count} reports at {result.index_path}")
            print(f"Embedded reports: {result.embedded_count}")
            print(f"New embeddings generated: {result.generated_embedding_count}")

        elif args.command == "search":
            if args.show_rerank:
                args.rerank = True
            if args.rerank:
                args.use_planner = True
                if args.rerank_candidates < args.limit:
                    raise ValueError(
                        "--rerank-candidates must be at least --limit"
                    )
                if args.rerank_candidates > MAX_RERANK_CANDIDATES:
                    raise ValueError(
                        "--rerank-candidates cannot exceed "
                        f"{MAX_RERANK_CANDIDATES}"
                    )
            if args.show_plan and not args.use_planner:
                raise ValueError("--show-plan requires --use-planner")
            if args.show_resolution and not args.use_planner:
                raise ValueError("--show-resolution requires --use-planner")

            plan_result = None
            keyword_query = None
            vector_query = None
            if args.use_planner:
                planner = QueryPlanner(
                    index_path=args.index_path,
                    model=args.planner_model,
                )
                plan_result = planner.plan(
                    args.query,
                    use_cache=not args.refresh_plan,
                )
                keyword_query = plan_result.plan.keyword_query
                vector_query = plan_result.plan.semantic_query
                _print_plan(plan_result, include_json=args.show_plan)

            filters = {
                "player": args.player,
                "team": args.team,
                "source": args.source,
                "season": args.season,
                "document_type": args.document_type,
                "storyline": args.storyline,
            }
            if plan_result is not None:
                resolver = EntityResolver()
                resolution = resolver.resolve(plan_result.plan)
                validate_resolution_bounds(resolution)
                active_plan = plan_result.plan
                if not args.no_escalation:
                    routing = EscalationRouter(
                        index_path=args.index_path,
                        model=args.escalation_model,
                    ).route(
                        args.query,
                        active_plan,
                        resolution,
                        resolver=resolver,
                        use_cache=not args.refresh_escalation,
                    )
                    active_plan = routing.plan
                    resolution = routing.resolution
                    _print_escalation(routing.event)
                    if args.show_plan and routing.event.plan_changed:
                        print("Identity-adjusted direct plan:")
                        print(active_plan.model_dump_json(indent=2))
                if not args.no_context_planner:
                    context_result = ContextPlanner(
                        index_path=args.index_path,
                        model=args.context_model,
                    ).expand(
                        args.query,
                        active_plan,
                        resolution,
                        use_cache=not args.refresh_context,
                    )
                    active_plan = context_result.plan
                    resolution = resolver.resolve(active_plan)
                    validate_resolution_bounds(resolution)
                    _print_context_plan(
                        context_result,
                        include_json=args.show_plan,
                    )
                    if args.show_plan:
                        print("Combined execution plan:")
                        print(active_plan.model_dump_json(indent=2))
                enrichment = StructuredLookupExecutor().execute(
                    active_plan,
                    resolution,
                )
                execution = QueryPlanExecutor(store).execute(
                    args.query,
                    active_plan,
                    resolution,
                    mode=args.mode,
                    limit=(
                        args.rerank_candidates if args.rerank else args.limit
                    ),
                    embedding_model=args.embedding_model,
                    manual_filters=filters,
                    enrichment=enrichment,
                )
                hits = execution.hits
                if args.show_resolution:
                    _print_resolution(resolution)
                    lookup_count = sum(
                        len(group.lookups)
                        for group in [
                            *enrichment.targets,
                            *enrichment.contexts,
                        ]
                    )
                    print(
                        f"Executor: {execution.strategy} | "
                        f"branches {execution.branch_candidates} | "
                        f"lookups {lookup_count} | "
                        "new document-player links "
                        f"{execution.linked_document_entities}"
                    )
                    for group in [*enrichment.targets, *enrichment.contexts]:
                        for lookup in group.lookups:
                            print(
                                "  Structured lookup: "
                                f"{lookup.lookup_id} | {lookup.operation} | "
                                f"{lookup.purpose} | {lookup.status} | "
                                f"entities {len(lookup.entities)}"
                            )
                            if lookup.fallback_used:
                                print(
                                    "    Fallback: "
                                    f"{lookup.fallback_reason}"
                                )
                            if lookup.error:
                                print(f"    Error: {lookup.error}")
                if args.rerank:
                    rerank_result = ReportReranker(
                        index_path=args.index_path,
                        model=args.rerank_model,
                    ).rerank(
                        args.query,
                        active_plan,
                        resolution,
                        hits,
                        limit=args.limit,
                        use_cache=not args.refresh_rerank,
                        enrichment=enrichment,
                    )
                    hits = rerank_result.hits
                    _print_rerank(
                        rerank_result,
                        include_judgments=args.show_rerank,
                    )
            else:
                if args.mode != "keyword":
                    filters["embedding_model"] = args.embedding_model
                hits = store.search(
                    args.query,
                    mode=args.mode,
                    limit=args.limit,
                    keyword_query=keyword_query,
                    vector_query=vector_query,
                    **filters,
                )
            if not hits:
                print("No matching reports found.")
            for index, hit in enumerate(hits, start=1):
                _print_hit(index, hit)

        elif args.command == "plan":
            planner = QueryPlanner(
                index_path=args.index_path,
                model=args.planner_model,
            )
            result = planner.plan(
                args.query,
                use_cache=not args.refresh_plan,
            )
            _print_plan(result, include_json=True)
            resolver = EntityResolver()
            active_plan = result.plan
            resolution = resolver.resolve(active_plan)
            validate_resolution_bounds(resolution)
            if not args.no_escalation:
                routing = EscalationRouter(
                    index_path=args.index_path,
                    model=args.escalation_model,
                ).route(
                    args.query,
                    active_plan,
                    resolution,
                    resolver=resolver,
                    use_cache=not args.refresh_escalation,
                )
                active_plan = routing.plan
                resolution = routing.resolution
                _print_escalation(routing.event)
                if routing.event.plan_changed:
                    print("Identity-adjusted direct plan:")
                    print(active_plan.model_dump_json(indent=2))
            if not args.no_context_planner:
                context_result = ContextPlanner(
                    index_path=args.index_path,
                    model=args.context_model,
                ).expand(
                    args.query,
                    active_plan,
                    resolution,
                    use_cache=not args.refresh_context,
                )
                _print_context_plan(context_result, include_json=True)
                print("Combined execution plan:")
                print(context_result.plan.model_dump_json(indent=2))

        elif args.command == "evaluate":
            passes, results = evaluate_retrieval(
                store,
                mode=args.mode,
                top_k=args.top_k,
            )
            for case, ids, passed in results:
                marker = "PASS" if passed else "FAIL"
                print(f"{marker}: {case.name}")
                print(f"  Query: {case.query}")
                print(f"  Retrieved: {', '.join(ids) or '(none)'}")
            print(f"\n{passes}/{len(CASES)} cases passed at top {args.top_k}.")
            if passes != len(CASES):
                raise SystemExit(1)

        elif args.command == "status":
            status = store.status()
            status["escalation_router"] = EscalationRouter(
                index_path=args.index_path
            ).stats()
            status["reranker"] = ReportReranker(
                index_path=args.index_path
            ).stats()
            print(json.dumps(status, indent=2))

    except ResolutionValidationError as error:
        parser.exit(
            1,
            "Retrieval stopped: plan_validation_failed\n"
            f"Code: {error.code}\n"
            f"Reason: {error}\n",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
