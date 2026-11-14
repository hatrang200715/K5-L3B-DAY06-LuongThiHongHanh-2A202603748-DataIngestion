"""Run the Day 06 ingestion and retrieval benchmark."""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .models import DocumentPage
from .store import PgVectorStore

STARTER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = STARTER_DIR / "data"
BENCHMARK_PATH = STARTER_DIR / "benchmark" / "questions.json"
MANIFEST_PATH = DATA_DIR / "manifest.yaml"
RESULTS_DIR = STARTER_DIR / "results"
DEFAULT_CORPUS_PATH = DATA_DIR / "pg54392-images.html"
FIXED_CHUNK_SIZE = 500
FIXED_OVERLAP = 50
PARAGRAPH_CHUNK_SIZE = 500
EMBEDDING_BATCH_SIZE = 64


class _PrintedPageParser(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self, counted_pages: set[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.counted_pages = counted_pages
        self.pages: dict[str, str] = {}
        self.current_page: str | None = None
        self.current_text: list[str] = []
        self.skip_anchor = False

    def _finish_page(self) -> None:
        if self.current_page is not None:
            self.pages[self.current_page] = "".join(self.current_text)
        self.current_page = None
        self.current_text = []

    def _begin_page(self, page: str) -> None:
        self._finish_page()
        if page in self.counted_pages:
            self.current_page = page

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if (
            tag == "img"
            and attributes.get("id") == "img_images_cover.jpg"
            and "cover" in self.counted_pages
        ):
            self.pages["cover"] = " ".join((attributes.get("alt") or "").split())
            return
        marker = re.fullmatch(r"Page_([A-Za-z0-9]+)", attributes.get("id") or "")
        if tag == "a" and marker:
            self._begin_page(marker.group(1).lower())
            self.skip_anchor = True
            return
        if self.current_page is not None and tag in self._BLOCK_TAGS:
            self.current_text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.skip_anchor:
            self.skip_anchor = False
            return
        if self.current_page is not None and tag in self._BLOCK_TAGS:
            self.current_text.append("\n")

    def handle_data(self, data: str) -> None:
        if self.current_page is not None and not self.skip_anchor:
            self.current_text.append(data)

    def close(self) -> None:
        super().close()
        self._finish_page()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required file not found: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            manifest = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required corpus manifest not found: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("The pipeline requires exactly one source in the manifest")
    source = sources[0]
    required = (
        "source_id",
        "source_path_or_url",
        "arabic_page_range",
        "front_matter_pages",
        "page_character_checks",
        "counted_page_count",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"Manifest source is missing required fields: {missing}")
    return source


def _page_manifest(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    page_range = source["arabic_page_range"]
    if (
        not isinstance(page_range, list)
        or len(page_range) != 2
        or any(not isinstance(page, int) or isinstance(page, bool) for page in page_range)
        or page_range[0] > page_range[1]
    ):
        raise ValueError("Manifest arabic_page_range must contain an ordered start and end")

    source_id = source["source_id"]
    pages = {
        str(page_number): {
            "marker_id": str(page_number),
            "page_number": page_number,
            "printed_page": page_number,
            "page_id": f"{source_id}-p{page_number:03d}",
        }
        for page_number in range(page_range[0], page_range[1] + 1)
    }
    front_matter_pages = source["front_matter_pages"]
    if not isinstance(front_matter_pages, list):
        raise ValueError("Manifest front_matter_pages must be a list")
    for page in front_matter_pages:
        if not isinstance(page, dict):
            raise ValueError("Every front-matter page entry must be an object")
        marker_id = page.get("marker_id")
        page_number = page.get("page_number")
        if (
            not isinstance(marker_id, str)
            or not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or marker_id in pages
        ):
            raise ValueError("Front-matter entries require unique marker IDs and integer page numbers")
        pages[marker_id] = page

    character_checks = source["page_character_checks"]
    if not isinstance(character_checks, list):
        raise ValueError("Manifest page_character_checks must be a list")
    for check in character_checks:
        if not isinstance(check, dict) or not isinstance(check.get("printed_page"), int):
            raise ValueError("Each page character check must identify an integer printed page")
        page = pages.get(str(check["printed_page"]))
        if page is None:
            raise ValueError(
                f"Page character check references an uncounted page: {check['printed_page']}"
            )
        page["normalized_characters"] = check.get("normalized_characters")

    if len(pages) != source["counted_page_count"]:
        raise ValueError("Manifest counted_page_count does not match its page ranges")
    return pages


def _load_html_pages(
    path: Path, source: dict[str, Any]
) -> tuple[list[DocumentPage], list[dict[str, int]]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus HTML not found at {path}. Download the declared source with:\n"
            "curl -L --fail --output "
            f"{path.relative_to(STARTER_DIR)} {source['source_path_or_url']}"
        )

    page_manifest = _page_manifest(source)
    marker_ids = set(page_manifest)
    parser = _PrintedPageParser(marker_ids)
    with path.open(encoding="utf-8") as file:
        parser.feed(file.read())
    parser.close()

    missing_pages = sorted(marker_ids - parser.pages.keys())
    if missing_pages:
        raise ValueError(f"Corpus HTML is missing declared page markers: {missing_pages}")

    pages = []
    character_count_warnings = []
    for marker_id, page_info in sorted(
        page_manifest.items(), key=lambda item: item[1]["page_number"]
    ):
        page_number = page_info["page_number"]
        raw_text = parser.pages[marker_id]
        if marker_id == "cover":
            raw_text = " ".join(
                (raw_text, source["title"], source["author"])
            )
        text = " ".join(raw_text.split())
        expected_characters = page_info.get("normalized_characters")
        if isinstance(expected_characters, int) and not isinstance(expected_characters, bool):
            actual_characters = len(text)
            if actual_characters != expected_characters:
                character_count_warnings.append(
                    {
                        "page": page_number,
                        "manifest": expected_characters,
                        "extracted": actual_characters,
                    }
                )
        pages.append(
            DocumentPage(
                document_id=source["source_id"],
                source=source["source_path_or_url"],
                page=page_number,
                text=text,
                metadata={
                    "page_id": page_info["page_id"],
                    "printed_page": page_info["printed_page"],
                    "extraction_status": "extracted" if text else "no_text",
                },
            )
        )
    return pages, character_count_warnings


def _page_sections(schema: dict[str, Any]) -> dict[int, str]:
    values = schema.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError("benchmark metadata_schema.values must be a non-empty object")
    sections: dict[int, str] = {}
    for section, description in values.items():
        if not isinstance(section, str) or not isinstance(description, str):
            raise ValueError("section names and page descriptions must be strings")
        match = re.fullmatch(
            r"Printed pages? (\d+)(?:-(\d+))?", description.strip(), flags=re.IGNORECASE
        )
        if not match:
            raise ValueError(f"Cannot parse printed page range for section {section!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            raise ValueError(f"Invalid printed page range for section {section!r}")
        for page in range(start, end + 1):
            if page in sections:
                raise ValueError(f"Printed page {page} belongs to multiple sections")
            sections[page] = section
    return sections


def _validate_questions(
    questions: Any,
    metadata_schema: dict[str, Any],
    counted_pages: set[int],
    source_id: str,
) -> list[dict[str, Any]]:
    if not isinstance(questions, list) or len(questions) < 20:
        raise ValueError("Benchmark must contain at least 20 questions")
    if any(not isinstance(question, dict) for question in questions):
        raise ValueError("Every benchmark question must be an object")

    identifiers = [question.get("id") for question in questions]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("Every benchmark question must have a non-empty string id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Benchmark question IDs must be unique")

    splits = {split: sum(q.get("split") == split for q in questions) for split in ("dev", "test")}
    if splits["dev"] < 15 or splits["test"] < 5:
        raise ValueError("Benchmark requires at least 15 dev and 5 test questions")
    filter_field = metadata_schema.get("filter_field")
    filter_values = metadata_schema.get("values")
    filter_questions = 0
    for question in questions:
        if question.get("split") not in ("dev", "test"):
            raise ValueError(f"Question {question['id']} has an unsupported split")
        if (
            question.get("gold_source_id") != source_id
            or question.get("gold_page") not in counted_pages
        ):
            raise ValueError(f"Question {question['id']} gold source/page is outside the corpus")
        if not isinstance(question.get("evidence_quote"), str) or not question["evidence_quote"].strip():
            raise ValueError(f"Question {question['id']} has no evidence quote")
        filters = question.get("filter")
        if filters is None:
            continue
        if not isinstance(filters, dict) or set(filters) != {filter_field}:
            raise ValueError(
                f"Question {question['id']} filter must contain only {filter_field!r}"
            )
        value = filters[filter_field]
        if value not in filter_values:
            raise ValueError(f"Question {question['id']} has an unknown metadata filter value")
        filter_questions += 1
    if filter_questions < 3 or not any(
        question.get("split") == "test" and question.get("filter") for question in questions
    ):
        raise ValueError("Benchmark requires at least 3 filter questions, including one test")
    return questions


def _embed_in_batches(embedder: MyEmbedder, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        batch_vectors = embedder.embed_documents(batch)
        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                f"Embedding provider returned {len(batch_vectors)} vectors "
                f"for {len(batch)} chunks"
            )
        vectors.extend(batch_vectors)
    return vectors


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Ingest the manifest corpus and save A/B/C benchmark traces."""
    load_dotenv(STARTER_DIR / ".env", override=False)
    source = _read_manifest(MANIFEST_PATH)
    benchmark = _read_json(BENCHMARK_PATH)
    corpus = benchmark.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("Benchmark is missing corpus metadata")
    if corpus.get("source_id") != source["source_id"]:
        raise ValueError("Benchmark source_id does not match the corpus manifest")
    if corpus.get("source_url") != source["source_path_or_url"]:
        raise ValueError("Benchmark source_url does not match the corpus manifest")
    metadata_schema = benchmark.get("metadata_schema")
    if not isinstance(metadata_schema, dict):
        raise ValueError("Benchmark is missing metadata_schema")

    pages, character_count_warnings = _load_html_pages(
        Path(os.getenv("CORPUS_HTML_PATH", DEFAULT_CORPUS_PATH)), source
    )
    corpus_path = Path(os.getenv("CORPUS_HTML_PATH", DEFAULT_CORPUS_PATH))
    counted_pages = {
        page["page_number"] for page in _page_manifest(source).values()
    }
    if len(pages) < 30:
        raise ValueError(f"Corpus contains {len(pages)} counted pages; at least 30 are required")
    if len(pages) != source.get("counted_page_count"):
        raise ValueError("Extracted page count does not match the manifest")
    questions = _validate_questions(
        benchmark.get("questions"), metadata_schema, counted_pages, source["source_id"]
    )
    sections = _page_sections(metadata_schema)
    if not counted_pages.issubset(sections):
        raise ValueError("Metadata schema does not assign a section to every counted page")
    for page in pages:
        page.metadata["section"] = sections[page.page]

    model_name = os.getenv("EMBEDDING_MODEL", "").strip()
    if not model_name:
        raise ValueError("Set EMBEDDING_MODEL in starter/.env before running the pipeline")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("Set DATABASE_URL in starter/.env before running the pipeline")

    embedder = MyEmbedder(model_name)
    query_vectors = {
        question["id"]: embedder.embed_query(question["question"])
        for question in questions
    }
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    chunks_by_strategy: dict[str, list[Any]] = {
        "A": [
            chunk
            for page in pages
            for chunk in fixed_size(page, FIXED_CHUNK_SIZE, FIXED_OVERLAP)
        ],
        "B": [
            chunk
            for page in pages
            for chunk in by_paragraph(page, PARAGRAPH_CHUNK_SIZE)
        ],
    }
    chunks_by_strategy["C"] = chunks_by_strategy["B"]

    vectors_by_strategy: dict[str, list[list[float]]] = {}
    for strategy in ("A", "B", "C"):
        if strategy == "C":
            vectors = vectors_by_strategy["B"]
        else:
            chunks = chunks_by_strategy[strategy]
            if not chunks:
                raise ValueError(f"Strategy {strategy} produced no chunks")
            vectors = _embed_in_batches(embedder, [chunk.text for chunk in chunks])
            if any(len(vector) != embedder.dimensions for vector in vectors):
                raise RuntimeError(f"Strategy {strategy} contains an invalid vector dimension")
            vectors_by_strategy[strategy] = vectors
        store.upsert(strategy, chunks_by_strategy[strategy], vectors)

    traces = []
    for strategy in ("A", "B", "C"):
        for question in questions:
            filters = question.get("filter") if strategy == "C" else None
            hits = store.search(
                strategy,
                query_vectors[question["id"]],
                top_k=3,
                filters=filters,
            )
            if len(hits) > 3:
                raise RuntimeError("Vector store returned more than the requested top-3 hits")
            trace = evaluate_query(
                {**question, "strategy": strategy},
                hits,
            )
            traces.append(trace)

    summary = summarize(traces)
    test_questions = {question["id"] for question in questions if question["split"] == "test"}
    grounding_samples = [
        {
            "strategy": trace["strategy"],
            "question_id": trace["question_id"],
            "question": trace["question"],
            "hits": trace["hits"],
            "automatic_evidence_hit_at_3": trace["evidence_hit_at_3"],
            "human_grounding_score": None,
            "human_notes": "",
        }
        for trace in traces
        if trace["question_id"] in test_questions
    ]
    coherence_samples = []
    first_pages = sorted(counted_pages)[:10]
    for strategy, chunks in chunks_by_strategy.items():
        for page_number in first_pages:
            page_chunks = [chunk for chunk in chunks if chunk.page == page_number]
            if page_chunks:
                chunk = page_chunks[0]
                coherence_samples.append(
                    {
                        "strategy": strategy,
                        "page": page_number,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "human_coherence_score": None,
                        "human_notes": "",
                    }
                )

    configuration = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_source_id": source["source_id"],
        "corpus_source": source["source_path_or_url"],
        "corpus_sha256": _sha256_file(corpus_path),
        "benchmark_sha256": _sha256_file(BENCHMARK_PATH),
        "counted_pages": sorted(counted_pages),
        "page_count": len(pages),
        "character_count_warnings": character_count_warnings,
        "model_name": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "page_inventory": [
            {
                "page_id": page.metadata["page_id"],
                "page_number": page.page,
                "printed_page": page.metadata["printed_page"],
                "normalized_characters": len(page.text),
                "extraction_status": page.metadata["extraction_status"],
            }
            for page in pages
        ],
        "empty_page_count": sum(not page.text for page in pages),
        "strategies": {
            "A": {"chunking": "fixed_size", "size": FIXED_CHUNK_SIZE, "overlap": FIXED_OVERLAP, "filtering": False},
            "B": {"chunking": "by_paragraph", "max_size": PARAGRAPH_CHUNK_SIZE, "filtering": False},
            "C": {"chunking": "by_paragraph", "max_size": PARAGRAPH_CHUNK_SIZE, "filtering": metadata_schema["filter_field"]},
        },
        "question_count": len(questions),
        "test_split_status": (
            "Exploratory rerun: the test split was inspected in earlier runs; "
            "do not interpret these results as an independent blind test."
        ),
        "comparison_controls": {
            "same_corpus_for_all_strategies": True,
            "same_model_for_all_strategies": True,
            "same_benchmark_for_all_strategies": True,
            "query_embeddings_reused_across_strategies": True,
        },
        "chunk_counts": {
            strategy: len(chunks) for strategy, chunks in chunks_by_strategy.items()
        },
    }
    _write_json(RESULTS_DIR / "run_metadata.json", configuration)
    _write_json(RESULTS_DIR / "traces.json", traces)
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_json(
        RESULTS_DIR / "manual_review.json",
        {
            "notice": "Human labels are intentionally blank; complete them after reviewing the evidence.",
            "grounding_samples": grounding_samples,
            "coherence_samples": coherence_samples,
        },
    )
    if character_count_warnings:
        pages_with_count_differences = ", ".join(
            str(item["page"]) for item in character_count_warnings
        )
        print(
            "Warning: extracted character counts differ from the manifest on "
            f"printed pages {pages_with_count_differences}; see run_metadata.json."
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved run metadata, traces, summary, and manual review samples in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
