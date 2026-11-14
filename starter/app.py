"""Streamlit demo app for the Day 06 retrieval chatbot."""
import json
import os
from pathlib import Path
from typing import Any

import psycopg
import streamlit as st
from dotenv import load_dotenv

from src.chatbot import RetrievalChatbot
from src.embeddings import MyEmbedder
from src.store import PgVectorStore


STARTER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STARTER_DIR / "results"
BENCHMARK_PATH = STARTER_DIR / "benchmark" / "questions.json"


@st.cache_resource
def load_chatbot() -> tuple[RetrievalChatbot, dict[str, Any], dict[str, Any]]:
    load_dotenv(STARTER_DIR / ".env", override=False)
    with (RESULTS_DIR / "run_metadata.json").open(encoding="utf-8") as file:
        run_metadata = json.load(file)
    with BENCHMARK_PATH.open(encoding="utf-8") as file:
        benchmark = json.load(file)

    model_name = os.getenv("EMBEDDING_MODEL", "").strip()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not model_name:
        raise ValueError("Set EMBEDDING_MODEL in starter/.env")
    if not database_url:
        raise ValueError("Set DATABASE_URL in starter/.env")

    embedder = MyEmbedder(model_name)
    if embedder.model_name != run_metadata["model_name"]:
        raise ValueError(
            "EMBEDDING_MODEL does not match the model used to build the stored vectors "
            f"({run_metadata['model_name']})"
        )
    if embedder.dimensions != run_metadata["embedding_dimensions"]:
        raise ValueError(
            "Embedding dimensions do not match the stored vectors "
            f"({run_metadata['embedding_dimensions']})"
        )
    store = PgVectorStore(database_url, embedder.dimensions)
    chatbot = RetrievalChatbot(embedder, store)
    return chatbot, benchmark, run_metadata


def _show_sources(hits: list[dict[str, Any]]) -> None:
    if not hits:
        return
    with st.expander("Xem đoạn nguồn đã truy xuất"):
        for index, hit in enumerate(hits, start=1):
            section = hit.get("metadata", {}).get("section", "không rõ")
            st.markdown(
                f"**[{index}] Trang {hit['page']} · {section} · "
                f"similarity {hit['score']:.3f}**"
            )
            st.caption(hit["text"])


def _show_metrics() -> None:
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.is_file():
        return
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    rows = []
    for strategy in ("A", "B", "C"):
        result = summary[strategy]
        rows.append(
            {
                "Cấu hình": strategy,
                "Dev Hit@1": (
                    f"{result['dev']['hit_at_1']['hits']}/"
                    f"{result['dev']['hit_at_1']['total']}"
                ),
                "Dev Evidence Hit@3": (
                    f"{result['dev']['evidence_hit_at_3']['hits']}/"
                    f"{result['dev']['evidence_hit_at_3']['total']}"
                ),
                "Test Hit@1": (
                    f"{result['test']['hit_at_1']['hits']}/"
                    f"{result['test']['hit_at_1']['total']}"
                ),
                "Test Evidence Hit@3": (
                    f"{result['test']['evidence_hit_at_3']['hits']}/"
                    f"{result['test']['evidence_hit_at_3']['total']}"
                ),
            }
        )
    st.table(rows)


def main() -> None:
    st.set_page_config(page_title="Music History RAG Demo", page_icon="🎼", layout="wide")
    st.title("🎼 Music History — RAG Chatbot")
    st.write(
        "Đặt câu hỏi về tài liệu. Bot truy xuất top-3 đoạn liên quan, trả lời dựa trên "
        "bằng chứng và hiển thị trang nguồn."
    )

    try:
        chatbot, benchmark, run_metadata = load_chatbot()
    except psycopg.Error as exc:
        st.error(f"Không thể kết nối PostgreSQL: {exc}")
        st.stop()
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        st.error(f"Chưa thể khởi động chatbot: {exc}")
        st.stop()

    sections = benchmark["metadata_schema"]["values"]
    with st.sidebar:
        st.header("Cấu hình truy xuất")
        strategy = st.selectbox(
            "Chiến lược",
            ("A", "B", "C"),
            format_func=lambda value: {
                "A": "A · Fixed-size",
                "B": "B · Theo đoạn",
                "C": "C · Theo đoạn + filter",
            }[value],
        )
        section_options = ["Tất cả section", *sections]
        selected_section = st.selectbox(
            "Lọc section (chỉ áp dụng với C)",
            section_options,
            disabled=strategy != "C",
        )
        if strategy == "C":
            st.caption("Filter được áp dụng trước khi tìm top-3.")
        else:
            st.caption("Theo thiết kế lab, A và B không lọc metadata.")
        st.divider()
        st.write(f"**Embedding:** `{run_metadata['model_name']}`")
        st.write(f"**Kích thước corpus:** {run_metadata['page_count']} trang")
        st.write(f"**Chat model:** `{chatbot.chat_model}`")
        if st.button("Xóa hội thoại"):
            st.session_state.messages = []
            st.rerun()

    st.subheader("Kết quả benchmark")
    _show_metrics()
    if run_metadata.get("test_split_status"):
        st.caption(run_metadata["test_split_status"])
    st.divider()

    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _show_sources(message.get("hits", []))
                if message.get("filters"):
                    st.caption(f"Filter đã dùng: {message['filters']}")

    prompt = st.chat_input("Ví dụ: According to the book, what are the raw materials of music?")
    if prompt:
        section_filter = (
            selected_section if strategy == "C" and selected_section != "Tất cả section" else None
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            with st.chat_message("assistant"):
                with st.spinner("Đang embedding câu hỏi, truy xuất và tổng hợp bằng chứng..."):
                    result = chatbot.answer(prompt, strategy, section=section_filter, top_k=3)
                st.markdown(result["answer"])
                _show_sources(result["hits"])
                if result["filters"]:
                    st.caption(f"Filter đã dùng: {result['filters']}")
            messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "hits": result["hits"],
                        "filters": result["filters"],
                    },
                ]
            )
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"Không thể trả lời câu hỏi: {exc}")
        except psycopg.Error as exc:
            st.error(f"Không thể truy xuất dữ liệu từ PostgreSQL: {exc}")


if __name__ == "__main__":
    main()
