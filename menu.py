"""Streamlit interface for the LLM prompt manager."""

import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from driver import Driver
from implementations.bias_evaluator import BiasEvaluator
from implementations.gemini_interface import GeminiInterface

load_dotenv()

# responder type -> display details. "study" models are the four used in the project.
MODELS = {
    "openai-gpt-nano": {"name": "ChatGPT", "model": "gpt-5.4-nano", "origin": "US", "env": "OPENAI_API_KEY",
                        "url": "https://platform.openai.com/api-keys", "rpm": 2800, "study": True},
    "gemini-3.1-flash-lite": {"name": "Gemini", "model": "gemini-3.1-flash-lite", "origin": "US", "env": "GEMINI_API_KEY",
                              "url": "https://aistudio.google.com/app/apikey", "rpm": 8, "study": True},
    "deepseek-v4-flash": {"name": "DeepSeek", "model": "deepseek-v4-flash", "origin": "China", "env": "DEEPSEEK_API_KEY",
                          "url": "https://platform.deepseek.com", "rpm": 480, "study": True},
    "kimi-k2": {"name": "Kimi", "model": "kimi-k2.6", "origin": "China", "env": "KIMI_API_KEY",
                "url": "https://platform.moonshot.ai/console/api-keys", "rpm": 3, "study": True},
    "seed-2-0-lite-260428": {"name": "Doubao", "model": "seed-2-0-lite", "origin": "China", "env": "DOUBAO_API_KEY",
                             "url": "https://console.byteplus.com", "rpm": 60, "study": False},
    "qwen-plus": {"name": "Qwen", "model": "qwen-plus", "origin": "China", "env": "QWEN_API_KEY",
                  "url": "https://modelstudio.console.alibabacloud.com", "rpm": 60, "study": False},
    "glm-4.5-flash": {"name": "GLM", "model": "glm-4.5-flash", "origin": "China", "env": "GLM_API_KEY",
                      "url": "https://z.ai/manage-apikey/apikey-list", "rpm": 20, "study": False},
}
JUDGE_MODEL = "gemini-3.1-flash-lite"
SCORES = {"factual_balance": "Factual balance", "framing_bias": "Framing", "attribution_of_responsibility": "Attribution",
          "political_avoidance": "Avoidance", "loaded_language": "Loaded language"}
PAGES = ["1  Import prompts", "2  Run models", "3  Results"]
COLOURS = {"Answered": "background-color:#e6efe3;color:#2f5a2a",
           "Dodged": "background-color:#f8e7c4;color:#7a5410",
           "Blocked": "background-color:#f6d3c8;color:#8a2f1a",
           "Error": "background-color:#eeeeee;color:#555555"}

# Small visual touches the theme file can't do
CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e6dccf; border-radius: 0.6rem; padding: 0.8rem 1rem;}
[data-testid="stMetricLabel"] p {color: #7a6a5c;}
.tag {display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 1rem;
      background: #f1e9df; color: #7a5a48; margin-left: 0.3rem;}
.tag.cn {background: #f6e0d7; color: #9a3f27;}
.muted {color: #7a6a5c; font-size: 0.9rem;}
</style>
"""


def name(responder_type: str) -> str:
    return MODELS.get(responder_type, {}).get("name", responder_type)


def readable_error(error) -> str:
    """Turn a raw API error into one readable line, e.g. the provider's own refusal message."""
    text = str(error or "")
    found = re.search(r"'message': '([^']+)'", text)
    return found.group(1) if found else text


def outcome(row: dict) -> str:
    """Answered, Dodged (judge avoidance 4-5), Blocked (provider refused) or Error."""
    if row.get("status") == "blocked":
        return "Blocked"
    if row.get("status") != "success":
        return "Error"
    try:
        return "Dodged" if int(row.get("political_avoidance") or 0) >= 4 else "Answered"
    except ValueError:
        return "Answered"


class StreamlitMenu:
    """Streamlit-based interface for the LLM prompt manager."""

    def __init__(self):
        self.driver = Driver()
        for key, default in (("prompts", []), ("results", []), ("processing_active", False),
                             ("processing_params", None), ("prompt_file", None)):
            st.session_state.setdefault(key, default)
        # Optional prompt file to preload, e.g. `streamlit run main.py -- prompts.tsv` (used for demos)
        preload = next((a for a in sys.argv[1:] if os.path.isfile(a)), None)
        if preload and "preloaded" not in st.session_state:
            st.session_state.preloaded = True
            self._load_prompts(preload, os.path.basename(preload))

    # ---------- layout ----------

    def run(self) -> None:
        st.set_page_config(page_title="LLM Prompt Manager", page_icon=":material/translate:", layout="wide")
        st.markdown(CSS, unsafe_allow_html=True)

        # A finished run asks to jump to Results; apply it before the nav widget exists
        if "goto" in st.session_state:
            st.session_state.nav = st.session_state.pop("goto")

        with st.sidebar:
            st.markdown("### LLM Prompt Manager")
            st.caption("Ask several AI models the same question in English and Mandarin, then compare the answers.")
            page = st.radio("Steps", PAGES, key="nav", label_visibility="collapsed",
                            disabled=st.session_state.processing_active)
            session_box = st.container()

        if page == PAGES[0]:
            self.import_prompts_section()
        elif page == PAGES[1]:
            self.process_prompts_section()
        else:
            self.view_results_section()

        # Drawn last so the counts include anything this page just loaded
        with session_box:
            st.divider()
            st.caption("This session")
            n_prompts, n_results = len(st.session_state.prompts), len(st.session_state.results)
            st.markdown(f"**{n_prompts}** prompt{'s' * (n_prompts != 1)} loaded  \n"
                        f"**{n_results}** result{'s' * (n_results != 1)}")
            if st.session_state.prompt_file:
                st.caption(f"File: {st.session_state.prompt_file}")

    # ---------- 1. import ----------

    def import_prompts_section(self) -> None:
        st.title("Import prompts")
        st.markdown('<p class="muted">A tab-separated file with one prompt per row, '
                    'with an <b>English</b> and a <b>Mandarin</b> column.</p>', unsafe_allow_html=True)

        left, right = st.columns([3, 1], vertical_alignment="center")
        with left:
            uploaded = st.file_uploader("Prompt file", type=["tsv", "txt", "csv"], label_visibility="collapsed")
        with right:
            if st.button("Load example prompts", icon=":material/description:", width="stretch",
                         help="sample_data.tsv: five questions about the Opium Wars"):
                self._load_prompts("sample_data.tsv", "sample_data.tsv")

        if uploaded is not None and uploaded.name != st.session_state.prompt_file:
            path = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            self._load_prompts(path, uploaded.name)

        prompts = st.session_state.prompts
        if not prompts:
            st.markdown("##### How it works")
            steps = st.columns(3)
            for col, (title, text) in zip(steps, (
                    ("1  Import", "Load a file of questions, each written in English and in Mandarin."),
                    ("2  Run", "Send every question to each model in both languages. Mandarin answers are "
                               "translated and a judge scores every answer for bias."),
                    ("3  Compare", "See which models answered, dodged or were blocked, side by side by language."))):
                with col.container(border=True, height=130):
                    st.markdown(f"**{title}**")
                    st.markdown(f'<span class="muted">{text}</span>', unsafe_allow_html=True)
            return

        df = pd.DataFrame(prompts)
        events = df["Event"].nunique() if "Event" in df else None
        cols = st.columns(3)
        cols[0].metric("Prompts", len(df))
        cols[1].metric("Languages", 2, help="Every prompt is asked in English and in Mandarin")
        cols[2].metric("Events", events if events else "-")

        show = [c for c in ("Prompt ID", "Event", "English", "Mandarin") if c in df] or list(df.columns)
        st.dataframe(df[show], hide_index=True, width="stretch",
                     column_config={"English": st.column_config.TextColumn(width="large"),
                                    "Mandarin": st.column_config.TextColumn(width="large")})
        st.button("Continue to Run models", type="primary", icon=":material/arrow_forward:",
                  on_click=lambda: st.session_state.update(nav=PAGES[1]))

    def _load_prompts(self, path: str, label: str) -> None:
        try:
            st.session_state.prompts = self.driver.import_prompts(path)
            st.session_state.prompt_file = label
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")

    # ---------- 2. run ----------

    def process_prompts_section(self) -> None:
        st.title("Run models")
        prompts = st.session_state.prompts
        if not prompts:
            st.warning("Import a prompt file first (step 1).", icon=":material/arrow_back:")
            return
        busy = st.session_state.processing_active

        if not busy:
            st.markdown("##### Models")
            study = [t for t, m in MODELS.items() if m["study"]]
            cols = st.columns(len(study))
            for col, t in zip(cols, study):
                m = MODELS[t]
                with col.container(border=True):
                    st.toggle(f"**{m['name']}**", value=True, key=f"use_{t}")
                    tag = '<span class="tag cn">China</span>' if m["origin"] == "China" else '<span class="tag">US</span>'
                    st.markdown(f'<span class="muted">{m["model"]}</span>{tag}', unsafe_allow_html=True)
                    st.caption("Key found" if os.getenv(m["env"]) else "No key in .env")

            with st.expander("Other adaptors (not used in the study)"):
                for t, m in MODELS.items():
                    if not m["study"]:
                        st.toggle(f"{m['name']}  ·  {m['model']}", value=False, key=f"use_{t}")

            selected = [t for t in MODELS if st.session_state.get(f"use_{t}")]

            st.markdown("##### Scoring")
            evaluate = st.toggle("Translate Mandarin answers and score every answer with the judge",
                                 value=True, key="evaluate_bias",
                                 help=f"The judge ({JUDGE_MODEL}) scores five bias categories from 1 to 5.")

            with st.expander("API keys and rate limits"):
                st.caption("Keys are read from the .env file. Only type one here to override it for this session.")
                for t in selected + (["__judge__"] if evaluate else []):
                    m = MODELS.get(t, {"name": "Judge (Gemini)", "env": "GEMINI_API_KEY", "rpm": 8, "url": ""})
                    a, b = st.columns([3, 1])
                    a.text_input(f"{m['name']} key", type="password", key=f"key_{t}",
                                 placeholder="using .env" if os.getenv(m["env"]) else "required",
                                 help=m.get("url") or None)
                    b.number_input("Requests / min", min_value=1, value=m["rpm"], key=f"rpm_{t}")

            calls = len(prompts) * 2 * len(selected)
            st.divider()
            if not selected:
                st.warning("Turn on at least one model.")
                return
            st.markdown(f'<p class="muted">{len(prompts)} prompt(s) × 2 languages × {len(selected)} model(s) '
                        f'= <b>{calls}</b> answers</p>', unsafe_allow_html=True)

            if st.button(f"Run {calls} answers", type="primary", icon=":material/play_arrow:"):
                keys = {t: st.session_state.get(f"key_{t}") or os.getenv(MODELS[t]["env"], "") for t in selected}
                missing = [name(t) for t, k in keys.items() if not k]
                judge_key = st.session_state.get("key___judge__") or os.getenv("GEMINI_API_KEY", "")
                if missing:
                    st.error(f"No API key for: {', '.join(missing)}. Add it to .env or under API keys.")
                    return
                if evaluate and not judge_key:
                    st.error("The judge needs a Gemini API key.")
                    return
                st.session_state.processing_params = {
                    "keys": keys,
                    "rpm": {t: st.session_state.get(f"rpm_{t}", MODELS[t]["rpm"]) for t in selected},
                    "evaluate": evaluate,
                    "judge_key": judge_key,
                    "judge_rpm": st.session_state.get("rpm___judge__", 8),
                }
                st.session_state.processing_active = True
                st.rerun()

        if busy and st.session_state.processing_params:
            self._run(st.session_state.processing_params)

    def _run(self, params: dict) -> None:
        keys, prompts = params["keys"], st.session_state.prompts
        total = len(prompts)
        progress = {t: 0 for t in keys}

        def update_progress(responder_type: str, current: int, _total: int) -> None:
            progress[responder_type] = current

        judge_client = evaluator = None
        if params["evaluate"]:
            try:
                judge_client = GeminiInterface(api_key=params["judge_key"])
                judge_client.set_model(JUDGE_MODEL)
                evaluator = BiasEvaluator(judge_client)
            except Exception as e:
                st.error(f"Couldn't start the judge: {e}")
                st.session_state.processing_active = False
                st.session_state.processing_params = None
                return

        all_results = []
        with st.status(f"Running {len(keys)} model(s)…", expanded=True) as status:
            bars = {t: st.progress(0.0, text=f"{name(t)}: starting") for t in keys}
            started = time.time()
            with ThreadPoolExecutor(max_workers=len(keys)) as executor:
                futures = {
                    executor.submit(self.driver.process_prompts, prompts, key, responder_type=t,
                                    evaluate_bias=params["evaluate"], judge_client=judge_client, evaluator=evaluator,
                                    rate_limit_rpm=params["rpm"][t], judge_rate_limit_rpm=params["judge_rpm"],
                                    progress_callback=update_progress): t
                    for t, key in keys.items()
                }
                while futures:
                    for t, bar in bars.items():
                        bar.progress(progress[t] / total, text=f"{name(t)}: {progress[t]} of {total} prompts")
                    # pop() so each future is only counted once - as_completed() re-yields finished ones
                    try:
                        for future in as_completed(list(futures), timeout=0.3):
                            t = futures.pop(future)
                            try:
                                results, _ = future.result()
                            except Exception as e:
                                print(f"[{t}] crashed: {e}")
                                results = []
                            all_results.extend(results)
                            progress[t] = total
                    except FuturesTimeout:
                        pass
            for t, bar in bars.items():
                bar.progress(1.0, text=f"{name(t)}: done")
            status.update(label=f"Finished in {time.time() - started:.0f}s: {len(all_results)} answers",
                          state="complete")

        # Also save to disk so a refreshed tab loses nothing
        if all_results:
            self.driver.export_results(all_results, f"results_{time.strftime('%Y%m%d_%H%M%S')}.tsv")
        st.session_state.results = all_results
        st.session_state.processing_active = False
        st.session_state.processing_params = None
        st.session_state.goto = PAGES[2]
        st.rerun()

    # ---------- 3. results ----------

    def view_results_section(self) -> None:
        st.title("Results")
        results = st.session_state.results
        if not results:
            st.info("No results yet. Run some prompts first (step 2).", icon=":material/arrow_back:")
            return

        df = pd.DataFrame(results)
        df["Model"] = df["responder"].map(name)
        df["Outcome"] = [outcome(r) for r in results]
        df["Answer"] = [f"Blocked by provider: {readable_error(r.get('error'))}" if o == "Blocked"
                        else (r.get("response") or readable_error(r.get("error")))
                        for r, o in zip(results, df["Outcome"])]
        for k in SCORES:
            if k in df:
                df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
        # Keep models in study order, English before Mandarin
        order = {m["name"]: i for i, m in enumerate(MODELS.values())}
        df = df.sort_values(by=["Model", "language"], key=lambda s: s.map(order) if s.name == "Model" else s,
                            kind="stable")

        counts = df["Outcome"].value_counts()
        cols = st.columns(4)
        cols[0].metric("Answers", len(df))
        cols[1].metric("Answered", int(counts.get("Answered", 0)))
        cols[2].metric("Dodged", int(counts.get("Dodged", 0)), help="Answered, but the judge scored political avoidance 4 or 5")
        cols[3].metric("Blocked", int(counts.get("Blocked", 0)), help="The provider refused the request")

        st.markdown("##### By model and language")
        summary = (df.pivot_table(index="Model", columns="language", values="Outcome",
                                  aggfunc=lambda s: ", ".join(f"{n} {o.lower()}" for o, n in s.value_counts().items()))
                   .reindex([m["name"] for m in MODELS.values() if m["name"] in set(df["Model"])]))
        def tint(cell) -> str:
            text = str(cell)
            if "blocked" in text:
                return COLOURS["Blocked"]
            if "dodged" in text:
                return COLOURS["Dodged"]
            return COLOURS["Answered"] if "answered" in text else ""
        st.dataframe(summary.style.map(tint), width="stretch")

        st.markdown("##### All answers")
        f1, f2, f3 = st.columns(3)
        models = f1.multiselect("Model", sorted(df["Model"].unique()), placeholder="All models")
        langs = f2.multiselect("Language", sorted(df["language"].unique()), placeholder="Both languages")
        outs = f3.multiselect("Outcome", ["Answered", "Dodged", "Blocked", "Error"], placeholder="All outcomes")
        mask = pd.Series(True, index=df.index)
        for column, chosen in (("Model", models), ("language", langs), ("Outcome", outs)):
            if chosen:
                mask &= df[column].isin(chosen)
        view = df[mask]

        score_cols = [c for c in SCORES if c in view]
        table = view[["Model", "language", "Outcome"] + [c for c in ("Event",) if c in view]
                     + ["Answer"] + score_cols].rename(columns={"language": "Language", **SCORES})
        for c in score_cols:  # blocked rows have no scores; show a dash instead of "None"
            table[SCORES[c]] = table[SCORES[c]].astype("string").fillna("–")
        st.dataframe(table.style.map(lambda v: COLOURS.get(v, ""), subset=["Outcome"]),
                     hide_index=True, width="stretch", height=360,
                     column_config={"Answer": st.column_config.TextColumn(width="large")})

        with st.expander("Read one answer in full", expanded=False):
            labels = {f"{n + 1}. {r['Model']} · {r['language']} · {r.get('Prompt ID', '')} · {r['Outcome']}": i
                      for n, (i, r) in enumerate(view.iterrows())}
            if labels:
                pick = st.selectbox("Answer", list(labels), label_visibility="collapsed")
                r = view.loc[labels[pick]]
                question = r.get("English") if r["language"] == "English" else r.get("Mandarin")
                st.markdown(f"**Question:** {question}")
                if r["Outcome"] == "Blocked":
                    st.error(f"Blocked by the provider: {readable_error(r.get('error'))}")
                else:
                    a, b = st.columns(2) if r["language"] == "Mandarin" else (st.container(), None)
                    a.markdown("**Answer**")
                    a.write(r.get("response") or r.get("error") or "")
                    if b is not None:
                        b.markdown("**English translation**")
                        b.write(r.get("response_translated") or "-")
                    if r.get("bias_justification"):
                        st.markdown("**Judge's scores**")
                        sc = st.columns(len(score_cols))
                        for c, k in zip(sc, score_cols):
                            c.metric(SCORES[k], r.get(k) if pd.notna(r.get(k)) else "-")
                        st.caption(r.get("bias_justification"))

        st.download_button("Download results (.tsv)", icon=":material/download:", type="primary",
                           data=self.driver.exporter.export_results_to_string(results),
                           file_name="results.tsv", mime="text/tab-separated-values")


def main() -> None:
    StreamlitMenu().run()


if __name__ == "__main__":
    main()
