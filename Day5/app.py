"""
app.py — Yessenov Foundation Grants AI Assistant (Streamlit chat on gemma4).

A grounded RAG chatbot about the Shakhmardan Yessenov Foundation's grants and programs.
Its first job is to NOT invent facts: every answer is built only from chunks retrieved
from data we scraped (data/docs), and when the data has no answer it says so honestly.

Standout features here:
  * Source links + a "What I found" expander under every answer (proof it didn't invent)
  * Honest, friendly refusal when the data has no answer
  * Grant recommender (asks 2-3 questions -> suggests real programs)
  * Smart email: notices a "leave a request" intent and offers a (button-only) summary email
  * Multilingual (answers in RU / KK / EN), streaming output, example-question buttons
  * Winners-per-year chart built from real YDL winners lists (data/stats.csv)

Run:  streamlit run app.py    (after scraper.py + build_index.py)
"""
from __future__ import annotations
import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import rag
import emailer

BASE = Path(__file__).parent
ASSETS = BASE / "assets"
LOGO = str(ASSETS / "logo_yessenov_2.png")          # crisp circular brand mark (icon, header, avatar)
FOUNDER = str(ASSETS / "shakhmardan-yessenov.jpg")
BANNER = str(ASSETS / "shakhmardan-yessenov-banner.jpg")
PURPLE = "#79378b"

st.set_page_config(page_title="Ассистент фонда Есенова", page_icon=LOGO, layout="wide")

# --------------------------------------------------------------------------- styling
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 4.5rem; }}   /* clear Streamlit's top toolbar so the header shows */
      .brand-title {{ color: {PURPLE}; font-weight: 800; font-size: 2rem; margin: 0; line-height: 1.15; }}
      .brand-sub {{ color: #6b6b6b; font-size: 0.98rem; margin-top: .2rem; }}
      section[data-testid="stSidebar"] [data-testid="stImage"],
      section[data-testid="stSidebar"] [data-testid="stImageContainer"] {{ width: 100% !important; text-align: center; }}
      section[data-testid="stSidebar"] [data-testid="stImage"] img,
      section[data-testid="stSidebar"] [data-testid="stImageContainer"] img {{ display: inline-block !important; max-width: 210px; height: auto; }}
      .src-pill a {{
          display:inline-block; background:{PURPLE}1a; color:{PURPLE}!important;
          border:1px solid {PURPLE}55; border-radius:999px; padding:2px 12px; margin:3px 6px 3px 0;
          font-size:.83rem; text-decoration:none; }}
      .stButton>button {{ border-radius: 10px; }}
      .example-btn button {{ text-align:left; }}
      div[data-testid="stChatMessage"] {{ background: transparent; }}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Когда проходит Yessenov Data Lab 2026 и кто может участвовать?",
    "Кто прошёл на Yessenov Data Lab 2026?",
    "Построй график победителей Data Lab по годам",
    "Расскажи про стипендию имени академика Есенова",
]
REQUEST_HINTS = ("заявк", "записаться", "подать", "хочу участвовать", "хочу подать",
                 "свяжитесь", "оставить контакт", "регистрац", "как поступить", "запиши меня")
CHART_HINTS = ("график", "графи", "диаграмм", "chart", "статистик", "построй", "визуал",
               "по годам", "динамик", "сколько победител", "winners chart")


# --------------------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def load_stats() -> pd.DataFrame | None:
    p = BASE / "data" / "stats.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def index_ready() -> bool:
    return (BASE / "data" / "index" / "embeddings.npy").exists()


@st.cache_data(show_spinner=False)
def program_options() -> list[str]:
    """Program names (from the scraped docs) for the comparison picker."""
    out, seen = [], set()
    for p in sorted((BASE / "data" / "docs").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("type") in ("program", "edition") and d.get("title") and d["title"] not in seen:
            seen.add(d["title"])
            out.append(d["title"])
    return out


@st.cache_data(show_spinner=False)
def load_program_stats() -> pd.DataFrame | None:
    p = BASE / "data" / "program_stats.csv"
    return pd.read_csv(p) if p.exists() else None


def detect_chart_intent(q: str) -> str | None:
    """Return which chart the user is asking for ('programs'/'years'), or None."""
    ql = q.lower()
    if not any(h in ql for h in CHART_HINTS):
        return None
    if any(w in ql for w in ("програм", "program")):
        return "programs"
    return "years"


def chart_intro(kind: str) -> str:
    """A factual one-liner for a chart, computed in Python from the CSV (never by the LLM)."""
    if kind == "programs":
        df = load_program_stats()
        if df is None or df.empty:
            return "Пока нет данных для графика по программам."
        return "Вот сколько мест в последнем наборе по программам фонда (по реальным данным сайта):"
    df = load_stats()
    if df is None or df.empty:
        return "Пока нет данных для графика."
    total, lo, hi = int(df["winners"].sum()), int(df["winners"].min()), int(df["winners"].max())
    return (f"Вот победители Yessenov Data Lab по годам — всего {total} человек за {len(df)} наборов "
            f"(от {lo} до {hi} в год):")


def render_chart(kind: str) -> None:
    """Draw the requested chart from REAL numbers (CSV)."""
    if kind == "programs":
        df = load_program_stats()
        if df is None or df.empty:
            st.caption("Нет данных по программам (запустите build_program_stats.py).")
            return
        chart = (alt.Chart(df).mark_bar(color=PURPLE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                 .encode(x=alt.X("program:N", title="Программа", sort="-y"),
                         y=alt.Y("places:Q", title="Мест в наборе"),
                         tooltip=["program", "places", "kind", "year"]))
        st.altair_chart(chart, use_container_width=True)
        st.caption("Мест/грантов в последнем наборе по программам. Источник: страницы программ фонда.")
        return
    df = load_stats()
    if df is None or df.empty:
        st.caption("Нет данных статистики (запустите build_stats.py).")
        return
    chart = (alt.Chart(df).mark_bar(color=PURPLE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
             .encode(x=alt.X("year:O", title="Год"), y=alt.Y("winners:Q", title="Победителей"),
                     tooltip=["year", "winners"]))
    st.altair_chart(chart, use_container_width=True)
    st.caption("Источник: официальные списки победителей на сайте фонда. 2021–2022 — наборов не было.")


def render_sources(hits: list[dict]) -> None:
    """Source pills + a transparent 'what I found' expander under an answer."""
    best = max((h.get("score", 0.0) for h in hits), default=0.0)
    if best >= rag.RELEVANCE_FLOOR:
        seen, pills = set(), []
        for h in hits:
            url, title = h.get("url", ""), h.get("title", "")
            if h.get("score", 0.0) < rag.RELEVANCE_FLOOR or url in seen:
                continue
            if not url.startswith(("http://", "https://")):     # only safe link schemes
                continue
            seen.add(url)
            safe_url, safe_title = html.escape(url, quote=True), html.escape(title)
            pills.append(f'<span class="src-pill"><a href="{safe_url}" target="_blank">🔗 {safe_title}</a></span>')
            if len(pills) >= 3:
                break
        if pills:
            st.markdown("<div>Источники: " + " ".join(pills) + "</div>", unsafe_allow_html=True)
    else:
        st.caption("⚠️ Точного совпадения в собранных данных нет — ниже лишь ближайшие фрагменты.")

    with st.expander("🔎 Что я нашёл в данных (прозрачность)"):
        for i, h in enumerate(hits, 1):
            text = h.get("text", "")
            st.markdown(f"**[{i}] {h.get('title','')}** · _{h.get('section','')}_ · близость **{h.get('score',0.0):.2f}**")
            st.caption(h.get("url", ""))
            st.write(text[:600] + ("…" if len(text) > 600 else ""))
            if i < len(hits):
                st.divider()


def build_summary_email(messages: list[dict]) -> tuple[str, str, str]:
    """Ask the model to summarize the chat into a short admin note (subject, html, text)."""
    recent = [m for m in messages if m.get("content")][-12:]      # cap so summarization stays in context
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)[:6000]
    prompt = [
        {"role": "system", "content": "Ты помощник, который кратко резюмирует диалог для администратора "
                                      "фонда. Выдели: кто писал, что человек хочет/спрашивал, есть ли заявка "
                                      "или контакт. 4-6 предложений, по-деловому, на русском."},
        {"role": "user", "content": f"Резюмируй диалог:\n\n{convo}"},
    ]
    try:
        summary = rag.complete_chat(prompt, temperature=0.3)
    except Exception:
        summary = convo[:1500]
    safe = html.escape(summary).replace("\n", "<br>")            # model output -> escape before HTML
    html_body = (f"<h2 style='color:{PURPLE}'>Саммари разговора — ассистент фонда Есенова</h2>"
                 f"<p>{safe}</p><hr>"
                 f"<p style='color:#888;font-size:12px'>Отправлено по кнопке из чат-ассистента. "
                 f"Получатель — только администратор (вы).</p>")
    return "Саммари разговора из чат-ассистента фонда", html_body, summary


def do_send_email(out=None) -> None:
    """Send the summary. `out` is where feedback is shown (sidebar by default, the chat column
    when invoked from an in-chat offer button)."""
    out = out or st.sidebar
    msgs = st.session_state.messages
    if not any(m["role"] == "user" for m in msgs):
        out.warning("Сначала задайте боту вопрос — потом можно отправить саммари.")
        return
    with st.spinner("Готовлю и отправляю саммари…"):
        subject, html_body, text = build_summary_email(msgs)
        res = emailer.send_summary(subject, html_body, text)
    if res.get("ok"):
        out.success(f"📧 Письмо отправлено на {res.get('to')}")
    else:
        # No auto-send configured/working -> the no-setup path: open in the user's mail app.
        out.info("Авто-отправка не настроена. Можно отправить вручную в один клик:")
        out.link_button("✉️ Открыть готовое письмо в почте", emailer.mailto_link(subject, text))
        with out.expander("…или скопировать текст саммари"):
            st.code(text)
        out.caption(f"(авто-отправка пока не работает: {res.get('error','')[:90]})")


# --------------------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def queue_prompt(text: str) -> None:
    st.session_state.pending = text


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.image(FOUNDER, width=200)              # fixed size — don't scale with the sidebar slider
    st.markdown("#### Шахмардан Есенов")
    st.caption("Академик, геолог, первый президент Академии наук Казахской ССР. "
               "Фонд его имени поддерживает науку и образование в Казахстане.")
    st.divider()

    with st.expander("📊 Победители Yessenov Data Lab по годам"):
        stats = load_stats()
        if stats is not None:
            import altair as alt
            chart = (
                alt.Chart(stats).mark_bar(color=PURPLE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("year:O", title="Год"),
                    y=alt.Y("winners:Q", title="Победителей"),
                    tooltip=["year", "winners"],
                )
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Источник: официальные списки победителей на сайте фонда. "
                       "2021–2022 — наборов не было (поэтому их нет на графике).")
        else:
            st.caption("Нет данных статистики (запустите build_stats.py).")

    with st.expander("📈 Места по программам (последний набор)"):
        pstats = load_program_stats()
        if pstats is not None and not pstats.empty:
            render_chart("programs")
        else:
            st.caption("Нет данных (запустите build_program_stats.py).")

    with st.expander("🎯 Подобрать программу"):
        with st.form("recommender"):
            who = st.selectbox("Кто вы?", ["Студент бакалавриата", "Магистрант / докторант",
                                           "Действующий учёный / специалист", "Преподаватель"])
            field = st.selectbox("Направление", ["Наука и исследования", "Data Science / IT",
                                                 "Английский язык", "Шахматы", "Личное развитие", "Не уверен(а)"])
            want = st.selectbox("Что ищете?", ["Грант / стипендию", "Стажировку", "Обучение / интенсив", "Любое"])
            go = st.form_submit_button("Подобрать", width="stretch")
        if go:
            queue_prompt(
                f"Я — {who.lower()}, направление: {field.lower()}, ищу: {want.lower()}. "
                f"Какие программы и гранты фонда мне подойдут? Перечисли подходящие и кратко поясни условия."
            )
            st.rerun()

    progs = program_options()
    if len(progs) >= 2:
        with st.expander("⚖️ Сравнить две программы"):
            with st.form("compare"):
                pa = st.selectbox("Программа A", progs, index=0)
                pb = st.selectbox("Программа B", progs, index=1)
                cmp = st.form_submit_button("Сравнить", width="stretch")
            if cmp:
                if pa == pb:
                    st.warning("Выберите две разные программы.")
                else:
                    queue_prompt(
                        f"Сравни программы «{pa}» и «{pb}» по данным фонда: для кого, сроки или статус "
                        f"(действует/завершена), что даёт. Ответь короткой таблицей в markdown. "
                        f"Если по какой-то программе данных нет — честно отметь это в таблице."
                    )
                    st.rerun()

    st.divider()
    if st.button("📧 Отправить саммари мне на почту", width="stretch"):
        do_send_email()
    st.caption("Письмо уходит только на ваш собственный ящик и только по этой кнопке.")
    st.divider()
    if st.button("🗑️ Очистить чат", width="stretch"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()
    st.caption("Бот отвечает только по данным с сайта фонда и честно говорит, если ответа нет. "
               "Языки: русский, қазақша, English.")


# --------------------------------------------------------------------------- header
c1, c2 = st.columns([1, 5], vertical_alignment="center")
with c1:
    st.image(LOGO, width=130)
with c2:
    st.markdown('<p class="brand-title">Ассистент по грантам фонда Есенова</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Консультации по грантам, стипендиям и программам фонда на основе '
                'официальных данных сайта. Если сведений в источниках нет — честно сообщу об этом.</p>',
                unsafe_allow_html=True)

if not index_ready():
    st.error("Индекс не найден. Запустите в терминале:  `python scraper.py`  затем  `python build_index.py`.")
    st.stop()

# Welcome screen with example buttons (only before the first message)
if not st.session_state.messages and not st.session_state.pending:
    _, bmid, _ = st.columns([1, 2, 1])           # smaller, centered banner
    with bmid:
        st.image(BANNER, width="stretch")
    st.markdown("##### Например, спросите:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        with cols[i % 2]:
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(ex, key=f"ex_{i}", width="stretch"):
                queue_prompt(ex)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- render history
for mi, m in enumerate(st.session_state.messages):
    avatar = LOGO if m["role"] == "assistant" else None
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
        if m.get("chart"):
            render_chart(m["chart"])
        if m["role"] == "assistant" and m.get("hits"):
            render_sources(m["hits"])
        if m.get("offer_email"):
            st.info("Похоже, вы хотите оставить заявку — могу отправить короткое саммари администратору "
                    "фонда (только на ваш собственный ящик, по кнопке).")
            if st.button("📨 Оставить заявку — отправить саммари администратору", key=f"offer_{mi}"):
                do_send_email(out=st)


# --------------------------------------------------------------------------- handle input
typed = st.chat_input("Напишите вопрос про гранты, стипендии или программы фонда…")
queued = st.session_state.pending
st.session_state.pending = None          # consume unconditionally — no stale auto-fire next run
prompt = typed or queued

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chart_kind = detect_chart_intent(prompt)
    with st.chat_message("assistant", avatar=LOGO):
        if chart_kind:                                   # the bot CAN draw charts (real numbers only)
            answer = chart_intro(chart_kind)
            st.markdown(answer)
            render_chart(chart_kind)
            hits = []
        else:
            try:
                history = st.session_state.messages[:-1]
                search_query = rag.condense_query(prompt, history)   # resolve follow-ups for retrieval
                hits = rag.retrieve(search_query)
                messages = rag.make_messages(prompt, hits, history)
                answer = st.write_stream(rag.stream_chat(messages))
            except Exception as e:
                answer = ("Извините, у меня сейчас техническая ошибка при обращении к модели. "
                          "Попробуйте ещё раз чуть позже.")
                hits = []
                st.markdown(answer)
                st.caption(f"({type(e).__name__})")
            if hits:
                render_sources(hits)

    # the "leave a request" offer is rendered from history (persists after rerun)
    offer_email = (not chart_kind) and any(h in prompt.lower() for h in REQUEST_HINTS)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits, "offer_email": offer_email, "chart": chart_kind}
    )
    st.rerun()
