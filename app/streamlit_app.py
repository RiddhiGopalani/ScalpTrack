"""ScalpTrack web app: upload a scalp photo and get an explained, honest analysis.

Run from the project root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import time
from pathlib import Path

import engine
import streamlit as st
from condition_info import CONDITIONS, GENERAL_ROUTINE, RED_FLAGS

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="ScalpTrack · Scalp analysis", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- helpers


def html_block(markup: str) -> None:
    """Render HTML. Lines are left-stripped so Markdown never treats indented HTML as a code block."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def bullet_list(items, warn: bool = False) -> str:
    cls = "st-list warn" if warn else "st-list"
    return f'<ul class="{cls}">' + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


@st.cache_resource(show_spinner=False)
def resources() -> engine.Resources:
    return engine.load_resources()


html_block(f"<style>{(APP_DIR / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>")
RES = resources()

# --------------------------------------------------------------------------- header

html_block("""
<div class="st-topbar">
  <div class="st-brand"><div class="st-mark">S</div>ScalpTrack</div>
  <div class="st-pill">Research preview · not a medical device</div>
</div>
<div class="st-hero">
  <div class="st-eyebrow">Explainable scalp analysis</div>
  <h1>Understand what your scalp<br/>photo is showing.</h1>
  <p>Upload a clear close-up of your scalp. ScalpTrack compares it with ten scalp conditions using a
  pretrained MobileNetV2 network. It first checks that the photo looks like a scalp image, and it tells
  you when it isn't sure. You also get general care guidance and advice on when a dermatologist should
  take a look.</p>
</div>
""")

if RES.model_error:
    html_block(
        f'<div class="st-note alert"><b>The analysis model is not available.</b> {html.escape(RES.model_error)}'
        "<br/>Train it first: <code>python scripts/audit_dataset.py</code> then "
        "<code>python scripts/train_classifier.py</code>.</div>"
    )
    st.stop()

# --------------------------------------------------------------------------- upload

left, right = st.columns([1.05, 1], gap="large")
with left:
    with st.container(border=True, key="upload_card"):
        html_block("""
        <h3 style="margin:0 0 4px">1 · Upload a scalp photo</h3>
        <div class="sub" style="color:var(--muted);font-size:.88rem;margin-bottom:6px">JPG or PNG, up to 15 MB.
        The photo is processed in memory and is not stored.</div>""")
        upload = st.file_uploader("Scalp photo", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")
        samples = engine.sample_images(RES)
        sample_choice = None
        if samples and upload is None:
            sample_choice = st.selectbox(
                "Or try a sample image from the dataset's held-out test set",
                ["—"] + list(samples),
                help="Images the model never saw during training or model selection.",
            )
        if upload is not None:
            source_name, source_bytes = upload.name, upload.getvalue()
        elif sample_choice and sample_choice != "—":
            source_name, source_bytes = f"sample · {sample_choice}", samples[sample_choice].read_bytes()
        else:
            source_name, source_bytes = None, None
    html_block("""
    <div class="st-card">
      <div class="st-label">For the most reliable result</div>
      <div class="st-tips">
        <div class="st-tip"><div class="st-dot">1</div><div><b>Close-up</b><br/>Fill the frame with the scalp area you are concerned about.</div></div>
        <div class="st-tip"><div class="st-dot">2</div><div><b>Even light</b><br/>Use daylight. Avoid flash glare and deep shadows.</div></div>
        <div class="st-tip"><div class="st-dot">3</div><div><b>In focus</b><br/>Hold steady. Blurry photos switch off the hair measurements.</div></div>
        <div class="st-tip"><div class="st-dot">4</div><div><b>Part the hair</b><br/>Make sure the skin of the scalp is visible.</div></div>
      </div>
    </div>""")

image = None
with right:
    if source_bytes is not None:
        try:
            image = engine.decode_upload(source_bytes)
        except engine.UploadError as exc:
            html_block(f'<div class="st-note alert"><b>We could not use this file.</b> {html.escape(str(exc))}</div>')
        if image is not None:
            st.image(image, caption=f"{source_name} · {image.shape[1]}×{image.shape[0]} px (analysed size)", width="stretch")
            if st.button("Analyse my scalp", type="primary", width="stretch"):
                steps = [
                    "Checking image quality",
                    "Detecting hair and scalp regions",
                    "Extracting colour, edge and texture features",
                    "Comparing with the trained model",
                ]
                with st.status("Analysing your photo…", expanded=True) as status:
                    for s in steps[:-1]:
                        st.write(f"· {s}")
                    t0 = time.perf_counter()
                    try:
                        st.session_state["analysis"] = engine.analyze(image, RES)
                        st.session_state["analysis_name"] = source_name
                        st.write(f"· {steps[-1]}")
                        status.update(label=f"Analysis complete in {time.perf_counter() - t0:.1f} s", state="complete", expanded=False)
                    except engine.UploadError as exc:
                        st.session_state.pop("analysis", None)
                        status.update(label="Analysis failed", state="error")
                        st.error(str(exc))
                    except Exception:  # noqa: BLE001 - never show a stack trace to the user
                        st.session_state.pop("analysis", None)
                        status.update(label="Analysis failed", state="error")
                        st.error("Something went wrong while analysing this image. Please try another photo.")
    else:
        st.session_state.pop("analysis", None)
        html_block("""
        <div class="st-card" style="min-height: 330px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
          <div class="st-mark" style="width:56px;height:56px;font-size:1.6rem;margin-bottom:14px;">S</div>
          <h3>Your preview will appear here</h3>
          <div class="sub" style="max-width:340px;">Results take about a second. Nothing is uploaded to a server
          beyond this app, and nothing is saved.</div>
        </div>""")

analysis = st.session_state.get("analysis")
if not analysis or source_name is None or st.session_state.get("analysis_name") != source_name:
    html_block("""
    <div class="st-footer">ScalpTrack is an experimental computer-vision project. It is not a medical device and does not
    provide a diagnosis. Always consult a qualified healthcare professional about health concerns.</div>""")
    st.stop()

# --------------------------------------------------------------------------- results

r = analysis["result"]
cls = r["classification"]
scores: dict = cls["class_scores"]
top_key = cls["predicted_label"]
info = CONDITIONS.get(top_key, {}) if analysis["verdict"] not in ("ood", "healthy") else {}
verdict = analysis["verdict"]
top = analysis["top_score"]
quality = r["quality"]

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("## 2 · Your results")

ring_color = {"match": "#B83255", "healthy": "#0E7C7B", "no_match": "#0E7C7B", "ood": "#8A93A5", "unavailable": "#8A93A5"}[verdict]
ring_value = analysis["p_healthy"] if verdict == "healthy" else top
deg = int(360 * (ring_value or 0))
if verdict == "ood":
    tag = '<span class="st-tag tag-slate">Photo not recognised</span>'
    title = "This photo does not look like the scalp images the model knows"
    body = (
        "Its visual features are further from every training image than almost any genuine validation photo "
        f"(similarity {analysis['similarity']:.2f} against a minimum of {RES.ood_threshold:.2f}). So no condition is "
        "reported. Try a sharp, well-lit close-up of the scalp, with the hair parted so the skin is visible."
    )
elif verdict == "match":
    tag = '<span class="st-tag tag-rose">Potential concern detected</span>'
    title = f"Pattern resembling {html.escape(info.get('name', top_key))}"
    body = (
        f"{html.escape(info.get('summary', ''))} Its match score for this condition is "
        f"<b>{pct(top)}</b>, above the {pct(RES.threshold)} needed to report a condition."
    )
elif verdict == "healthy":
    tag = '<span class="st-tag tag-teal">Healthy-looking scalp</span>'
    title = "This scalp looks healthy to the model"
    body = (
        f"The model's score for a healthy scalp is <b>{pct(analysis['p_healthy'])}</b>, and every condition scores low "
        f"(highest: {html.escape(CONDITIONS.get(top_key, {}).get('name', top_key).split(' (')[0])} at {pct(top)}). "
        "The healthy class was learned from only a handful of photos, so treat this as a rough indication, not a check-up."
    )
elif verdict == "no_match":
    tag = '<span class="st-tag tag-teal">No condition matched</span>'
    title = "This photo does not match any of the ten conditions"
    body = (
        f"Every condition scored below {pct(RES.threshold)} (highest: {html.escape(info.get('name', top_key).split(' (')[0])} "
        f"at {pct(top)}). The photo is less typical of each condition than almost all of that condition's "
        "validation photos. This is what a healthy-looking scalp should give, but the model cannot confirm health."
    )
else:
    tag = '<span class="st-tag tag-slate">Classification unavailable</span>'
    title = "Classification could not be completed"
    body = html.escape(cls.get("reason", ""))

html_block(f"""
<div class="st-verdict {'no_match' if verdict == 'healthy' else verdict}">
  <div class="ring" style="background: conic-gradient({ring_color} {deg}deg, #EFECE5 {deg}deg);"><span>{pct(ring_value)}</span></div>
  <div>{tag}<h2>{title}</h2><p>{body}</p></div>
</div>""")

if RES.healthy_threshold is None:
    html_block("""
    <div class="st-note info"><b>About "healthy".</b> This model was trained only on photos of ten scalp
    conditions. It has no "healthy scalp" category, so it <b>cannot confirm that a scalp is healthy</b>. Each condition
    is scored separately (scores do not add up to 100%), so a low score for every condition is the closest it can come.</div>""")
else:
    html_block(f"""
    <div class="st-note info"><b>About "healthy".</b> The healthy-scalp class was learned from only
    {int(RES.report.get('healthy', {}).get('unique_photos', 0))} photos, against hundreds per condition, so it is a rough
    indication only. Each condition is scored separately (scores do not add up to 100%).</div>""")

if quality["flags"]:
    flags = ", ".join(f.replace("_", " ") for f in quality["flags"])
    html_block(f'<div class="st-note warn"><b>Photo quality warning:</b> {html.escape(flags)}. Results may be less reliable. '
               "A sharper, evenly lit close-up gives better results.</div>")
if r["faces"].get("n_faces"):
    html_block('<div class="st-note warn"><b>A face was detected.</b> ScalpTrack works best on close-ups of the scalp. '
               "Whole-head photos are analysed as a whole and may be less reliable.</div>")

hc, sw = r["hair_count"], r["shaft_width"]
width_txt = "—" if sw["median_px"] is None else f"{sw['median_px']:.1f} px"
html_block(f"""
<div class="st-stats">
  <div class="st-stat"><div class="st-label">{'Matched pattern' if verdict == 'match' else 'Result' if verdict == 'healthy' else 'Closest pattern'}</div><div class="v" style="font-size:1.1rem">{'Healthy-looking' if verdict == 'healthy' else html.escape(info.get('name', '—').split(' (')[0])}</div><div class="d">{f"healthy score {pct(analysis['p_healthy'])}" if verdict == 'healthy' else html.escape(info.get('category', ''))}</div></div>
  <div class="st-stat"><div class="st-label">Highest match</div><div class="v">{pct(top)}</div><div class="d">reported from {pct(RES.threshold)}</div></div>
  <div class="st-stat"><div class="st-label">Image quality</div><div class="v">{'Good' if quality['measurement_eligible'] and not quality['flags'] else 'Check'}</div><div class="d">sharpness {quality['laplacian_var']:.0f} (Laplacian var.)</div></div>
  <div class="st-stat"><div class="st-label">Hair crossings</div><div class="v">{hc['value'] if hc['value'] is not None else '—'}</div><div class="d">along the counting line · {html.escape(hc['status'])}</div></div>
</div>""")

tab_overview, tab_measure, tab_plan, tab_outlook, tab_doctor, tab_about = st.tabs(
    ["Overview", "Hair measurements", "Care plan", "What to expect", "When to see a doctor", "About this result"]
)

with tab_overview:
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        signs = bullet_list(info.get("signs", []))
        if verdict == "healthy":
            html_block(f"""
            <div class="st-card"><div class="st-label">Result</div><h3>Healthy-looking scalp</h3>
            <p style="color:var(--ink-soft);line-height:1.6">The photo looks more like the healthy-scalp photos than like any of the
            ten conditions (healthy score {pct(analysis['p_healthy'])}). Keep a gentle routine, and re-check if you notice itching,
            flaking, redness or hair loss.</p>{bullet_list(GENERAL_ROUTINE["daily"][:3])}</div>""")
        elif info:
            html_block(f"""
        <div class="st-card"><div class="st-label">{html.escape(info.get('category', ''))}</div>
        <h3>{html.escape(info.get('name', top_key))}</h3>
        <p style="color:var(--ink-soft);line-height:1.6">{html.escape(info.get('summary', ''))}</p>
        <div class="st-label" style="margin-top:14px">Typical signs</div>{signs}</div>""")
        else:
            html_block("""<div class="st-card"><h3>No condition reported</h3><p style="color:var(--ink-soft);line-height:1.6">
            There is no condition-specific information for this photo. If you have symptoms, a dermatologist can assess them.</p></div>""")
    with c2:
        bars = "".join(
            f'<div class="st-bar {"top" if i == 0 else ""}"><div class="row"><span>{html.escape(CONDITIONS.get(k, {}).get("name", k).split(" (")[0])}</span>'
            f'<span>{pct(v)}</span></div><div class="track"><div class="fill" style="width:{100 * v:.1f}%"></div></div></div>'
            for i, (k, v) in enumerate(scores.items())
        )
        html_block(f"""<div class="st-card"><h3>Match score for every condition</h3>
        <div class="sub">Each condition is scored on its own, so these do not add up to 100%. A score of 50% means the photo is as
        typical of that condition as an average validation photo of it. A condition is reported only from {pct(RES.threshold)}.
        These are not probabilities of having the condition.</div>{bars}</div>""")

with tab_measure:
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.image(analysis["overlay"], caption="Rose: hair candidates · Cyan: bright flake candidates · Gold: counting line", width="stretch")
    with c2:
        cov = r["coverage"]
        rows = [
            ("Hair crossings on the counting line", hc["value"] if hc["value"] is not None else "unavailable", hc["status"]),
            ("Separate hair-like strands", r["hair_components"]["value"] if r["hair_components"]["value"] is not None else "unavailable", r["hair_components"]["status"]),
            ("Apparent shaft width", width_txt, sw["status"]),
            ("Shaft width in micrometres", "unavailable", sw.get("physical_status", "unavailable")),
            ("Hair-covered area", pct(cov.get("hair_coverage_fraction")), "exploratory"),
            ("Visible scalp area", pct(cov.get("visible_scalp_fraction")), "exploratory"),
            ("Bright flake-candidate area", pct(cov.get("flake_candidate_coverage")), "exploratory"),
        ]
        table = "".join(
            f'<div class="st-bar"><div class="row"><span>{html.escape(a)}</span><span><b style="color:var(--ink)">{html.escape(str(b))}</b> · {html.escape(c)}</span></div></div>'
            for a, b, c in rows
        )
        html_block(f"""<div class="st-card"><h3>Measurements from this photo</h3>
        <div class="sub">Computed automatically from the image in pixels. Without a ruler or calibration in the photo there is no
        physical scale, so these are relative, exploratory values, not clinical measurements.</div>{table}</div>""")
        html_block("""<div class="st-note info"><b>How to read this.</b> "Crossings" counts hair-like strands crossing one line
        drawn across the photo. It is not your total hair count or density. Flake candidates are bright spots, and glare can look
        the same. These values are most useful for comparing photos of the same area taken the same way over time.</div>""")

with tab_plan:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if verdict == "match" and info:
            html_block(f"""<div class="st-card"><div class="st-label">For a pattern like {html.escape(info['name'].split(' (')[0])}</div>
            <h3>Care suggestions</h3>{bullet_list(info['care'])}</div>""")
        else:
            html_block("""<div class="st-card"><h3>Care suggestions</h3><div class="sub">No condition matched, so these are
            general scalp-care basics rather than condition-specific advice.</div>""" + bullet_list(GENERAL_ROUTINE["daily"][:3]) + "</div>")
        html_block(f'<div class="st-card"><h3>Diet & nutrition</h3><div class="sub">General guidance for healthy hair, not a treatment.</div>{bullet_list(GENERAL_ROUTINE["nutrition"])}</div>')
    with c2:
        html_block(f'<div class="st-card"><div class="st-label">Suggested routine</div><h3>Daily</h3>{bullet_list(GENERAL_ROUTINE["daily"])}</div>')
        html_block(f'<div class="st-card"><h3>Weekly</h3>{bullet_list(GENERAL_ROUTINE["weekly"])}</div>')

with tab_outlook:
    expectation = info.get("expectations", "") if verdict == "match" else (
        "Without a matched condition there is no condition-specific outlook. If symptoms are present, a clinician's "
        "assessment is the reliable next step."
    )
    html_block(f"""<div class="st-card"><h3>What you may reasonably expect</h3>
    <div class="sub">Typical courses vary widely between people. Nothing here is a promise of results.</div>
    <div class="st-timeline">
      <div class="st-step"><b>Now</b><span>Start the gentle routine, and stop any product that seems to irritate your scalp.
      Take a baseline photo of the same area, in the same light.</span></div>
      <div class="st-step"><b>Typical course</b><span>{html.escape(expectation)}</span></div>
      <div class="st-step"><b>Every 4–6 weeks</b><span>Re-photograph the same area the same way and compare.
      Re-run this analysis if you find it helpful, but judge change by your symptoms and photos, not by the score alone.</span></div>
      <div class="st-step"><b>If things are not improving</b><span>Stop self-care experiments and book an appointment
      with a dermatologist.</span></div>
    </div></div>""")

with tab_doctor:
    urgency = info.get("urgency", "If symptoms persist") if verdict == "match" else "If you have symptoms"
    advice = info.get("doctor", "") if verdict == "match" else "See a doctor or dermatologist if you have scalp symptoms, whatever this tool says."
    html_block(f"""<div class="st-card"><div class="st-label">Dermatologist visit</div>
    <h3>{html.escape(urgency)}</h3><p style="color:var(--ink-soft);line-height:1.6">{html.escape(advice)}</p></div>""")
    html_block(f'<div class="st-card"><h3>See a doctor promptly if you notice</h3>{bullet_list(RED_FLAGS, warn=True)}</div>')

with tab_about:
    tm = RES.test_metrics or {}
    html_block(f"""<div class="st-card"><h3>How this result was produced</h3>
    <p style="color:var(--ink-soft);line-height:1.6">Each photo is turned into a 1280-number visual description by a
    MobileNetV2 network pretrained on ImageNet. A linear classifier trained on the public Kaggle "Hair Diseases" dataset
    then scores {html.escape(str(len(RES.bundle['classes'])))} conditions. The training images were checked for duplicates,
    and copies of the same photo were kept together in one split. Two checks calibrated on held-out validation images
    protect the result: the photo must resemble the training images, and a condition is reported only if its
    own match score (how typical the photo is of that condition, compared with its validation photos) reaches
    {pct(RES.threshold)}.</p>
    <div class="st-label" style="margin-top:12px">Limitations you should know</div>
    {bullet_list([
        f"On held-out test photos kept apart from training by copy group, the model scored {pct(tm.get('accuracy'))} accuracy. "
        "Real-world photos from other cameras and people were not part of that test, so expect lower accuracy on them.",
        "The labels come from the dataset publisher, not from clinical diagnosis, and some conditions look very similar.",
        "The healthy-scalp class was learned from very few user-added photos, so it is a rough indication and cannot confirm health." if RES.healthy_threshold is not None else "There is no 'healthy scalp' category, so the model cannot confirm that a scalp is healthy.",
        "Photos unlike the training images (other lighting, other skin and hair types, whole-head shots) may give unreliable results.",
        "Hair measurements are in pixels. Physical units such as micrometres or hairs per cm² need calibrated imaging.",
        "This is an educational research project, not a medical device. It does not diagnose, and its guidance is general.",
    ], warn=True)}</div>""")

html_block("""
<div class="st-footer">ScalpTrack is an experimental computer-vision project. It is not a medical device, does not provide a
diagnosis and does not replace professional advice. Care suggestions are general information. Always consult a qualified healthcare
professional about any health concern. Photos are processed in memory and are not stored.</div>""")
