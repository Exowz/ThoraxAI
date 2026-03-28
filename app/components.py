"""Reusable Streamlit HTML components."""

import streamlit as st

MODEL_NAMES = ["cnn_baseline", "resnet18", "densenet121", "efficientnet"]
MODEL_DISPLAY = {
    "cnn_baseline": "CNN Baseline",
    "resnet18": "ResNet-18",
    "densenet121": "DenseNet-121",
    "efficientnet": "EfficientNet-B0",
}
TAG_LABELS = ["PyTorch", "Transfer Learning", "Grad-CAM", "Grad-CAM++", "K-Fold"]


def render_header():
    tags_html = "".join(f'<span class="tag">{t}</span>' for t in TAG_LABELS)
    st.markdown(f"""
    <div class="app-header">
        <div class="logo-row">
            <div class="logo-mark">Tx</div>
            <span class="logo-text">ThoraxAI</span>
        </div>
        <p class="header-desc">
            Systeme d'aide a la detection de pneumonie sur radiographies thoraciques
        </p>
        <div class="tag-row">{tags_html}</div>
        <div class="tag-authors">ECE Paris 2026<br>M.K.E. Kapoor & T.M. Rakotomalala</div>
    </div>
    """, unsafe_allow_html=True)


def render_control_bar():
    ctl1, ctl2, ctl3, ctl4 = st.columns([2, 2, 1.5, 1.5])
    with ctl1:
        primary_model = st.selectbox(
            "Modele",
            MODEL_NAMES,
            index=2,
            format_func=lambda x: MODEL_DISPLAY[x],
        )
    with ctl2:
        threshold = st.slider("Seuil", 0.10, 0.95, 0.50, 0.05)
    with ctl3:
        show_cam = st.toggle("Grad-CAM", value=True)
    with ctl4:
        compare_mode = st.toggle("Multi-modeles", value=False)
    return primary_model, threshold, show_cam, compare_mode


def render_result_banner(prediction: str, confidence: float, ground_truth_label: str | None):
    cls = "normal" if prediction == "NORMAL" else "pneumonia"
    clr = "#34d399" if prediction == "NORMAL" else "#f87171"

    gt_html = ""
    if ground_truth_label:
        is_correct = prediction == ground_truth_label
        gt_cls = "gt-correct" if is_correct else "gt-wrong"
        gt_icon = "\u2713" if is_correct else "\u2717"
        gt_html = f'<span class="gt-badge {gt_cls}">{gt_icon} Vrai label : {ground_truth_label}</span>'

    st.markdown(f"""
    <div class="result-banner {cls}">
        <div class="rb-label" style="color:{clr}">Diagnostic</div>
        <div class="rb-pred" style="color:{clr}">{prediction} {gt_html}</div>
        <div class="rb-conf" style="color:{clr}">Confiance {confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)


def render_prob_bar(prob: float, prediction: str):
    bar_color = "#f87171" if prediction == "PNEUMONIA" else "#34d399"
    bar_pct = prob * 100
    st.markdown(f"""
    <div class="prob-bar-container">
        <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:{bar_pct:.1f}%;background:{bar_color}"></div>
        </div>
        <div class="prob-bar-label" style="color:{bar_color}">P(pneumonie) : {bar_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_card(label: str, value: str, color_class: str = "c-white",
                     detail: str = "", font_size: str = "30px", extra_style: str = ""):
    detail_html = f'<div class="stat-detail">{detail}</div>' if detail else ""
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <p class="stat-value {color_class}" style="font-size:{font_size};{extra_style}">{value}</p>
        {detail_html}
    </div>
    """, unsafe_allow_html=True)


def render_compare_card(name: str, display_name: str, pred: str, prob: float,
                        is_primary: bool = False):
    bc = "badge-p" if pred == "PNEUMONIA" else "badge-n"
    pc = "#f87171" if pred == "PNEUMONIA" else "#34d399"
    active = " active" if is_primary else ""
    conf = prob if pred == "PNEUMONIA" else 1 - prob
    st.markdown(f"""
    <div class="cmp-card{active}">
        <div class="cmp-name">{display_name}</div>
        <span class="badge {bc}">{pred}</span>
        <div class="cmp-prob" style="color:{pc}">{prob:.4f}</div>
        <div style="font-size:11px;color:#475569">Confiance {conf:.1%}</div>
    </div>
    """, unsafe_allow_html=True)


def render_compare_unavailable(display_name: str):
    st.markdown(f"""
    <div class="cmp-card">
        <div class="cmp-name">{display_name}</div>
        <div style="color:#475569;padding:20px 0;font-size:13px">Non disponible</div>
    </div>
    """, unsafe_allow_html=True)


def render_history_item(item: dict):
    pc = "#f87171" if item["prediction"] == "PNEUMONIA" else "#34d399"
    bc = "badge-p" if item["prediction"] == "PNEUMONIA" else "badge-n"
    st.markdown(f"""
    <div class="hist-row">
        <div class="hist-meta">
            <span class="hist-time">{item['time']}</span>
            <span class="hist-file">{item['file']}</span>
            <span class="hist-model">{item['model']}</span>
        </div>
        <div style="display:flex;align-items:center;gap:12px">
            <span style="font-family:'JetBrains Mono',monospace;font-size:14px;color:{pc}">{item['probability']:.4f}</span>
            <span class="badge {bc}">{item['prediction']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
