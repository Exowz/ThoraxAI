"""
ThoraxAI -- Detection de pneumonie sur radiographies thoraciques.

Usage:
    uv run streamlit run app.py
"""

import random
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from app.styles import get_css
from app.components import (
    MODEL_NAMES,
    MODEL_DISPLAY,
    render_header,
    render_control_bar,
    render_result_banner,
    render_prob_bar,
    render_stat_card,
    render_compare_card,
    render_compare_unavailable,
    render_history_item,
)
from app.inference import load_model, predict, make_gradcam
from app.data import load_results, samples_available, list_samples


def run():
    st.set_page_config(
        page_title="ThoraxAI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(get_css(), unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None
    if "selected_sample_label" not in st.session_state:
        st.session_state.selected_sample_label = None

    # === HEADER ===
    render_header()

    # === CONTROL BAR ===
    primary_model, threshold, show_cam, compare_mode = render_control_bar()

    model, device, loaded, epoch = load_model(primary_model)
    if not loaded:
        st.error(f"{MODEL_DISPLAY[primary_model]} non trouve. Lancez le notebook 02 d'abord.")
        st.stop()

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # === INPUT MODE ===
    has_samples = samples_available()

    if has_samples:
        input_mode = st.radio(
            "Source de l'image",
            ["Upload", "Exemples"],
            horizontal=True,
            label_visibility="collapsed",
        )
    else:
        input_mode = "Upload"

    uploaded = None
    ground_truth_label = None

    if input_mode == "Upload":
        st.session_state.selected_sample = None
        st.session_state.selected_sample_label = None
        uploaded = st.file_uploader(
            "Deposer une radiographie thoracique",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    else:
        _render_samples_panel()
        if st.session_state.selected_sample:
            ground_truth_label = st.session_state.selected_sample_label

    # === LANDING STATE ===
    has_image = uploaded or st.session_state.selected_sample

    if not has_image:
        _render_landing()
        st.stop()

    # === ANALYSIS ===
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_name = uploaded.name
    else:
        image = Image.open(st.session_state.selected_sample).convert("RGB")
        image_name = Path(st.session_state.selected_sample).name

    prob = predict(model, image, device)
    prediction = "PNEUMONIA" if prob > threshold else "NORMAL"
    confidence = prob if prediction == "PNEUMONIA" else 1 - prob

    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "file": image_name,
        "model": MODEL_DISPLAY[primary_model],
        "prediction": prediction,
        "probability": prob,
    })

    # === TABS ===
    tab_names = (
        ["Analyse", "Comparaison", "Grad-CAM", "Historique"]
        if compare_mode
        else ["Analyse", "Grad-CAM", "Historique"]
    )
    tabs = st.tabs(tab_names)

    # --- TAB: Analyse ---
    with tabs[0]:
        left, right = st.columns([1.1, 1])
        with left:
            st.image(image, use_container_width=True)
        with right:
            render_result_banner(prediction, confidence, ground_truth_label)
            render_prob_bar(prob, prediction)

            m1, m2, m3 = st.columns(3)
            with m1:
                render_stat_card("Probabilite", f"{prob:.4f}", font_size="22px")
            with m2:
                render_stat_card("Seuil", f"{threshold:.2f}", font_size="22px")
            with m3:
                render_stat_card(
                    "Modele",
                    MODEL_DISPLAY[primary_model],
                    color_class="c-blue",
                    font_size="15px",
                    detail=f"epoque {epoch}",
                    extra_style="letter-spacing:0",
                )

    # --- TAB: Comparaison ---
    if compare_mode:
        with tabs[1]:
            cols = st.columns(4)
            preds_all = []
            for i, name in enumerate(MODEL_NAMES):
                mc, dc, ok, _ = load_model(name)
                if ok:
                    p = predict(mc, image, dc)
                    pred = "PNEUMONIA" if p > threshold else "NORMAL"
                    preds_all.append(pred)
                    with cols[i]:
                        render_compare_card(name, MODEL_DISPLAY[name], pred, p,
                                            is_primary=(name == primary_model))
                else:
                    with cols[i]:
                        render_compare_unavailable(MODEL_DISPLAY[name])

            if preds_all:
                pc_count = preds_all.count("PNEUMONIA")
                total = len(preds_all)
                cons = "PNEUMONIA" if pc_count > total / 2 else "NORMAL"
                cons_c = "#f87171" if cons == "PNEUMONIA" else "#34d399"
                st.markdown(f"""
                <div class="consensus-bar">
                    <div class="stat-label">Consensus ({pc_count}/{total} modeles)</div>
                    <p class="stat-value" style="color:{cons_c};font-size:26px">{cons}</p>
                </div>
                """, unsafe_allow_html=True)

    # --- TAB: Grad-CAM ---
    gcam_idx = 2 if compare_mode else 1
    with tabs[gcam_idx]:
        if show_cam:
            orig, hm, overlay = make_gradcam(model, image, device)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original**")
                st.image(orig, use_container_width=True)
            with c2:
                st.markdown("**Heatmap**")
                st.image(hm, use_container_width=True)
            with c3:
                st.markdown("**Superposition**")
                st.image(overlay, use_container_width=True)

            st.markdown("""
            <div class="warn-box">
                Les cartes de chaleur analysent le comportement du modele, pas les lesions cliniques.
                Cet outil est un projet academique et ne remplace en aucun cas un diagnostic medical professionnel.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p class="empty-state">Activez Grad-CAM dans la barre de controle</p>',
                unsafe_allow_html=True,
            )

    # --- TAB: Historique ---
    hist_idx = 3 if compare_mode else 2
    with tabs[hist_idx]:
        if st.session_state.history:
            for item in st.session_state.history[:25]:
                render_history_item(item)
            st.markdown("")
            if st.button("Effacer l'historique"):
                st.session_state.history = []
                st.rerun()
        else:
            st.markdown(
                '<p class="empty-state">Aucune analyse effectuee</p>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_samples_panel():
    """Render the sample image selection panel (Exemples mode)."""
    normals = list_samples("NORMAL")
    pneumonias = list_samples("PNEUMONIA")

    # Build combined options for selectbox
    all_options: list[tuple[str, str]] = []  # (display_name, path)
    for p in normals:
        all_options.append((f"NORMAL / {p.name}", str(p), "NORMAL"))
    for p in pneumonias:
        all_options.append((f"PNEUMONIA / {p.name}", str(p), "PNEUMONIA"))

    btn_col, sel_col = st.columns([1, 3])

    with btn_col:
        if st.button("Image aleatoire"):
            pick = random.choice(all_options)
            st.session_state.selected_sample = pick[1]
            st.session_state.selected_sample_label = pick[2]
            st.rerun()

    with sel_col:
        display_names = [o[0] for o in all_options]
        # Find current index
        current_idx = 0
        if st.session_state.selected_sample:
            for i, o in enumerate(all_options):
                if o[1] == st.session_state.selected_sample:
                    current_idx = i
                    break

        choice = st.selectbox(
            "Choisir un exemple",
            range(len(all_options)),
            index=current_idx,
            format_func=lambda i: display_names[i],
            label_visibility="collapsed",
        )

        if all_options:
            selected = all_options[choice]
            st.session_state.selected_sample = selected[1]
            st.session_state.selected_sample_label = selected[2]

    # Preview thumbnails in two columns
    col_n, col_p = st.columns(2)

    shown_normals = random.Random(42).sample(normals, min(5, len(normals)))
    shown_pneumonias = random.Random(42).sample(pneumonias, min(5, len(pneumonias)))

    with col_n:
        st.markdown('<div class="sample-section-title">Normal</div>', unsafe_allow_html=True)
        thumb_cols = st.columns(min(5, len(shown_normals)))
        for j, img_path in enumerate(shown_normals):
            with thumb_cols[j]:
                st.image(str(img_path), use_container_width=True)

    with col_p:
        st.markdown('<div class="sample-section-title">Pneumonia</div>', unsafe_allow_html=True)
        thumb_cols = st.columns(min(5, len(shown_pneumonias)))
        for j, img_path in enumerate(shown_pneumonias):
            with thumb_cols[j]:
                st.image(str(img_path), use_container_width=True)


def _render_landing():
    """Show model stats and empty-state message when no image is selected."""
    rd = load_results()
    if rd and "models" in rd:
        cols = st.columns(4)
        for i, name in enumerate(MODEL_NAMES):
            if name in rd["models"]:
                m = rd["models"][name]
                cc = "c-green" if m["auc_roc"] > 0.96 else "c-blue"
                with cols[i]:
                    render_stat_card(
                        MODEL_DISPLAY[name],
                        f"{m['auc_roc']:.3f}",
                        color_class=cc,
                        detail=f"F1 {m['f1']:.3f} &middot; Recall {m['recall']:.3f} &middot; Spec {m['specificity']:.3f}",
                    )

    st.markdown(
        '<p class="empty-state">Deposez une radiographie ou selectionnez un exemple pour lancer l\'analyse</p>',
        unsafe_allow_html=True,
    )
