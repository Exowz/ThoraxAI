"""All CSS styles for the ThoraxAI Streamlit app."""


def get_css() -> str:
    return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global reset */
    *, *::before, *::after { box-sizing: border-box; }

    .stApp {
        font-family: 'DM Sans', sans-serif;
        background: #060a13;
    }

    /* Kill sidebar completely */
    section[data-testid="stSidebar"] { display: none; }
    button[data-testid="stSidebarCollapsedControl"] { display: none; }
    #MainMenu { display: none; }
    footer { display: none; }
    .stDeployButton { display: none; }
    header[data-testid="stHeader"] { display: none; }

    /* Main container breathing room */
    .block-container {
        padding-top: 24px;
        padding-bottom: 24px;
        max-width: 1280px;
    }

    /* ---- HEADER ---- */
    .app-header {
        background: linear-gradient(135deg, #0c1222 0%, #151d33 60%, #0e1628 100%);
        border: 1px solid rgba(80, 120, 200, 0.12);
        border-radius: 18px;
        padding: 32px 40px 28px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .app-header::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 40%;
        height: 100%;
        background: radial-gradient(ellipse at 80% 40%, rgba(56, 138, 221, 0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .logo-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }
    .logo-mark {
        width: 36px; height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 16px; color: #fff;
        letter-spacing: -0.5px;
        flex-shrink: 0;
    }
    .logo-text {
        font-size: 24px;
        font-weight: 700;
        color: #e8ecf4;
        letter-spacing: -0.8px;
    }
    .header-desc {
        font-size: 13.5px;
        color: #6b7a94;
        margin: 2px 0 0 0;
        line-height: 1.5;
    }
    .tag-row { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    .tag {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.3px;
        background: rgba(37, 99, 235, 0.1);
        color: #6b9aed;
        border: 1px solid rgba(37, 99, 235, 0.15);
    }
    .tag-authors {
        position: absolute;
        right: 40px;
        bottom: 16px;
        font-size: 11px;
        color: #3d4e66;
        text-align: right;
    }

    /* ---- CONTROL BAR ---- */
    .control-bar {
        background: #0c1222;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 20px;
    }

    /* ---- STAT CARDS ---- */
    .stat-card {
        background: linear-gradient(160deg, #0f1729, #0c1222);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 20px 22px;
        transition: border-color 0.25s;
    }
    .stat-card:hover { border-color: rgba(80,120,200,0.18); }
    .stat-label {
        font-size: 10.5px;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        color: #475569;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 30px;
        font-weight: 600;
        letter-spacing: -1px;
        margin: 0;
        line-height: 1;
    }
    .stat-detail {
        font-size: 11.5px;
        color: #475569;
        margin-top: 6px;
    }
    .c-green { color: #34d399; }
    .c-red { color: #f87171; }
    .c-blue { color: #60a5fa; }
    .c-white { color: #e2e8f0; }

    /* ---- RESULT BANNER ---- */
    .result-banner {
        border-radius: 14px;
        padding: 22px 28px;
        margin: 8px 0 16px;
        border: 1px solid;
    }
    .result-banner.normal {
        background: linear-gradient(145deg, rgba(16,65,48,0.35), rgba(6,35,25,0.5));
        border-color: rgba(52,211,153,0.2);
    }
    .result-banner.pneumonia {
        background: linear-gradient(145deg, rgba(80,20,20,0.35), rgba(50,10,10,0.5));
        border-color: rgba(248,113,113,0.2);
    }
    .rb-label {
        font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
        font-weight: 600; margin-bottom: 2px;
    }
    .rb-pred { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }
    .rb-conf { font-size: 13px; opacity: 0.65; margin-top: 2px; }

    /* ---- COMPARE CARD ---- */
    .cmp-card {
        background: #0c1222;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .cmp-card.active { border-color: rgba(37,99,235,0.35); }
    .cmp-name { font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 10px; }
    .cmp-prob {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px; font-weight: 500; margin: 6px 0;
    }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 6px;
        font-size: 10.5px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-n { background: rgba(52,211,153,0.12); color: #34d399; }
    .badge-p { background: rgba(248,113,113,0.12); color: #f87171; }

    /* ---- CONSENSUS ---- */
    .consensus-bar {
        background: #0c1222;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 18px 24px;
        text-align: center;
        margin-top: 14px;
    }

    /* ---- HISTORY ---- */
    .hist-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 16px; border-radius: 10px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.03);
        margin: 5px 0;
    }
    .hist-meta { display: flex; align-items: center; gap: 12px; }
    .hist-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px; color: #475569;
    }
    .hist-file { font-size: 13px; color: #cbd5e1; font-weight: 500; }
    .hist-model { font-size: 11px; color: #475569; }

    /* ---- DISCLAIMER ---- */
    .warn-box {
        background: rgba(234,179,8,0.06);
        border: 1px solid rgba(234,179,8,0.12);
        border-radius: 12px;
        padding: 13px 20px;
        font-size: 12.5px; color: #b89830;
        margin-top: 18px;
        line-height: 1.55;
    }

    /* ---- MISC ---- */
    .sep {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(80,120,200,0.12), transparent);
        margin: 18px 0;
    }
    .empty-state {
        text-align: center; color: #3d4e66; padding: 48px 0;
        font-size: 14px;
    }

    /* Fix Streamlit upload widget look */
    .stFileUploader > div {
        border-radius: 14px;
        border: 2px dashed rgba(80,120,200,0.2);
        background: rgba(80,120,200,0.02);
    }
    .stFileUploader > div:hover {
        border-color: rgba(80,120,200,0.35);
        background: rgba(80,120,200,0.04);
    }

    /* Selectbox & slider restyle */
    .stSelectbox > div > div { border-radius: 10px; }
    .stSlider > div > div > div { border-radius: 10px; }

    /* ---- SAMPLE THUMBNAILS ---- */
    .sample-section-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        color: #475569;
        margin-bottom: 8px;
    }

    /* ---- PROBABILITY BAR ---- */
    .prob-bar-container {
        background: #0f1729;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 14px 20px;
        margin: 10px 0 6px;
    }
    .prob-bar-track {
        width: 100%;
        height: 10px;
        background: #1a2238;
        border-radius: 5px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.4s ease;
    }
    .prob-bar-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        margin-top: 6px;
        text-align: right;
    }

    /* ---- GROUND TRUTH BADGE ---- */
    .gt-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 10px;
    }
    .gt-correct {
        background: rgba(52,211,153,0.12);
        color: #34d399;
        border: 1px solid rgba(52,211,153,0.2);
    }
    .gt-wrong {
        background: rgba(248,113,113,0.12);
        color: #f87171;
        border: 1px solid rgba(248,113,113,0.2);
    }
</style>
"""
