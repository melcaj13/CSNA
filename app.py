import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import random
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  SAYFA YAPIKANDIRMASI
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dijital İtibar Risk Modeli",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  GLOBAL STİL
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Genel ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e8ecf4;
}
section[data-testid="stSidebar"] {
    background: #0d1221;
    border-right: 1px solid #1e2640;
}
section[data-testid="stSidebar"] * {
    color: #c8d0e8 !important;
}

/* ── Başlık alanı ── */
.hero-block {
    background: linear-gradient(135deg, #0d1221 0%, #111827 50%, #0d1221 100%);
    border: 1px solid #1e2a4a;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,102,241,.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-block::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(16,185,129,.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: #ffffff;
    margin: 0 0 6px;
    letter-spacing: -.5px;
}
.hero-title span { color: #818cf8; }
.hero-sub {
    font-size: .9rem;
    color: #64748b;
    margin: 0;
    font-weight: 400;
}
.badge {
    display: inline-block;
    background: rgba(99,102,241,.15);
    border: 1px solid rgba(99,102,241,.35);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: .75rem;
    font-weight: 500;
    margin-bottom: 12px;
    letter-spacing: .5px;
}

/* ── KPI Kartları ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1e2640;
    border-radius: 14px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.kpi-card:hover { border-color: #3b4c7a; }
.kpi-card .accent-bar {
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 14px 0 0 14px;
}
.kpi-label {
    font-size: .78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .8px;
    font-weight: 600;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f1f5f9;
    line-height: 1;
}
.kpi-delta {
    font-size: .78rem;
    margin-top: 6px;
}

/* ── Bölüm başlıkları ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px;
}
.section-header h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #e2e8f0;
    margin: 0;
}
.section-divider {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e2640, transparent);
}

/* ── Tablo ── */
.risk-table-wrapper {
    background: #111827;
    border: 1px solid #1e2640;
    border-radius: 14px;
    overflow: hidden;
}
.dataframe { background: transparent !important; }
thead tr th {
    background: #0d1221 !important;
    color: #818cf8 !important;
    font-size: .78rem !important;
    letter-spacing: .6px !important;
    text-transform: uppercase !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid #1e2640 !important;
}
tbody tr td {
    color: #cbd5e1 !important;
    font-size: .85rem !important;
    padding: 10px 16px !important;
    border-bottom: 1px solid #0f172a !important;
}
tbody tr:hover td { background: #1a2235 !important; }

/* ── Info / Warning kutuları ── */
.info-box {
    background: rgba(99,102,241,.08);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: .85rem;
    color: #a5b4fc;
    margin-bottom: 20px;
}
.warn-box {
    background: rgba(245,158,11,.07);
    border: 1px solid rgba(245,158,11,.25);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: .85rem;
    color: #fcd34d;
    margin-bottom: 20px;
}

/* ── Sidebar ── */
.sidebar-section {
    background: #111827;
    border: 1px solid #1e2640;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
.sidebar-title {
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: #475569 !important;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Plotly arka plan uyumu */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  ÖRNEK VERİ ÜRETİCİ
# ─────────────────────────────────────────────────────────
NEGATIVE_TEMPLATES = [
    "The service was terrible and I will never come back.",
    "Absolutely awful experience, food was cold and staff rude.",
    "Very disappointing, worst meal I've had in years.",
    "Staff was completely unprofessional and ignored us.",
    "Food took forever and tasted horrible, total waste of money.",
    "I can't believe how bad this place has gotten.",
    "Do not eat here, got sick after the meal.",
    "Management doesn't care about customers at all.",
    "The place was dirty and the food was inedible.",
    "Zero stars if I could, absolutely the worst.",
    "Extremely rude behavior from the owner.",
    "Never in my life have I had such poor service.",
]
POSITIVE_TEMPLATES = [
    "Great food and excellent service, highly recommend!",
    "One of the best dining experiences I've had.",
    "Amazing atmosphere and delicious meals.",
    "Staff was very friendly and attentive.",
    "Food was fresh and portions were generous.",
    "Will definitely come back, loved everything.",
    "Perfect spot for a family dinner.",
    "Outstanding quality for the price.",
    "Cozy place with fantastic cocktails.",
    "Highly recommended, everything was perfect.",
]
NEUTRAL_TEMPLATES = [
    "It was okay, nothing special but not bad either.",
    "Average food, decent prices.",
    "Service was fine, food was acceptable.",
    "Not bad, not great, just average.",
    "Typical restaurant experience.",
]

RESTORAN_ISIMLERI = [
    "La Maison Philly", "Rocky's Steakhouse", "Philly Bites", "The Corner Table",
    "Broad Street Grill", "Liberty Eats", "Ben's Diner", "Old City Kitchen",
    "Fishtown Flavors", "South Philly Bistro",
]
SEMTLER = [
    "Center City", "Old City", "Fishtown", "South Philly", "Northern Liberties",
    "Rittenhouse", "Manayunk", "Chestnut Hill", "Germantown", "West Philly",
]


@st.cache_data
def ornek_veri_olustur(n=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    satirlar = []
    for i in range(1, n + 1):
        r = random.random()
        if r < 0.35:
            metin = random.choice(NEGATIVE_TEMPLATES)
            puan = random.choice([1, 2])
        elif r < 0.60:
            metin = random.choice(POSITIVE_TEMPLATES)
            puan = random.choice([4, 5])
        else:
            metin = random.choice(NEUTRAL_TEMPLATES)
            puan = 3

        satirlar.append({
            "Kullanici_ID": f"USR_{i:04d}",
            "Restoran": random.choice(RESTORAN_ISIMLERI),
            "Semt": random.choice(SEMTLER),
            "Yorum_Metni": metin,
            "Yildiz_Puani": puan,
            "Arkadas_Sayisi": int(np.random.exponential(scale=40) + 1),
            "Yorum_Sayisi": int(np.random.exponential(scale=20) + 1),
            "Faydali_Oy": int(np.random.exponential(scale=15)),
        })

    return pd.DataFrame(satirlar)


# ─────────────────────────────────────────────────────────
#  ANALİZ FONKSİYONLARI
# ─────────────────────────────────────────────────────────
def nlp_analizi(df: pd.DataFrame) -> pd.DataFrame:
    """TextBlob ile duygu skoru hesapla."""
    df = df.copy()
    if "Yorum_Metni" in df.columns:
        def polarity(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except Exception:
                return 0.0
        df["Duygu_Skoru"] = df["Yorum_Metni"].apply(polarity)
    else:
        df["Duygu_Skoru"] = 0.0
    return df


def sna_analizi(df: pd.DataFrame) -> pd.DataFrame:
    """Arkadaş sayısı üzerinden normalize edilmiş merkezilik skoru."""
    df = df.copy()
    kaynak_sutun = None
    for col in ["Arkadas_Sayisi", "Arkadaş_Sayısı", "Arkadaslar", "Friend_Count", "friends"]:
        if col in df.columns:
            kaynak_sutun = col
            break

    if kaynak_sutun:
        vals = pd.to_numeric(df[kaynak_sutun], errors="coerce").fillna(0)
        mn, mx = vals.min(), vals.max()
        df["Merkezilik"] = (vals - mn) / (mx - mn + 1e-9)
    else:
        df["Merkezilik"] = np.random.uniform(0.05, 0.95, len(df))

    return df


def risk_skoru_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Duygu_Siddeti"] = df["Duygu_Skoru"].abs()
    df["Risk_Skoru"] = df["Duygu_Siddeti"] * df["Merkezilik"]
    df["Risk_Skoru"] = (df["Risk_Skoru"] - df["Risk_Skoru"].min()) / \
                       (df["Risk_Skoru"].max() - df["Risk_Skoru"].min() + 1e-9)
    df["Risk_Seviyesi"] = pd.cut(
        df["Risk_Skoru"],
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["🟢 Düşük", "🟡 Orta", "🔴 Kritik"],
    )
    return df


def tam_analiz(df: pd.DataFrame) -> pd.DataFrame:
    df = nlp_analizi(df)
    df = sna_analizi(df)
    df = risk_skoru_hesapla(df)
    return df


# ─────────────────────────────────────────────────────────
#  GÖRSELLEŞTİRME YARDIMCILARI
# ─────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8"),
    margin=dict(l=16, r=16, t=36, b=16),
    xaxis=dict(
        gridcolor="#1e2640", zeroline=False, tickfont=dict(size=11),
        linecolor="#1e2640",
    ),
    yaxis=dict(
        gridcolor="#1e2640", zeroline=False, tickfont=dict(size=11),
        linecolor="#1e2640",
    ),
)


def risk_matrisi_ciz(df: pd.DataFrame) -> go.Figure:
    renk_map = {
        "🟢 Düşük": "#10b981",
        "🟡 Orta":  "#f59e0b",
        "🔴 Kritik": "#ef4444",
    }

    fig = go.Figure()

    # Kritik bölge arka planı
    fig.add_shape(
        type="rect",
        x0=0.5, y0=0.5, x1=1.05, y1=1.05,
        fillcolor="rgba(239,68,68,.07)",
        line=dict(color="rgba(239,68,68,.3)", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=0.75, y=0.92,
        text="⚠️ KRİTİK RİSK BÖLGESİ",
        showarrow=False,
        font=dict(size=10, color="#ef4444", family="DM Sans"),
        bgcolor="rgba(239,68,68,.1)",
        bordercolor="rgba(239,68,68,.3)",
        borderwidth=1,
        borderpad=5,
    )

    for seviye, renk in renk_map.items():
        alt = df[df["Risk_Seviyesi"] == seviye]
        if alt.empty:
            continue
        id_col = "Kullanici_ID" if "Kullanici_ID" in df.columns else df.index.astype(str)
        hover_text = (
            alt.get("Kullanici_ID", pd.Series(["?"] * len(alt))).astype(str)
            + "<br>Risk Skoru: " + alt["Risk_Skoru"].round(3).astype(str)
            + "<br>Duygu: " + alt["Duygu_Skoru"].round(3).astype(str)
        )
        fig.add_trace(go.Scatter(
            x=alt["Merkezilik"],
            y=alt["Duygu_Siddeti"],
            mode="markers",
            name=seviye,
            marker=dict(
                color=renk,
                size=alt["Risk_Skoru"] * 18 + 5,
                opacity=0.82,
                line=dict(color="rgba(255,255,255,.15)", width=1),
            ),
            hovertemplate="<b>%{text}</b><extra></extra>",
            text=hover_text,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Risk Matrisi — Merkezilik × Duygu Şiddeti",
            font=dict(size=13, color="#e2e8f0"),
            x=0.02,
        ),
        xaxis_title="Ağ Merkeziliği (SNA)",
        yaxis_title="Duygu Şiddeti |Polarity| (NLP)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=480,
    )
    return fig


def duygu_dagilimi_ciz(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["Duygu_Skoru"],
        nbinsx=30,
        marker=dict(
            color=df["Duygu_Skoru"],
            colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
            line=dict(color="rgba(0,0,0,.3)", width=.5),
        ),
        hovertemplate="Duygu Aralığı: %{x}<br>Yorum: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Duygu Skoru Dağılımı",
            font=dict(size=13, color="#e2e8f0"),
            x=0.02,
        ),
        xaxis_title="Polarity Skoru",
        yaxis_title="Yorum Sayısı",
        height=280,
    )
    return fig


def risk_seviyesi_pie(df: pd.DataFrame) -> go.Figure:
    sayimlar = df["Risk_Seviyesi"].value_counts()
    fig = go.Figure(go.Pie(
        labels=sayimlar.index.tolist(),
        values=sayimlar.values.tolist(),
        hole=0.62,
        marker=dict(
            colors=["#10b981", "#f59e0b", "#ef4444"],
            line=dict(color="#0a0e1a", width=3),
        ),
        textfont=dict(size=12, family="DM Sans"),
        hovertemplate="%{label}<br>%{value} kullanıcı (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Risk Seviyesi Dağılımı",
            font=dict(size=13, color="#e2e8f0"),
            x=0.02,
        ),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        height=280,
    )
    return fig


# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <div style='font-family:"DM Serif Display",serif;font-size:1.15rem;color:#e2e8f0;'>
            🔬 İtibar Risk Modeli
        </div>
        <div style='font-size:.75rem;color:#475569;margin-top:4px;'>
            Hibrit NLP × SNA Analiz Platformu
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-title">📂 VERİ KAYNAĞI</div>', unsafe_allow_html=True)
    yuklenen_dosya = st.file_uploader(
        "Excel veya CSV yükleyin",
        type=["xlsx", "csv"],
        help="Dosyanızda: Kullanici_ID, Yorum_Metni, Arkadas_Sayisi sütunları bulunmalıdır.",
    )

    st.divider()

    st.markdown('<div class="sidebar-title">⚙️ ANALİZ PARAMETRELERİ</div>', unsafe_allow_html=True)
    risk_esik = st.slider("Kritik Risk Eşiği", 0.5, 0.95, 0.66, 0.01,
                          help="Bu değerin üzerindeki kullanıcılar kritik sayılır.")
    max_tablo_satir = st.slider("Tablodaki Maksimum Satır", 5, 50, 15, 5)

    st.divider()

    st.markdown("""
    <div style='font-size:.75rem;color:#334155;line-height:1.6;'>
        <b style='color:#475569;'>Metodoloji:</b><br>
        • NLP → TextBlob Polarity<br>
        • SNA → Degree Centrality (normalize)<br>
        • Risk = |Polarity| × Merkezilik<br><br>
        <b style='color:#475569;'>Kaynak veri:</b><br>
        Philadelphia Yelp Restaurant Dataset<br>
        <i>(Yüksek Lisans Tezi — 2024)</i>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  VERİ YÜKLEMESİ
# ─────────────────────────────────────────────────────────
veri_kaynagi = "örnek"
if yuklenen_dosya is not None:
    try:
        if yuklenen_dosya.name.endswith(".csv"):
            ham_df = pd.read_csv(yuklenen_dosya)
        else:
            ham_df = pd.read_excel(yuklenen_dosya)
        veri_kaynagi = "kullanici"
    except Exception as e:
        st.error(f"Dosya okunurken hata oluştu: {e}")
        ham_df = ornek_veri_olustur()
        veri_kaynagi = "örnek"
else:
    ham_df = ornek_veri_olustur()

with st.spinner("Model çalışıyor, analiz yapılıyor…"):
    df = tam_analiz(ham_df)

kritik_df = df[df["Risk_Skoru"] >= risk_esik].copy()


# ─────────────────────────────────────────────────────────
#  HERO BAŞLIK
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-block">
    <div class="badge">{'📁 KULLANICI VERİSİ' if veri_kaynagi == 'kullanici' else '🧪 ÖRNEK VERİ — Philadelphia Yelp'}</div>
    <div class="hero-title">Hibrit Dijital İtibar <span>Risk Modeli</span></div>
    <p class="hero-sub">
        e‑WOM Tabanlı Kanaat Önderliği Tespiti &nbsp;·&nbsp;
        NLP × SNA Entegrasyon Çerçevesi &nbsp;·&nbsp;
        {len(df):,} Kullanıcı Analiz Edildi
    </p>
</div>
""", unsafe_allow_html=True)

if veri_kaynagi == "örnek":
    st.markdown("""
    <div class="info-box">
        💡 Şu an <b>örnek veri</b> görüntülüyorsunuz. Kendi verinizi analiz etmek için sol menüden
        <b>.xlsx</b> veya <b>.csv</b> dosyası yükleyin.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-box">
        ✅ <b>Kullanıcı verisi</b> yüklendi — <b>{yuklenen_dosya.name}</b>
        ({len(df):,} satır başarıyla analiz edildi).
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  KPI KARTLARI
# ─────────────────────────────────────────────────────────
ort_duygu = df["Duygu_Skoru"].mean()
kpi_html = f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="accent-bar" style="background:linear-gradient(#818cf8,#4f46e5);"></div>
    <div class="kpi-label">Toplam Yorum</div>
    <div class="kpi-value">{len(df):,}</div>
    <div class="kpi-delta" style="color:#64748b;">Veri setindeki toplam kayıt</div>
  </div>
  <div class="kpi-card">
    <div class="accent-bar" style="background:linear-gradient(#34d399,#059669);"></div>
    <div class="kpi-label">Ort. Duygu Skoru</div>
    <div class="kpi-value" style="color:{'#ef4444' if ort_duygu < 0 else '#10b981'};">
      {ort_duygu:+.3f}
    </div>
    <div class="kpi-delta" style="color:{'#ef4444' if ort_duygu < 0 else '#64748b'};">
      {'⚠️ Negatif eğilim — dikkat!' if ort_duygu < -0.05 else '✅ Genel duygu dengeli'}
    </div>
  </div>
  <div class="kpi-card">
    <div class="accent-bar" style="background:linear-gradient(#f87171,#dc2626);"></div>
    <div class="kpi-label">Yüksek Riskli Kullanıcı</div>
    <div class="kpi-value" style="color:#ef4444;">{len(kritik_df):,}</div>
    <div class="kpi-delta" style="color:#ef4444;">
      🔴 Acil müdahale gerektiriyor ({len(kritik_df)/len(df)*100:.1f}%)
    </div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  RİSK MATRİSİ
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <h3>📊 İnteraktif Risk Matrisi</h3>
  <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

st.plotly_chart(risk_matrisi_ciz(df), use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────
#  ALT GRAFİKLER
# ─────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    <div class="section-header">
      <h3>📉 Duygu Dağılımı</h3>
      <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(duygu_dagilimi_ciz(df), use_container_width=True,
                    config={"displayModeBar": False})

with col_b:
    st.markdown("""
    <div class="section-header">
      <h3>🎯 Risk Segmentasyonu</h3>
      <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(risk_seviyesi_pie(df), use_container_width=True,
                    config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────
#  ACİL MÜDAHALE TABLOSU
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <h3>🚨 Acil Müdahale Gerektirenler</h3>
  <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

if len(kritik_df) == 0:
    st.markdown("""
    <div class="info-box">
        ✅ Seçili risk eşiğinin üzerinde kullanıcı bulunmuyor. Eşik değerini sol panelden düşürebilirsiniz.
    </div>
    """, unsafe_allow_html=True)
else:
    gosterilecek_sutunlar = ["Kullanici_ID", "Duygu_Skoru", "Merkezilik",
                             "Risk_Skoru", "Risk_Seviyesi"]
    if "Yorum_Metni" in kritik_df.columns:
        gosterilecek_sutunlar.insert(1, "Yorum_Metni")
    if "Restoran" in kritik_df.columns:
        gosterilecek_sutunlar.insert(1, "Restoran")

    gosterilecek_sutunlar = [c for c in gosterilecek_sutunlar if c in kritik_df.columns]

    tablo_df = (
        kritik_df[gosterilecek_sutunlar]
        .sort_values("Risk_Skoru", ascending=False)
        .head(max_tablo_satir)
        .reset_index(drop=True)
    )

    # Sayısal sütunları yuvarlayalım
    for col in ["Duygu_Skoru", "Merkezilik", "Risk_Skoru"]:
        if col in tablo_df.columns:
            tablo_df[col] = tablo_df[col].round(4)

    # İndeks 1'den başlasın
    tablo_df.index = tablo_df.index + 1

    st.dataframe(
        tablo_df,
        use_container_width=True,
        height=min(40 + len(tablo_df) * 38, 520),
    )

    st.markdown(f"""
    <div class="warn-box">
        ⚠️ <b>{len(kritik_df)} kullanıcı</b> kritik risk eşiğini ({risk_esik:.2f}) aşıyor.
        Bu kullanıcılar hem yüksek negatif duygu (NLP) hem de güçlü sosyal ağ etkisine (SNA) sahip
        <b>"gizli kanaat önderleri"</b> olarak sınıflandırılmıştır. Acil iletişim stratejisi önerilir.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:36px 0 20px;color:#1e2640;font-size:.78rem;'>
    Hibrit Dijital İtibar Risk Modeli &nbsp;·&nbsp; NLP × SNA &nbsp;·&nbsp;
    Yüksek Lisans Tezi Platformu &nbsp;·&nbsp; Powered by Streamlit
</div>
""", unsafe_allow_html=True)
