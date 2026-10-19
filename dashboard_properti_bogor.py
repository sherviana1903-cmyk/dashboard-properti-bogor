import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG 
st.set_page_config(
    page_title="Dashboard Properti Bogor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Sora:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3d2e 0%, #1a6b4a 60%, #0f3d2e 100%);
    }
    [data-testid="stSidebar"] * { color: #d1f0e0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #a8dfc0 !important; font-size: 13px; font-weight: 500;
    }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; background: #f7f9f6; }

    .hero-header {
        background: linear-gradient(135deg, #0f3d2e 0%, #1d7a50 50%, #2ea86e 100%);
        border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        color: white; position: relative; overflow: hidden;
    }
    .hero-title { font-family: 'Sora', sans-serif; font-size: 1.8rem; font-weight: 700; margin: 0 0 0.3rem 0; }
    .hero-sub { font-size: 0.9rem; opacity: 0.8; margin: 0; }

    .metric-card {
        background: white; border-radius: 16px; padding: 1.2rem 1.4rem;
        border: 1px solid #e8f0eb; box-shadow: 0 2px 12px rgba(15,61,46,0.06);
    }
    .metric-label { font-size: 0.75rem; color: #6b8f7a; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
    .metric-value { font-family: 'Sora', sans-serif; font-size: 1.9rem; font-weight: 700; color: #0f3d2e; }
    .metric-delta { font-size: 0.78rem; margin-top: 0.3rem; color: #2ea86e; font-weight: 500; }

    .section-title {
        font-family: 'Sora', sans-serif; font-size: 1.1rem; font-weight: 700;
        color: #0f3d2e; margin: 0 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8f0eb;
    }
    .result-box { border-radius: 14px; padding: 1.2rem 1.5rem; margin-top: 1rem; font-size: 0.95rem; }
    .result-murah    { background: #d1f0e0; border-left: 5px solid #2ea86e; color: #0f3d2e; border-radius: 0; }
    .result-menengah { background: #fef3c7; border-left: 5px solid #f59e0b; color: #78350f; border-radius: 0; }
    .result-mahal    { background: #fee2e2; border-left: 5px solid #ef4444; color: #7f1d1d; border-radius: 0; }

    hr { border: none; border-top: 1px solid #e8f0eb; margin: 1.5rem 0; }

    .stButton > button {
        background: linear-gradient(135deg, #1d7a50, #2ea86e);
        color: white; border: none; border-radius: 10px;
        padding: 0.5rem 1.5rem; font-weight: 600; width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #e8f0eb; padding: 4px; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px; padding: 6px 16px; font-size: 0.85rem; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# LOAD & BERSIHKAN DATA — persis meniru pipeline Colab
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset_properti.csv', encoding='utf-8-sig')

    # Rapikan nama kolom
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Pastikan data numerik
    df['Luas_Tanah']    = pd.to_numeric(df['Luas_Tanah'], errors='coerce')
    df['Luas_Bangunan'] = pd.to_numeric(df['Luas_Bangunan'], errors='coerce')
    df['Harga']         = pd.to_numeric(df['Harga'], errors='coerce')
    df['Tahun']         = pd.to_numeric(df['Tahun'], errors='coerce')

    # Isi nilai kosong dengan median (kolom numerik saja)
    df = df.fillna(df.median(numeric_only=True))

    # Buat kolom Kategori_Harga
    def kategorikan(h):
        if h < 500_000_000:     return 'Murah'
        elif h < 2_000_000_000: return 'Menengah'
        else:                   return 'Mahal'

    df['Kategori_Harga'] = df['Harga'].apply(kategorikan)
    return df

df = load_data()

# NAIVE BAYES MODEL (1 model gabungan — sama seperti kode Colab)
@st.cache_data
def train_model(df):
    le_lok  = LabelEncoder()
    le_jen  = LabelEncoder()
    le_stat = LabelEncoder()
    le_kat  = LabelEncoder()

    df = df.copy()
    df['Lokasi_enc']    = le_lok.fit_transform(df['Lokasi'])
    df['Jenis_enc']     = le_jen.fit_transform(df['Jenis_Properti'])
    df['Status_enc']    = le_stat.fit_transform(df['Status_Terjual'])
    df['Kategori_enc']  = le_kat.fit_transform(df['Kategori_Harga'])

    # Urutan kolom disamakan persis dengan kode Colab:
    # ['Luas_Tanah','Luas_Bangunan','Lokasi_enc','Jenis_enc','Status_enc','Tahun']
    X = df[['Luas_Tanah', 'Luas_Bangunan', 'Lokasi_enc', 'Jenis_enc', 'Status_enc', 'Tahun']]
    y = df['Kategori_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, le_lok, le_jen, le_stat, le_kat

model, X_test, y_test, y_pred, le_lok, le_jen, le_stat, le_kat = train_model(df)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
cr  = classification_report(
    y_test, y_pred, target_names=le_kat.classes_, output_dict=True
)

# HELPERS
CAT_COLORS  = {'Murah': '#1d9e75', 'Menengah': '#ef9f27', 'Mahal': '#e24b4a'}
STAT_COLORS = {'Sudah Terjual': '#1d9e75', 'Belum Terjual': '#e24b4a'}

def styled_fig(fig):
    fig.update_layout(
        font=dict(family='Plus Jakarta Sans, sans-serif', size=12, color='#1a2e25'),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
        title_font=dict(family='Sora, sans-serif', size=14, color='#0f3d2e'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f4f1', linecolor='#e8f0eb', tickfont_size=11)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f4f1', linecolor='#e8f0eb', tickfont_size=11)
    return fig

def fmt_rp(x):
    if pd.isna(x): return '-'
    if x >= 1_000_000_000: return f"Rp {x/1_000_000_000:.2f} M"
    return f"Rp {x/1_000_000:.0f} jt"

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:2.2rem;">🏡</div>
        <div style="font-family:'Sora',sans-serif; font-size:1rem; font-weight:700; color:#d1f0e0;">
            Properti Bogor
        </div>
        <div style="font-size:0.72rem; color:#88c4a6; margin-top:2px;">Analisis Naive Bayes</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigasi", [
        "📊 Dashboard Utama",
        "📅 Analisis Per Tahun",
        "🗺️ Analisis Spasial",
        "🤖 Prediksi Harga",
        "📈 Evaluasi Model",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1d5a3a; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem; color:#88c4a6; font-weight:600;"
        "text-transform:uppercase; letter-spacing:.06em; margin-bottom:.5rem;'>Filter Data</div>",
        unsafe_allow_html=True
    )

    tahun_list = sorted(df['Tahun'].unique().tolist())
    tahun_sel  = st.multiselect("Tahun", tahun_list, default=tahun_list)
    jen_sel    = st.multiselect("Jenis Properti", sorted(df['Jenis_Properti'].unique()),
                                 default=list(df['Jenis_Properti'].unique()))
    stat_sel   = st.multiselect("Status Terjual", sorted(df['Status_Terjual'].unique()),
                                 default=list(df['Status_Terjual'].unique()))
    kat_sel    = st.multiselect("Kategori Harga", ['Murah', 'Menengah', 'Mahal'],
                                 default=['Murah', 'Menengah', 'Mahal'])
    st.markdown(
        "<div style='font-size:0.7rem; color:#88c4a6; margin-bottom:3px;'>Lokasi (ketik untuk cari)</div>",
        unsafe_allow_html=True
    )
    lok_sel = st.multiselect("Lokasi", sorted(df['Lokasi'].unique()),
                              default=list(df['Lokasi'].unique()),
                              label_visibility="collapsed")

    df_f = df[
        df['Tahun'].isin(tahun_sel) &
        df['Jenis_Properti'].isin(jen_sel) &
        df['Status_Terjual'].isin(stat_sel) &
        df['Kategori_Harga'].isin(kat_sel) &
        (df['Lokasi'].isin(lok_sel) if lok_sel
         else pd.Series([True] * len(df), index=df.index))
    ]

    st.markdown("<hr style='border-color:#1d5a3a; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.75rem; color:#88c4a6; text-align:center;'>Data · Penjualan · Properti</div>",
        unsafe_allow_html=True
    )

# HERO
page = menu.split(" ", 1)[1]
tahun_range = f"{min(tahun_sel)}–{max(tahun_sel)}" if tahun_sel else "–"

st.markdown(f"""
<div class="hero-header">
    <p class="hero-sub">🎓 Implementasi Algoritma Naive Bayes &nbsp;·&nbsp; Kota Bogor</p>
    <h1 class="hero-title">Dashboard Analisis Penjualan Properti</h1>
    <p class="hero-sub">Halaman: {page} &nbsp;·&nbsp; {len(df_f):,} data &nbsp;·&nbsp; Tahun {tahun_range}</p>
</div>
""", unsafe_allow_html=True)

# PAGE 1 — DASHBOARD UTAMA

if page == "Dashboard Utama":

    c1, c2, c3, c4, c5 = st.columns(5)
    terjual     = df_f[df_f['Status_Terjual'] == 'Sudah Terjual']
    pct_terjual = len(terjual) / len(df_f) * 100 if len(df_f) > 0 else 0

    for col, label, val, delta in [
        (c1, "Total Properti",  f"{len(df_f):,}",
             "dalam dataset"),
        (c2, "Sudah Terjual",   f"{len(terjual):,}",
             f"{pct_terjual:.1f}% dari total"),
        (c3, "Rata-rata Harga", fmt_rp(df_f['Harga'].mean()) if len(df_f) > 0 else "-",
             "semua properti"),
        (c4, "Rentang Tahun",   tahun_range,
             f"{len(tahun_sel)} tahun data"),
        (c5, "Akurasi Model",   f"{acc*100:.1f}%",
             "Naive Bayes"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta">↑ {delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tren per tahun
    st.markdown('<p class="section-title">Tren Jumlah Properti per Tahun</p>',
                unsafe_allow_html=True)
    col_tr1, col_tr2 = st.columns(2)

    with col_tr1:
        tren = df_f.groupby(['Tahun', 'Status_Terjual']).size().reset_index(name='Jumlah')
        fig_tren = px.line(
            tren, x='Tahun', y='Jumlah', color='Status_Terjual',
            color_discrete_map=STAT_COLORS, markers=True,
            labels={'Jumlah': 'Jumlah Properti', 'Tahun': '', 'Status_Terjual': 'Status'},
        )
        fig_tren.update_traces(line_width=2.5, marker_size=8)
        fig_tren.update_xaxes(dtick=1, tickformat='d')
        fig_tren = styled_fig(fig_tren)
        fig_tren.update_layout(height=300, title='Tren Status Penjualan per Tahun')
        st.plotly_chart(fig_tren, use_container_width=True)

    with col_tr2:
        avg_thn = df_f.groupby('Tahun')['Harga'].mean().reset_index()
        avg_thn.columns = ['Tahun', 'Rata_Harga']
        fig_avg = px.bar(
            avg_thn, x='Tahun', y='Rata_Harga',
            color='Rata_Harga',
            color_continuous_scale=['#e1f5ee', '#1d9e75', '#0f3d2e'],
            labels={'Rata_Harga': 'Rata-rata Harga (Rp)', 'Tahun': ''},
        )
        fig_avg.update_xaxes(dtick=1, tickformat='d')
        fig_avg = styled_fig(fig_avg)
        fig_avg.update_layout(height=300, coloraxis_showscale=False,
                               title='Rata-rata Harga per Tahun')
        st.plotly_chart(fig_avg, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<p class="section-title">Distribusi Harga — Top 15 Lokasi Terbanyak</p>',
                    unsafe_allow_html=True)
        top15_lok = df_f['Lokasi'].value_counts().head(15).index.tolist()
        fig_box = px.box(
            df_f[df_f['Lokasi'].isin(top15_lok)].sort_values('Lokasi'),
            x='Lokasi', y='Harga',
            color='Kategori_Harga', color_discrete_map=CAT_COLORS,
            labels={'Harga': 'Harga (Rp)', 'Lokasi': '', 'Kategori_Harga': 'Kategori'},
        )
        fig_box = styled_fig(fig_box)
        fig_box.update_xaxes(tickangle=-35)
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-title">Status Penjualan</p>', unsafe_allow_html=True)
        pie_df = df_f['Status_Terjual'].value_counts().reset_index()
        pie_df.columns = ['Status', 'Jumlah']
        fig_pie = px.pie(
            pie_df, values='Jumlah', names='Status',
            color='Status', color_discrete_map=STAT_COLORS, hole=0.55,
        )
        fig_pie = styled_fig(fig_pie)
        fig_pie.update_traces(textfont_size=12, textposition='outside')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-title">Jumlah Properti per Jenis</p>',
                    unsafe_allow_html=True)
        bar_jen = df_f.groupby(
            ['Jenis_Properti', 'Kategori_Harga']
        ).size().reset_index(name='n')
        fig_bar = px.bar(
            bar_jen, x='Jenis_Properti', y='n',
            color='Kategori_Harga', color_discrete_map=CAT_COLORS, barmode='group',
            labels={'n': 'Jumlah', 'Jenis_Properti': '', 'Kategori_Harga': 'Kategori'},
        )
        fig_bar = styled_fig(fig_bar)
        fig_bar.update_xaxes(tickangle=-30)
        fig_bar.update_layout(height=320)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-title">Harga vs Luas Tanah</p>',
                    unsafe_allow_html=True)
        fig_sc = px.scatter(
            df_f, x='Luas_Tanah', y='Harga',
            color='Kategori_Harga', color_discrete_map=CAT_COLORS, opacity=0.5,
            labels={'Harga': 'Harga (Rp)', 'Luas_Tanah': 'Luas Tanah (m²)',
                    'Kategori_Harga': 'Kategori'},
        )
        fig_sc = styled_fig(fig_sc)
        fig_sc.update_layout(height=320)
        st.plotly_chart(fig_sc, use_container_width=True)

# PAGE 2 — ANALISIS PER TAHUN

elif page == "Analisis Per Tahun":

    tahun_tersedia = sorted(df_f['Tahun'].unique().tolist())
    if not tahun_tersedia:
        st.warning("Tidak ada data untuk filter yang dipilih.")
        st.stop()

    col_sel, _ = st.columns([2, 3])
    with col_sel:
        tahun_detail = st.selectbox(
            "🔍 Pilih Tahun untuk Melihat Detail:",
            tahun_tersedia, index=len(tahun_tersedia) - 1
        )

    df_thn  = df_f[df_f['Tahun'] == tahun_detail]
    prev    = tahun_detail - 1
    df_prev = df_f[df_f['Tahun'] == prev] if prev in df_f['Tahun'].values else None

    # ── Kartu ringkasan tahun terpilih ───────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-title">📌 Ringkasan Tahun {tahun_detail}</p>',
                unsafe_allow_html=True)

    terjual_thn = df_thn[df_thn['Status_Terjual'] == 'Sudah Terjual']
    pct_tjl     = len(terjual_thn) / len(df_thn) * 100 if len(df_thn) > 0 else 0

    if df_prev is not None and len(df_prev) > 0:
        delta_n     = len(df_thn) - len(df_prev)
        delta_harga = df_thn['Harga'].mean() - df_prev['Harga'].mean()
        str_n     = f"{'↑' if delta_n >= 0 else '↓'} {abs(delta_n):,} vs {prev}"
        str_harga = f"{'↑' if delta_harga >= 0 else '↓'} {fmt_rp(abs(delta_harga))} vs {prev}"
    else:
        str_n     = "tahun pertama dalam data"
        str_harga = "tahun pertama dalam data"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta in [
        (c1, f"Total Properti {tahun_detail}", f"{len(df_thn):,}",                           str_n),
        (c2, "Sudah Terjual",                  f"{len(terjual_thn):,}",                      f"{pct_tjl:.1f}% dari total"),
        (c3, "Rata-rata Harga",
             fmt_rp(df_thn['Harga'].mean()) if len(df_thn) > 0 else "-",                     str_harga),
        (c4, "Jenis Properti Aktif",
             f"{df_thn['Jenis_Properti'].nunique()} jenis",                                   "tersedia tahun ini"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tren semua tahun
    st.markdown('<p class="section-title">Tren Data Seluruh Tahun (2020–2026)</p>',
                unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        tren_total = df_f.groupby('Tahun').size().reset_index(name='Jumlah')
        warna_bar  = ['#ef9f27' if t == tahun_detail else '#1d9e75'
                      for t in tren_total['Tahun']]
        fig_tbar = px.bar(
            tren_total, x='Tahun', y='Jumlah', text='Jumlah',
            labels={'Jumlah': 'Jumlah Properti', 'Tahun': ''},
        )
        fig_tbar.update_traces(marker_color=warna_bar, textposition='outside')
        fig_tbar.update_xaxes(dtick=1, tickformat='d')
        fig_tbar = styled_fig(fig_tbar)
        fig_tbar.update_layout(height=320, title='Jumlah Properti per Tahun')
        st.plotly_chart(fig_tbar, use_container_width=True)

    with col_t2:
        avg_line = df_f.groupby('Tahun')['Harga'].mean().reset_index()
        avg_line.columns = ['Tahun', 'Rata_Harga']
        fig_aline = px.line(
            avg_line, x='Tahun', y='Rata_Harga', markers=True,
            labels={'Rata_Harga': 'Rata-rata Harga (Rp)', 'Tahun': ''},
        )
        fig_aline.update_traces(line_color='#1d9e75', line_width=2.5,
                                 marker_color='#0f3d2e', marker_size=9)
        y_val = avg_line[avg_line['Tahun'] == tahun_detail]['Rata_Harga'].values
        if len(y_val) > 0:
            fig_aline.add_scatter(
                x=[tahun_detail], y=[y_val[0]], mode='markers',
                marker=dict(color='#ef9f27', size=14, symbol='star'),
                name=f'Tahun {tahun_detail}', showlegend=True
            )
        fig_aline.update_xaxes(dtick=1, tickformat='d')
        fig_aline = styled_fig(fig_aline)
        fig_aline.update_layout(height=320, title='Rata-rata Harga per Tahun')
        st.plotly_chart(fig_aline, use_container_width=True)

    # Kategori harga per tahun
    st.markdown('<p class="section-title">Perbandingan Kategori Harga per Tahun</p>',
                unsafe_allow_html=True)
    kat_thn = df_f.groupby(['Tahun', 'Kategori_Harga']).size().reset_index(name='Jumlah')
    fig_kat = px.bar(
        kat_thn, x='Tahun', y='Jumlah', color='Kategori_Harga',
        color_discrete_map=CAT_COLORS, barmode='group', text_auto=True,
        labels={'Jumlah': 'Jumlah Properti', 'Tahun': '', 'Kategori_Harga': 'Kategori'},
    )
    fig_kat.update_traces(textposition='outside')
    fig_kat.update_xaxes(dtick=1, tickformat='d')
    fig_kat = styled_fig(fig_kat)
    fig_kat.update_layout(height=380)
    st.plotly_chart(fig_kat, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Detail tahun terpilih
    st.markdown(f'<p class="section-title">Detail Data Tahun {tahun_detail}</p>',
                unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        jen_thn = df_thn.groupby(
            ['Jenis_Properti', 'Status_Terjual']
        ).size().reset_index(name='n')
        fig_jen = px.bar(
            jen_thn, x='Jenis_Properti', y='n',
            color='Status_Terjual', color_discrete_map=STAT_COLORS, barmode='stack',
            labels={'n': 'Jumlah', 'Jenis_Properti': '', 'Status_Terjual': 'Status'},
            title=f'Jenis Properti — Tahun {tahun_detail}',
        )
        fig_jen = styled_fig(fig_jen)
        fig_jen.update_xaxes(tickangle=-30)
        fig_jen.update_layout(height=360)
        st.plotly_chart(fig_jen, use_container_width=True)

    with col_d2:
        stt = df_thn['Status_Terjual'].value_counts().reset_index()
        stt.columns = ['Status', 'Jumlah']
        fig_pie_t = px.pie(
            stt, values='Jumlah', names='Status',
            color='Status', color_discrete_map=STAT_COLORS, hole=0.55,
            title=f'Status Penjualan — Tahun {tahun_detail}',
        )
        fig_pie_t = styled_fig(fig_pie_t)
        fig_pie_t.update_traces(textfont_size=12, textposition='outside')
        fig_pie_t.update_layout(height=360)
        st.plotly_chart(fig_pie_t, use_container_width=True)

    col_d3, col_d4 = st.columns(2)
    with col_d3:
        top10 = df_thn['Lokasi'].value_counts().head(10).reset_index()
        top10.columns = ['Lokasi', 'Jumlah']
        top10 = top10.sort_values('Jumlah', ascending=True)
        fig_lok = px.bar(
            top10, x='Jumlah', y='Lokasi', orientation='h',
            color='Jumlah',
            color_continuous_scale=['#e1f5ee', '#1d9e75', '#0f3d2e'],
            text='Jumlah',
            labels={'Jumlah': 'Jumlah Properti', 'Lokasi': ''},
            title=f'Top 10 Lokasi — Tahun {tahun_detail}',
        )
        fig_lok = styled_fig(fig_lok)
        fig_lok.update_traces(textposition='outside')
        fig_lok.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_lok, use_container_width=True)

    with col_d4:
        avg_jen = df_thn.groupby('Jenis_Properti')['Harga'].mean().reset_index()
        avg_jen.columns = ['Jenis', 'Rata_Harga']
        avg_jen = avg_jen.sort_values('Rata_Harga', ascending=True)
        fig_aj = px.bar(
            avg_jen, x='Rata_Harga', y='Jenis', orientation='h',
            color='Rata_Harga',
            color_continuous_scale=['#e1f5ee', '#1d9e75', '#0f3d2e'],
            labels={'Rata_Harga': 'Rata-rata Harga (Rp)', 'Jenis': ''},
            title=f'Rata-rata Harga per Jenis — Tahun {tahun_detail}',
        )
        fig_aj = styled_fig(fig_aj)
        fig_aj.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_aj, use_container_width=True)

    # Tabel ringkasan semua tahun
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Ringkasan Statistik Semua Tahun</p>',
                unsafe_allow_html=True)

    ringkasan = df_f.groupby('Tahun').agg(
        Total        =('Harga', 'count'),
        Sudah_Terjual=('Status_Terjual', lambda x: (x == 'Sudah Terjual').sum()),
        Belum_Terjual=('Status_Terjual', lambda x: (x == 'Belum Terjual').sum()),
        Rata_Harga   =('Harga', 'mean'),
        Min_Harga    =('Harga', 'min'),
        Max_Harga    =('Harga', 'max'),
    ).reset_index()

    ringkasan['% Terjual']       = (ringkasan['Sudah_Terjual'] /
                                     ringkasan['Total'] * 100).round(1).astype(str) + '%'
    ringkasan['Rata-rata Harga'] = ringkasan['Rata_Harga'].apply(fmt_rp)
    ringkasan['Min Harga']       = ringkasan['Min_Harga'].apply(fmt_rp)
    ringkasan['Max Harga']       = ringkasan['Max_Harga'].apply(fmt_rp)

    tampil = ringkasan[['Tahun', 'Total', 'Sudah_Terjual', 'Belum_Terjual',
                         '% Terjual', 'Rata-rata Harga', 'Min Harga', 'Max Harga']].copy()
    tampil.columns = ['Tahun', 'Total', 'Sudah Terjual', 'Belum Terjual',
                      '% Terjual', 'Rata-rata Harga', 'Min Harga', 'Max Harga']
    st.dataframe(tampil.set_index('Tahun'), use_container_width=True)


# PAGE 3 — ANALISIS SPASIAL

elif page == "Analisis Spasial":
    st.markdown('<p class="section-title">Heatmap Rata-rata Harga per Lokasi × Jenis Properti</p>',
                unsafe_allow_html=True)

    top_lok = df_f['Lokasi'].value_counts().head(15).index.tolist()
    df_hm   = df_f[df_f['Lokasi'].isin(top_lok)]
    pivot   = df_hm.pivot_table(
        values='Harga', index='Lokasi', columns='Jenis_Properti', aggfunc='mean'
    ) / 1_000_000_000

    fig_hm, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.45)))
    cmap = mcolors.LinearSegmentedColormap.from_list('bogor', ['#e1f5ee', '#1d9e75', '#0f3d2e'])
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap,
                linewidths=0.5, linecolor='#f0f4f1',
                cbar_kws={'label': 'Rata-rata Harga (Miliar Rp)', 'shrink': 0.8}, ax=ax)
    ax.set_title('Rata-rata Harga Properti — Top 15 Lokasi (Miliar Rp)',
                 fontsize=14, fontweight='bold', color='#0f3d2e', pad=14)
    ax.set_xlabel('Jenis Properti', fontsize=11, color='#3d6b56', labelpad=8)
    ax.set_ylabel('Lokasi', fontsize=11, color='#3d6b56', labelpad=8)
    ax.tick_params(axis='both', labelsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    fig_hm.patch.set_facecolor('#f7f9f6')
    ax.set_facecolor('#f7f9f6')
    plt.tight_layout()
    st.pyplot(fig_hm)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Heatmap Korelasi Fitur Numerik</p>',
                    unsafe_allow_html=True)
        corr = df_f[['Luas_Tanah', 'Luas_Bangunan', 'Harga']].corr()
        fig_corr, ax2 = plt.subplots(figsize=(5, 4))
        cmap2 = mcolors.LinearSegmentedColormap.from_list('corr',
                                                           ['#fa7c58', '#ffffff', '#1d9e75'])
        sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap2, center=0,
                    linewidths=0.5, linecolor='#f0f4f1',
                    vmin=-1, vmax=1, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('Korelasi Antar Fitur', fontsize=12, fontweight='bold',
                      color='#0f3d2e', pad=10)
        ax2.tick_params(axis='both', labelsize=9)
        plt.xticks(rotation=20, ha='right')
        fig_corr.patch.set_facecolor('#f7f9f6')
        ax2.set_facecolor('#f7f9f6')
        plt.tight_layout()
        st.pyplot(fig_corr)

    with col2:
        st.markdown('<p class="section-title">Top 20 Lokasi — Rata-rata Harga</p>',
                    unsafe_allow_html=True)
        avg_lok = (df_f.groupby('Lokasi')['Harga'].mean().reset_index()
                   .sort_values('Harga', ascending=False).head(20)
                   .sort_values('Harga', ascending=True))
        fig_hlok = px.bar(
            avg_lok, x='Harga', y='Lokasi', orientation='h',
            color='Harga',
            color_continuous_scale=['#e1f5ee', '#1d9e75', '#0f3d2e'],
            labels={'Harga': 'Rata-rata Harga (Rp)', 'Lokasi': ''},
        )
        fig_hlok = styled_fig(fig_hlok)
        fig_hlok.update_layout(height=520, coloraxis_showscale=False)
        st.plotly_chart(fig_hlok, use_container_width=True)

    st.markdown('<p class="section-title">Proporsi Kategori Harga — Top 15 Lokasi Terbanyak</p>',
                unsafe_allow_html=True)
    top15 = df_f['Lokasi'].value_counts().head(15).index.tolist()
    stacked = (df_f[df_f['Lokasi'].isin(top15)]
               .groupby(['Lokasi', 'Kategori_Harga']).size().reset_index(name='n'))
    fig_stk = px.bar(
        stacked, x='Lokasi', y='n', color='Kategori_Harga',
        color_discrete_map=CAT_COLORS, barmode='stack',
        labels={'n': 'Jumlah', 'Lokasi': '', 'Kategori_Harga': 'Kategori'},
    )
    fig_stk = styled_fig(fig_stk)
    fig_stk.update_xaxes(tickangle=-35)
    fig_stk.update_layout(height=380)
    st.plotly_chart(fig_stk, use_container_width=True)

# PAGE 4 — PREDIKSI HARGA

elif page == "Prediksi Harga":
    st.markdown('<p class="section-title">Masukkan Data Properti untuk Prediksi Kategori Harga</p>',
                unsafe_allow_html=True)

    with st.form("form_prediksi"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1: in_lok  = st.selectbox("Lokasi", sorted(df['Lokasi'].unique()))
        with r1c2: in_jen  = st.selectbox("Jenis Properti", sorted(df['Jenis_Properti'].unique()))
        with r1c3: in_stat = st.selectbox("Status Terjual", sorted(df['Status_Terjual'].unique()))
        with r1c4: in_thn  = st.selectbox("Tahun", sorted(df['Tahun'].unique(), reverse=True))

        r2c1, r2c2 = st.columns(2)
        with r2c1: in_lt = st.number_input("Luas Tanah (m²)",    0, 50000, 150, 10)
        with r2c2: in_lb = st.number_input("Luas Bangunan (m²)",  0, 10000, 100, 10)

        submitted = st.form_submit_button("🔍 Prediksi Kategori Harga")

    if submitted:
        lok_enc  = le_lok.transform([in_lok])[0]
        jen_enc  = le_jen.transform([in_jen])[0]
        stat_enc = le_stat.transform([in_stat])[0]

        X_input = np.array([[in_lt, in_lb, lok_enc, jen_enc, stat_enc, in_thn]])
        pred    = model.predict(X_input)[0]
        prob    = model.predict_proba(X_input)[0]
        label   = le_kat.inverse_transform([pred])[0]

        css_cls = {'Murah': 'result-murah', 'Menengah': 'result-menengah', 'Mahal': 'result-mahal'}
        icons   = {'Murah': '🟢', 'Menengah': '🟡', 'Mahal': '🔴'}

        # Range harga berjenjang
        subset = df[
            (df['Kategori_Harga'] == label) &
            (df['Lokasi'] == in_lok) &
            (df['Jenis_Properti'] == in_jen)
        ]
        if len(subset) < 3:
            subset = df[(df['Kategori_Harga'] == label) & (df['Lokasi'] == in_lok)]
        if len(subset) < 3:
            subset = df[(df['Kategori_Harga'] == label) & (df['Jenis_Properti'] == in_jen)]
        if len(subset) < 3:
            subset = df[df['Kategori_Harga'] == label]

        est_harga = subset['Harga'].mean()
        min_h     = subset['Harga'].quantile(0.10)
        max_h     = subset['Harga'].quantile(0.90)
        med_h     = subset['Harga'].median()

        st.markdown(f"""
        <div class="result-box {css_cls[label]}">
            <strong style="font-size:1.15rem;">{icons[label]} Prediksi Kategori: <u>{label}</u></strong><br><br>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-top:6px;">
                <div style="text-align:center;">
                    <div style="font-size:0.7rem; opacity:0.7; font-weight:600; text-transform:uppercase;">Estimasi Harga</div>
                    <div style="font-size:1.05rem; font-weight:700; margin-top:2px;">{fmt_rp(est_harga)}</div>
                </div>
                <div style="text-align:center; border-left:1px solid rgba(0,0,0,0.1); border-right:1px solid rgba(0,0,0,0.1);">
                    <div style="font-size:0.7rem; opacity:0.7; font-weight:600; text-transform:uppercase;">Kisaran Harga</div>
                    <div style="font-size:0.85rem; font-weight:700; margin-top:2px;">{fmt_rp(min_h)} – {fmt_rp(max_h)}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.7rem; opacity:0.7; font-weight:600; text-transform:uppercase;">Median Pasar</div>
                    <div style="font-size:1.05rem; font-weight:700; margin-top:2px;">{fmt_rp(med_h)}</div>
                </div>
            </div>
            <div style="margin-top:10px; font-size:0.8rem; opacity:0.75;">
                {in_jen} di {in_lok} · {in_stat} · Tahun {in_thn} · LT {in_lt} m² · LB {in_lb} m²
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Probabilitas Tiap Kategori</p>',
                    unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            'Kategori': le_kat.classes_,
            'Probabilitas (%)': (prob * 100).round(2)
        }).sort_values('Probabilitas (%)', ascending=False)

        col_prob, col_pbar = st.columns([1, 2])
        with col_prob:
            badge_map = {
                'Murah':    ('#d1f0e0', '#085041'),
                'Menengah': ('#fef3c7', '#92400e'),
                'Mahal':    ('#fee2e2', '#991b1b'),
            }
            for _, row in prob_df.iterrows():
                bg, fg = badge_map[row['Kategori']]
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding: 0.7rem 1rem; border-radius:10px; margin-bottom:0.5rem;
                            background:white; border:1px solid #e8f0eb;">
                    <span style="background:{bg}; color:{fg}; padding:2px 10px;
                                 border-radius:20px; font-size:0.75rem; font-weight:600;">
                        {row['Kategori']}
                    </span>
                    <strong style="font-size:1.1rem;">{row['Probabilitas (%)']:.1f}%</strong>
                </div>
                """, unsafe_allow_html=True)

        with col_pbar:
            fig_prob = px.bar(
                prob_df, x='Kategori', y='Probabilitas (%)',
                color='Kategori', color_discrete_map=CAT_COLORS, text='Probabilitas (%)',
            )
            fig_prob = styled_fig(fig_prob)
            fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_prob.update_layout(height=300, showlegend=False, yaxis_range=[0, 110])
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown('<p class="section-title">Distribusi Harga Data Serupa</p>',
                    unsafe_allow_html=True)
        fig_dist = px.histogram(
            subset, x='Harga', nbins=20,
            color_discrete_sequence=[CAT_COLORS[label]],
            labels={'Harga': 'Harga (Rp)', 'count': 'Jumlah Properti'},
        )
        fig_dist.add_vline(x=est_harga, line_dash='dash', line_color='#0f3d2e',
                           annotation_text='Estimasi', annotation_position='top right')
        fig_dist = styled_fig(fig_dist)
        fig_dist.update_layout(height=280, bargap=0.05)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption(f"Berdasarkan {len(subset):,} properti dengan kategori serupa di dataset")

# PAGE 5 — EVALUASI MODEL

elif page == "Evaluasi Model":
    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, sub) in zip([c1, c2, c3, c4], [
        ("Akurasi",   f"{acc*100:.1f}%",                              "keseluruhan"),
        ("Precision", f"{cr['weighted avg']['precision']*100:.1f}%",  "weighted avg"),
        ("Recall",    f"{cr['weighted avg']['recall']*100:.1f}%",     "weighted avg"),
        ("F1-Score",  f"{cr['weighted avg']['f1-score']*100:.1f}%",   "weighted avg"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_cm, col_pr = st.columns(2)

    with col_cm:
        st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
        fig_cm = px.imshow(
            cm, x=le_kat.classes_, y=le_kat.classes_, text_auto=True,
            color_continuous_scale=[[0, '#e1f5ee'], [0.5, '#1d9e75'], [1, '#04342c']],
            labels=dict(x='Prediksi', y='Aktual'),
        )
        fig_cm = styled_fig(fig_cm)
        fig_cm.update_layout(height=380, coloraxis_showscale=False)
        fig_cm.update_xaxes(side='bottom')
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_pr:
        st.markdown('<p class="section-title">Precision, Recall, F1 per Kelas</p>',
                    unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            'Kelas':     le_kat.classes_,
            'Precision': [cr[k]['precision'] for k in le_kat.classes_],
            'Recall':    [cr[k]['recall']    for k in le_kat.classes_],
            'F1-Score':  [cr[k]['f1-score']  for k in le_kat.classes_],
        })
        mm = metrics_df.melt(id_vars='Kelas', var_name='Metrik', value_name='Skor')
        fig_met = px.bar(
            mm, x='Kelas', y='Skor', color='Metrik', barmode='group',
            color_discrete_sequence=['#1d9e75', '#ef9f27', '#e24b4a'],
            text_auto='.2f',
        )
        fig_met = styled_fig(fig_met)
        fig_met.update_layout(height=380, yaxis_range=[0, 1.15])
        fig_met.update_traces(textposition='outside')
        st.plotly_chart(fig_met, use_container_width=True)

    st.markdown('<p class="section-title">Perbandingan Aktual vs Prediksi</p>',
                unsafe_allow_html=True)
    al = le_kat.inverse_transform(y_test)
    pl = le_kat.inverse_transform(y_pred)
    cmp_df = pd.DataFrame({
        'Index': range(len(al)), 'Aktual': al,
        'Prediksi': pl, 'Benar': al == pl
    })
    fig_cmp = px.scatter(
        cmp_df, x='Index', y='Aktual', color='Benar', symbol='Prediksi',
        color_discrete_map={True: '#1d9e75', False: '#e24b4a'},
        labels={'Benar': 'Prediksi Benar?'}, opacity=0.75,
    )
    fig_cmp = styled_fig(fig_cmp)
    fig_cmp.update_layout(height=300)
    st.plotly_chart(fig_cmp, use_container_width=True)

    with st.expander("📋 Lihat Tabel Data Lengkap"):
        show_df = df_f[['Tahun', 'Lokasi', 'Jenis_Properti', 'Status_Terjual',
                         'Luas_Tanah', 'Luas_Bangunan', 'Harga', 'Kategori_Harga']].copy()
        show_df.columns = ['Tahun', 'Lokasi', 'Jenis', 'Status',
                           'Luas Tanah (m²)', 'Luas Bangunan (m²)', 'Harga (Rp)', 'Kategori']
        st.dataframe(
            show_df.style.format({'Harga (Rp)': 'Rp {:,.0f}'}),
            use_container_width=True, height=400
        )
