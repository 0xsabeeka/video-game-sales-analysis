import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="VG Analyst — Violent Games Study",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY   = "#0B1E3D"
GOLD   = "#C9A84C"
BEIGE  = "#F5ECD7"
CREAM  = "#FAF6EE"
SLATE  = "#1E3358"
MUTED  = "#8A9BB5"
WHITE  = "#FFFFFF"
ACCENT = "#D4A843"
LIGHT_NAVY = "#2A4A7F"

RATING_COLORS = {
    "E":    NAVY,
    "E10+": SLATE,
    "T":    ACCENT,
    "M":    GOLD
}

# Matplotlib global style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#D5C9B5'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#D5C9B5'
plt.rcParams['axes.facecolor'] = WHITE
plt.rcParams['figure.facecolor'] = CREAM

def load_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {CREAM};
        color: {NAVY};
    }}
    .main .block-container {{
        padding: 2.5rem 2.5rem 1.5rem;
        max-width: 1400px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, {SLATE} 100%) !important;
        border-right: 2px solid {GOLD};
    }}
    section[data-testid="stSidebar"] * {{ color: {BEIGE} !important; }}

    section[data-testid="stSidebar"] button {{
        background: transparent !important;
        border: 1px solid {GOLD}33 !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        transition: all 0.3s ease !important;
        color: {BEIGE} !important;
    }}
    section[data-testid="stSidebar"] button:hover {{
        background: {GOLD}22 !important;
        border-color: {GOLD} !important;
        color: {GOLD} !important;
    }}
    section[data-testid="stSidebar"] button[kind="primary"] {{
        background: {GOLD}33 !important;
        border-color: {GOLD} !important;
        color: {GOLD} !important;
        font-weight: 600 !important;
        box-shadow: 0 0 15px {GOLD}22 !important;
    }}
    section[data-testid="stSidebar"] button:focus {{
        background: {GOLD}33 !important;
        border-color: {GOLD} !important;
        color: {GOLD} !important;
    }}

    /* Headings */
    h1 {{
        font-family: 'Playfair Display', serif;
        color: {NAVY};
        letter-spacing: -0.02em;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }}
    h2, h3 {{
        font-family: 'Playfair Display', serif;
        color: {NAVY};
        letter-spacing: -0.02em;
    }}

    /* Metric cards */
    .metric-card {{
        background: {WHITE};
        border: 1px solid #E2D9C8;
        border-left: 4px solid {GOLD};
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(11,30,61,0.06);
        transition: transform 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(11,30,61,0.1);
    }}
    .metric-label {{
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {MUTED};
        margin-bottom: 4px;
    }}
    .metric-value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {NAVY};
        font-family: 'Playfair Display', serif;
    }}
    .metric-sub {{
        font-size: 0.78rem;
        color: {MUTED};
        margin-top: 2px;
    }}

    /* Section header */
    .section-header {{
        border-bottom: 2px solid {GOLD};
        padding-bottom: 0.5rem;
        margin-bottom: 2rem;
        margin-top: 2.5rem;
        font-family: 'Playfair Display', serif;
        font-size: 1.35rem;
        color: {NAVY};
        text-align: center;
    }}

    /* Callout box */
    .callout {{
        background: linear-gradient(135deg, {NAVY} 0%, {SLATE} 100%);
        color: {BEIGE} !important;
        border-radius: 10px;
        padding: 1.8rem 2rem;
        margin: 2rem 0;
        border: 1px solid {GOLD}33;
    }}
    .callout h3 {{
        color: {GOLD} !important;
        margin-bottom: 0.8rem;
        font-family: 'Playfair Display', serif;
    }}

    /* Data table */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
        margin: 1.5rem 0;
    }}

    /* Verdict badge */
    .verdict {{
        display: inline-block;
        background: {GOLD};
        color: {NAVY};
        font-weight: 700;
        font-size: 1rem;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }}

    /* Golden line */
    .golden-line {{
        border: none;
        border-top: 2px solid {GOLD};
        margin: 0.8rem 0;
        opacity: 0.8;
    }}

    /* Footer */
    .footer {{
        margin-top: 2rem;
        border-top: 1px solid #E2D9C8;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        font-size: 0.75rem;
        color: {MUTED};
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
        df = df.dropna(subset=["Global_Sales", "Rating"])
        df = df[df["Rating"].isin(["E", "E10+", "T", "M"])]
        df["Critic_Score"] = pd.to_numeric(df["Critic_Score"], errors="coerce")
        df["User_Score"]   = pd.to_numeric(df["User_Score"],   errors="coerce")
        df["Violence_Level"] = df["Rating"].map({"E": 0, "E10+": 1, "T": 2, "M": 3})
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'Video_Games_Sales_as_at_22_Dec_2016.csv' is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# ── Plot helper 
def styled_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(WHITE)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#D5C9B5")
    ax.spines[["left","bottom"]].set_linewidth(1.5)
    ax.tick_params(colors=NAVY, labelsize=10)
    ax.grid(True, alpha=0.3, color='#D5C9B5', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.title.set_fontsize(14)
    ax.title.set_fontweight('bold')
    ax.title.set_color(NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.xaxis.label.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_color(NAVY)
    ax.yaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(11)
    return fig, ax


# ── Helpers 
def metric_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def spacer():
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

# ── Sidebar 
def show_sidebar(df):
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    with st.sidebar:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            try:
                st.image("logo.png", width=200)
            except:
                st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)

        st.markdown(f"""
            <div style='text-align:center; padding:5px 0 2px 0;'>
                <h1 style='color:{GOLD}; font-size:24px; margin:0;
                           font-family:"Playfair Display",serif;'>
                    VG Analyst
                </h1>
                <p style='color:{BEIGE}; font-size:11px; margin:3px 0 0 0; opacity:0.85;'>
                    Violent Games Statistical Analysis
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="golden-line">', unsafe_allow_html=True)

        pages = [
            'Home',
            'Graphical Analysis',
            'Descriptive Statistics',
            'Probability & Distributions',
            'Regression & Prediction',
            'Conclusion'
        ]

        for page_name in pages:
            is_active = st.session_state.page == page_name
            if st.button(
                page_name,
                use_container_width=True,
                type="primary" if is_active else "secondary",
                key=f"nav_{page_name}"
            ):
                st.session_state.page = page_name
                st.rerun()

        st.markdown('<hr class="golden-line">', unsafe_allow_html=True)

        if not df.empty:
            st.markdown(f"""
                <div style='padding:12px; background:#0d2137; border-radius:10px;
                            text-align:center; border:1px solid {GOLD}33;'>
                    <p style='color:{GOLD}; font-size:14px; margin:0; font-weight:600;'>
                        Dataset Info
                    </p>
                    <p style='color:{BEIGE}; font-size:13px; margin:6px 0 0 0; line-height:1.8;'>
                        Games: {len(df):,}<br>
                        Variables: {df.shape[1]}<br>
                        Ratings: E, E10+, T, M<br>
                        Source: Kaggle
                    </p>
                </div>
            """, unsafe_allow_html=True)

    return st.session_state.page

# PAGE 1 — Home
def show_home(df):
    if df.empty:
        st.error("No data available.")
        return

    st.markdown(f"""
    <div style='padding:1rem 0 2rem 0;'>
        <h1 style='font-family:"Playfair Display",serif; font-size:2.8rem;
                   color:{NAVY}; margin-bottom:0.5rem;'>
            Do Violent Video Games
            <span style='color:{GOLD}'>Sell More?</span>
        </h1>
        <p style='color:{MUTED}; font-size:1.1rem; margin-top:0.3rem; text-align:center;'>
            A Statistical Analysis using ESRB Ratings & Global Sales Data
        </p>
    </div>
    <hr style='border:none; border-top:2px solid {GOLD}; margin:1rem 0 2.5rem;'>
    """, unsafe_allow_html=True)

    # KPI row
    r_counts = df["Rating"].value_counts()
    top_game = df.loc[df['Global_Sales'].idxmax()]
    top_game_name = top_game['Name'] if 'Name' in df.columns else "Unknown"
    top_game_sales = top_game['Global_Sales']

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("Total Games",      f"{len(df):,}",                        "after filtering"),
        ("M-Rated Games",    f"{r_counts.get('M',0):,}",            f"{r_counts.get('M',0)/len(df)*100:.1f}% of total"),
        ("E-Rated Games",    f"{r_counts.get('E',0):,}",            f"{r_counts.get('E',0)/len(df)*100:.1f}% of total"),
        ("Avg Global Sales", f"{df['Global_Sales'].mean():.3f}M",   "per game average"),
        ("Top Seller",       f"{top_game_sales:.1f}M",              str(top_game_name)[:30]),
    ]
    for col, (lbl, val, sub) in zip([c1,c2,c3,c4,c5], cards):
        col.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

    spacer()

    # Research question callout
    st.markdown(f"""
    <div class="callout">
        <h3>Research Question</h3>
        <p style='color:{BEIGE}; margin:0; font-size:0.95rem; line-height:1.8;'>
            Games like <b style='color:{GOLD}'>GTA</b> and
            <b style='color:{GOLD}'>Call of Duty</b> are household names —
            and they carry an <b>M (Mature)</b> ESRB rating.
            But is violent content actually <em>correlated</em> with higher sales,
            or is that just survivor bias?
            We use <b>{len(df):,} real games</b>, grouped by ESRB rating, and apply
            descriptive stats, probability theory, and regression to answer this
            question rigorously.
        </p>
    </div>
    """, unsafe_allow_html=True)

    spacer()

    # Dataset overview + data preview
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        section("Dataset Overview")
        st.markdown(f"""
        <table style='width:100%; border-collapse:collapse; margin-top:0.5rem;'>
            <tr><td style='padding:10px; border-bottom:1px solid #E2D9C8;'><b>Source</b></td>
                <td style='padding:10px; border-bottom:1px solid #E2D9C8;'>
                    Kaggle — Video Game Sales with Ratings</td></tr>
            <tr><td style='padding:10px; border-bottom:1px solid #E2D9C8;'><b>File</b></td>
                <td style='padding:10px; border-bottom:1px solid #E2D9C8;'>
                    Single CSV, {len(df):,} rows after cleaning</td></tr>
            <tr><td style='padding:10px; border-bottom:1px solid #E2D9C8;'><b>Variables</b></td>
                <td style='padding:10px; border-bottom:1px solid #E2D9C8;'>
                    Name, Platform, Genre, Sales (by region), Scores, Rating</td></tr>
            <tr><td style='padding:10px; border-bottom:1px solid #E2D9C8;'><b>Rating System</b></td>
                <td style='padding:10px; border-bottom:1px solid #E2D9C8;'>
                    ESRB — E (Everyone), E10+, T (Teen), M (Mature)</td></tr>
            <tr><td style='padding:10px;'><b>Period</b></td>
                <td style='padding:10px;'>Up to December 2016</td></tr>
        </table>
        """, unsafe_allow_html=True)

    with col_b:
        section("Raw Data Preview")
        preview_cols = ["Name","Platform","Genre","Rating","Global_Sales","Critic_Score"]
        preview_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(
            df[preview_cols].head(10).style.set_properties(**{
                'background-color': WHITE,
                'color': NAVY,
                'border-color': '#E2D9C8'
            }),
            use_container_width=True,
            hide_index=True
        )

    spacer()

    # Rating distribution bar chart 
    section("Games Count by ESRB Rating")
    rating_counts = df['Rating'].value_counts().reindex(["E","E10+","T","M"])
    colors = [RATING_COLORS.get(r, GOLD) for r in rating_counts.index]

    fig, ax = styled_fig(10, 5)
    bars = ax.bar(rating_counts.index, rating_counts.values,
                  color=colors, width=0.5,
                  edgecolor=WHITE, linewidth=1.5, alpha=0.95)
    ax.set_xlabel("ESRB Rating", fontweight='bold', fontsize=11)
    ax.set_ylabel("Number of Games", fontweight='bold', fontsize=11)
    ax.set_title("How Many Games Fall in Each Rating Category?",
                 fontweight='bold', fontsize=14, pad=15)
    for bar, val in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,}', ha='center', va='bottom',
                fontsize=11, color=NAVY, fontweight='bold')
    ax.set_ylim(0, rating_counts.max() * 1.15)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown(f"<div class='footer'>ESRB data sourced from Kaggle · Spring 2026 · Probability & Statistics Project</div>",
                unsafe_allow_html=True)


# PAGE 2 — Graphical Analysis
def show_graphs(df):
    if df.empty:
        st.error("No data available.")
        return

    st.markdown(f"""
    <h1 style='font-family:"Playfair Display",serif; font-size:2.4rem;
               margin-bottom:2rem;'>Graphical Analysis</h1>
    """, unsafe_allow_html=True)

    # Row 1: Avg sales bar + Pie chart
    col1, col2 = st.columns(2, gap="large")
    with col1:
        section("Average Global Sales by ESRB Rating")
        avg = df.groupby("Rating")["Global_Sales"].mean().reindex(["E","E10+","T","M"])
        colors = [RATING_COLORS.get(r, GOLD) for r in avg.index]

        fig, ax = styled_fig(7, 5)
        bars = ax.bar(range(len(avg)), avg.values, color=colors,
                      width=0.6, edgecolor=WHITE, linewidth=1.5, alpha=0.95)
        ax.set_xticks(range(len(avg)))
        ax.set_xticklabels(avg.index, fontweight='bold', fontsize=12)
        ax.set_ylabel("Average Global Sales (millions)", fontweight='bold', fontsize=11)
        ax.set_xlabel("ESRB Rating", fontweight='bold', fontsize=11)
        ax.set_title("Mean Sales by Rating Category", fontweight='bold', fontsize=14, pad=15)
        for bar, val in zip(bars, avg.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}M', ha='center', va='bottom',
                    fontsize=10, color=NAVY, fontweight='bold')
        ax.set_ylim(0, avg.max() * 1.15)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        section("Distribution of Games by Rating")
        counts = df["Rating"].value_counts().reindex(["E","E10+","T","M"])
        colors_pie = [RATING_COLORS.get(r, NAVY) for r in counts.index]

        fig, ax = styled_fig(7, 5)
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=None,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            wedgeprops=dict(edgecolor=WHITE, linewidth=2, width=0.4),
            pctdistance=0.75
        )
        legend_labels = [f'{r} ({c:,})' for r, c in zip(counts.index, counts.values)]
        ax.legend(wedges, legend_labels,
                  title="ESRB Ratings",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=10, title_fontsize=11)
        for at in autotexts:
            at.set_color(WHITE)
            at.set_fontsize(11)
            at.set_fontweight('bold')
        ax.set_title("Proportion of Games per Rating",
                     fontweight='bold', fontsize=14, pad=15)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    spacer()

    # Row 2: Genre chart + Heatmap
    col3, col4 = st.columns(2, gap="large")

    with col3:
        section("Top Genres by Global Sales")
        if "Genre" in df.columns:
            top_genres = (df.groupby("Genre")["Global_Sales"]
                          .sum().sort_values(ascending=True).tail(8))
            fig, ax = styled_fig(7, 5)
            violent_genres = ['Action', 'Shooter', 'Fighting']
            genre_colors = [GOLD if g in violent_genres else NAVY
                            for g in top_genres.index]

            bars = ax.barh(range(len(top_genres)), top_genres.values,
                           color=genre_colors, edgecolor=WHITE,
                           linewidth=1, alpha=0.9)
            ax.set_yticks(range(len(top_genres)))
            ax.set_yticklabels(top_genres.index, fontweight='bold', fontsize=10)
            ax.set_xlabel("Total Global Sales (millions)", fontweight='bold', fontsize=11)
            ax.set_title("Top 8 Genres by Revenue (Gold = Violent)",
                         fontweight='bold', fontsize=14, pad=15)
            for bar, val in zip(bars, top_genres.values):
                ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                        f'{val:.0f}M', va='center', fontsize=9,
                        color=NAVY, fontweight='bold')
            ax.set_xlim(0, top_genres.max() * 1.2)
            ax.legend(handles=[
                mpatches.Patch(color=GOLD, label='Violent Genres'),
                mpatches.Patch(color=NAVY, label='Non-Violent Genres')
            ], fontsize=9, loc='lower right')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col4:
        section("Correlation Between Numeric Variables")
        num_cols = ["Global_Sales","NA_Sales","EU_Sales","JP_Sales",
                    "Critic_Score","User_Score","Violence_Level"]
        num_cols = [c for c in num_cols if c in df.columns]
        corr = df[num_cols].corr()

        fig, ax = styled_fig(7, 5.5)
        colors_list = [WHITE, BEIGE, ACCENT, GOLD, NAVY]
        custom_cmap = sns.blend_palette(colors_list, as_cmap=True)

        sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
                    cmap=custom_cmap,
                    linewidths=1, linecolor=WHITE,
                    annot_kws={"size": 9, "fontweight": "bold"},
                    cbar_kws={"shrink": 0.8, "label": "Correlation"},
                    vmin=-1, vmax=1, center=0, square=True)

        for text in ax.texts:
            try:
                value = float(text.get_text())
                text.set_color(WHITE if abs(value) > 0.5 else NAVY)
            except:
                pass

        ax.set_title("Correlation Matrix", fontweight='bold', fontsize=14, pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    spacer()

    # Row 3: Platform chart
    section("Top 10 Platforms by Total Sales")
    platform_sales = (df.groupby("Platform")["Global_Sales"]
                      .sum().sort_values(ascending=False).head(10))

    fig, ax = styled_fig(12, 4.5)
    bar_colors = [GOLD if i == 0 else NAVY for i in range(len(platform_sales))]
    bars = ax.bar(range(len(platform_sales)), platform_sales.values,
                  color=bar_colors, edgecolor=WHITE, linewidth=1.5, alpha=0.95)
    ax.set_xticks(range(len(platform_sales)))
    ax.set_xticklabels(platform_sales.index, fontweight='bold', fontsize=11)
    ax.set_ylabel("Total Global Sales (millions)", fontweight='bold', fontsize=11)
    ax.set_xlabel("Platform", fontweight='bold', fontsize=11)
    ax.set_title("Which Platforms Generated the Most Sales? (Gold = #1)",
                 fontweight='bold', fontsize=14, pad=15)
    for bar, val in zip(bars, platform_sales.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}M', ha='center', va='bottom',
                fontsize=9, color=NAVY, fontweight='bold')
    ax.set_ylim(0, platform_sales.max() * 1.15)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    spacer()

    # Row 4: Box plot full width (horizontal)
    section("Sales Distribution by ESRB Rating (Box Plot)")

    fig, ax = styled_fig(12, 5)
    order = ["E","E10+","T","M"]
    bp = ax.boxplot(
        [df[df["Rating"]==r]["Global_Sales"].clip(upper=5).values for r in order],
        vert=False,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color=WHITE, linewidth=2.5),
        whiskerprops=dict(color=NAVY, linewidth=1.5),
        capprops=dict(color=NAVY, linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor=ACCENT, markersize=5,
                        markeredgecolor=NAVY, linestyle="none", alpha=0.6)
    )
    for patch, r in zip(bp["boxes"], order):
        patch.set_facecolor(RATING_COLORS[r])
        patch.set_alpha(0.85)

    ax.set_yticklabels(order, fontweight='bold', fontsize=11)
    ax.set_xlabel("Global Sales (millions, clipped at 5M)", fontweight='bold', fontsize=11)
    ax.set_ylabel("ESRB Rating", fontweight='bold', fontsize=11)
    ax.set_title("Sales Spread per Rating — Box = Middle 50% of Games, Dots = Outliers",
                 fontweight='bold', fontsize=14, pad=15)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# PAGE 3 — Descriptive Statistics
def show_descriptive(df):
    if df.empty:
        st.error("No data available.")
        return

    st.markdown(f"""
    <h1 style='font-family:"Playfair Display",serif; font-size:2.4rem;
               margin-bottom:2rem;'>Descriptive Statistics</h1>
    """, unsafe_allow_html=True)

    section("Summary Statistics by ESRB Rating")

    rows = []
    for rating in ["E","E10+","T","M"]:
        sub = df[df["Rating"] == rating]["Global_Sales"]
        if len(sub) == 0:
            continue
        n    = len(sub)
        mean = sub.mean()
        med  = sub.median()
        mode = sub.mode().iloc[0] if len(sub.mode()) > 0 else np.nan
        std  = sub.std()
        se   = stats.sem(sub) if n > 1 else 0
        if n > 1 and se > 0:
            ci_lo, ci_hi = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
        else:
            ci_lo, ci_hi = mean, mean
        rows.append({
            "Rating":      rating,
            "Count":       n,
            "Mean (M)":    round(mean, 3),
            "Median (M)":  round(med,  3),
            "Mode (M)":    round(mode, 3) if not pd.isna(mode) else "N/A",
            "Std Dev":     round(std,  3),
            "95% CI Low":  round(ci_lo,3),
            "95% CI High": round(ci_hi,3),
        })

    summary_df = pd.DataFrame(rows)

    styled_summary = summary_df.style.set_properties(**{
        'background-color': WHITE,
        'color': NAVY,
        'border-color': '#E2D9C8',
        'padding': '10px'
    }).set_table_styles([
        {'selector': 'thead th',
         'props': [('background-color', NAVY), ('color', BEIGE),
                   ('font-weight','bold'), ('padding','12px'),
                   ('text-align','center')]},
        {'selector': 'tbody td',
         'props': [('text-align','center'), ('padding','10px')]}
    ])
    st.dataframe(styled_summary, use_container_width=True, hide_index=True)

    spacer()

    # CI visualization
    section("Confidence Intervals Visualization")

    fig, ax = styled_fig(12, 5)
    for i, row in summary_df.iterrows():
        color = RATING_COLORS.get(row["Rating"], GOLD)
        ax.plot([row["95% CI Low"], row["95% CI High"]], [i+1, i+1],
                color=color, linewidth=10, alpha=0.3, solid_capstyle="round")
        ax.plot(row["Mean (M)"], i+1, "o",
                color=color, markersize=15,
                markeredgecolor=WHITE, markeredgewidth=2, zorder=5)
        ax.text(row["95% CI Low"] - 0.03, i+1,
                f'{row["95% CI Low"]:.3f}',
                ha='right', va='center', fontsize=9, color=NAVY, fontweight='bold')
        ax.text(row["95% CI High"] + 0.03, i+1,
                f'{row["95% CI High"]:.3f}',
                ha='left', va='center', fontsize=9, color=NAVY, fontweight='bold')
        ax.text(row["Mean (M)"], i+1.4,
                f'{row["Rating"]} (mean={row["Mean (M)"]:.3f}M)',
                ha='center', va='bottom', fontsize=11, color=NAVY, fontweight='bold')

    ax.set_yticks([1,2,3,4])
    ax.set_yticklabels([])
    ax.set_xlabel("Global Sales (millions)", fontweight='bold', fontsize=11)
    ax.set_title("95% Confidence Intervals for Mean Global Sales by Rating",
                 fontweight='bold', fontsize=14, pad=15)
    ax.set_ylim(0.5, 4.8)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # CI explanation
    m_row = summary_df[summary_df['Rating'] == 'M']
    if not m_row.empty:
        m_low  = m_row['95% CI Low'].values[0]
        m_high = m_row['95% CI High'].values[0]
        st.markdown(f"""
        <div class="callout">
            <h3>Understanding Confidence Intervals</h3>
            <p style='color:{BEIGE}; font-size:0.9rem; line-height:1.8; margin:0;'>
                A <b>95% Confidence Interval</b> represents the range where we expect
                the true population mean to fall with 95% confidence.
                The dot shows our sample mean; the bar width indicates variability.<br><br>
                For <b style='color:{GOLD}'>M-rated games</b>, we can say with 95% confidence
                that the true average global sales lies between
                <b>{m_low:.3f}M</b> and <b>{m_high:.3f}M</b> per game.
            </p>
        </div>
        """, unsafe_allow_html=True)

    spacer()

    # Hypothesis test
    section("Hypothesis Test — M-Rated vs E-Rated Sales")

    m_sales = df[df["Rating"] == "M"]["Global_Sales"].dropna()
    e_sales = df[df["Rating"] == "E"]["Global_Sales"].dropna()
    t_stat, p_value = stats.ttest_ind(m_sales, e_sales, equal_var=False)

    hc1, hc2, hc3 = st.columns(3)

    with hc1:
        st.markdown(f"""
        <div style='background:{BEIGE}; padding:18px; border-radius:10px; text-align:center;'>
            <p style='color:{NAVY}; font-size:13px; margin:0;'><b>H₀ Null Hypothesis</b></p>
            <p style='color:#2C2C2C; font-size:12px; margin:8px 0 0 0;'>
                M-rated and E-rated games have the <b>same</b> average global sales
            </p>
        </div>
        """, unsafe_allow_html=True)

    with hc2:
        st.markdown(f"""
        <div style='background:{NAVY}; padding:18px; border-radius:10px; text-align:center;'>
            <p style='color:{GOLD}; font-size:13px; margin:0;'><b>Welch's T-Test Results</b></p>
            <p style='color:{BEIGE}; font-size:13px; margin:8px 0 0 0;'>
                T-statistic: <b>{t_stat:.4f}</b><br>
                P-value: <b>{p_value:.6f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with hc3:
        if p_value < 0.05:
            verdict = "Reject H₀"
            explanation = "Difference IS statistically significant. M-rated games sell differently."
            bg = "#d4edda"
        else:
            verdict = "Fail to Reject H₀"
            explanation = "Difference is NOT statistically significant at α = 0.05."
            bg = "#f8d7da"

        st.markdown(f"""
        <div style='background:{bg}; padding:18px; border-radius:10px; text-align:center;'>
            <p style='color:{"#155724" if p_value < 0.05 else "#721c24"}; font-size:15px; margin:0;'>
                <b>{verdict}</b>
            </p>
            <p style='color:{"#155724" if p_value < 0.05 else "#721c24"}; font-size:12px; margin:8px 0 0 0;'>
                {explanation}
            </p>
        </div>
        """, unsafe_allow_html=True)

    spacer()

    section("Full Descriptive Statistics — All Numeric Variables")
    num_cols = ["Global_Sales","NA_Sales","EU_Sales","JP_Sales","Critic_Score","User_Score"]
    num_cols = [c for c in num_cols if c in df.columns]

    st.dataframe(
        df[num_cols].describe().round(3).style.set_properties(**{
            'background-color': WHITE,
            'color': NAVY,
            'border-color': '#E2D9C8'
        }).set_table_styles([
            {'selector': 'thead th',
             'props': [('background-color', NAVY), ('color', BEIGE),
                       ('font-weight','bold'), ('padding','12px')]},
            {'selector': 'tbody td', 'props': [('padding','10px')]}
        ]),
        use_container_width=True
    )

# PAGE 4 — Probability & Distributions
def show_probability(df):
    if df.empty:
        st.error("No data available.")
        return

    st.markdown(f"""
    <h1 style='font-family:"Playfair Display",serif; font-size:2.4rem;
               margin-bottom:2rem;'>Probability & Distributions</h1>
    """, unsafe_allow_html=True)

    # ── Pre-calculate all parameters ─────────────────────────
    sales        = df["Global_Sales"].dropna()
    sales_capped = sales[sales <= 10]

    mu_n, sigma_n  = stats.norm.fit(sales_capped)
    loc_e, scale_e = stats.expon.fit(sales_capped)
    lam_exp        = float(1 / scale_e)

    p_mature      = float((df["Rating"] == "M").mean())
    p_blockbuster = float((df["Global_Sales"] > 1.0).mean())

    # ══════════════════════════════════════════════════════════
    # SECTION 1 — DISTRIBUTION GALLERY
    # ══════════════════════════════════════════════════════════
    section("Distribution Gallery")

    st.markdown(f"""
    <p style='text-align:center; color:{MUTED}; margin-top:-1.5rem; margin-bottom:2rem;'>
        Each distribution is fitted using parameters calculated directly from the dataset.
    </p>
    """, unsafe_allow_html=True)

    # ── Row 1: Normal + Exponential ───────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        fig, ax = styled_fig(7, 4)
        x = np.linspace(0, 10, 300)
        ax.hist(sales_capped, bins=60, density=True,
                color=SLATE, alpha=0.35, edgecolor=NAVY,
                linewidth=0.4, label="Actual Sales")
        ax.plot(x, stats.norm.pdf(x, mu_n, sigma_n),
                color=GOLD, linewidth=3,
                label=f"Normal (μ={mu_n:.2f}, σ={sigma_n:.2f})")
        ax.axvline(mu_n, color=ACCENT, linewidth=1.8,
                   linestyle=':', alpha=0.8, label=f"Mean = {mu_n:.2f}M")
        ax.set_xlabel("Global Sales (millions)", fontweight='bold', fontsize=10)
        ax.set_ylabel("Density", fontweight='bold', fontsize=10)
        ax.set_title("Normal Distribution", fontweight='bold', fontsize=13, pad=12)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.set_xlim(0, 10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(f"""
        <div style='background:{WHITE}; border:1px solid #E2D9C8;
                    border-left:4px solid {GOLD}; border-radius:8px;
                    padding:1.2rem 1.4rem; margin-top:0.5rem;'>
            <p style='color:{NAVY}; font-size:13px; font-weight:700; margin:0 0 6px 0;'>
                Normal Distribution — N(μ, σ²)
            </p>
            <p style='color:{MUTED}; font-size:11px;
                      font-family:"JetBrains Mono",monospace; margin:0 0 8px 0;'>
                f(x) = (1 / σ√2π) · e^(−(x−μ)² / 2σ²)
            </p>
            <p style='color:{NAVY}; font-size:12px; margin:0; line-height:1.8;'>
                <b>μ (mean)</b> = {mu_n:.4f}M &nbsp;|&nbsp;
                <b>σ (std dev)</b> = {sigma_n:.4f}M<br>
                Describes symmetric spread around an average.
                Sales data is right-skewed so fit is approximate —
                most games cluster near zero with a long right tail.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig, ax = styled_fig(7, 4)
        ax.hist(sales_capped, bins=60, density=True,
                color=SLATE, alpha=0.35, edgecolor=NAVY,
                linewidth=0.4, label="Actual Sales")
        ax.plot(x, stats.expon.pdf(x, loc_e, scale_e),
                color="#E74C3C", linewidth=3, linestyle='--',
                label=f"Exponential (λ={lam_exp:.3f})")
        ax.axvline(scale_e, color=ACCENT, linewidth=1.8,
                   linestyle=':', alpha=0.8, label=f"Mean = {scale_e:.2f}M")
        ax.set_xlabel("Global Sales (millions)", fontweight='bold', fontsize=10)
        ax.set_ylabel("Density", fontweight='bold', fontsize=10)
        ax.set_title("Exponential Distribution", fontweight='bold', fontsize=13, pad=12)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.set_xlim(0, 10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(f"""
        <div style='background:{WHITE}; border:1px solid #E2D9C8;
                    border-left:4px solid #E74C3C; border-radius:8px;
                    padding:1.2rem 1.4rem; margin-top:0.5rem;'>
            <p style='color:{NAVY}; font-size:13px; font-weight:700; margin:0 0 6px 0;'>
                Exponential Distribution — Exp(λ)
            </p>
            <p style='color:{MUTED}; font-size:11px;
                      font-family:"JetBrains Mono",monospace; margin:0 0 8px 0;'>
                f(x) = λ · e^(−λx) &nbsp;&nbsp; for x ≥ 0
            </p>
            <p style='color:{NAVY}; font-size:12px; margin:0; line-height:1.8;'>
                <b>λ (rate)</b> = {lam_exp:.4f} &nbsp;|&nbsp;
                <b>Mean</b> = {scale_e:.4f}M<br>
                Best fit for this data. Most games sell very little
                while a few sell enormously — a classic
                exponential decay pattern.
            </p>
        </div>
        """, unsafe_allow_html=True)

    spacer()

    # ── Row 2: Binomial + Poisson ─────────────────────────────
    col3, col4 = st.columns(2, gap="large")

    with col3:
        n_binom  = st.slider("Binomial — Sample size (n):",
                             5, 50, 20, key="binom_n")
        k_range  = np.arange(0, n_binom + 1)
        binom_pmf = stats.binom.pmf(k_range, n=n_binom, p=p_mature)
        expected_k = int(round(n_binom * p_mature))

        fig, ax = styled_fig(7, 4)
        bar_colors = [GOLD if k == expected_k else NAVY for k in k_range]
        ax.bar(k_range, binom_pmf, color=bar_colors,
               edgecolor=WHITE, linewidth=0.8, alpha=0.9, width=0.7)
        ax.axvline(n_binom * p_mature, color=GOLD, linewidth=2,
                   linestyle='--', alpha=0.7,
                   label=f"Expected = {n_binom * p_mature:.1f}")
        ax.set_xlabel("Number of M-rated games (k)", fontweight='bold', fontsize=10)
        ax.set_ylabel("P(X = k)", fontweight='bold', fontsize=10)
        ax.set_title(f"Binomial  B(n={n_binom}, p={p_mature:.3f})",
                     fontweight='bold', fontsize=13, pad=12)
        ax.legend(fontsize=8.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        p_expected = float(stats.binom.pmf(expected_k, n=n_binom, p=p_mature))
        st.markdown(f"""
        <div style='background:{WHITE}; border:1px solid #E2D9C8;
                    border-left:4px solid {NAVY}; border-radius:8px;
                    padding:1.2rem 1.4rem; margin-top:0.5rem;'>
            <p style='color:{NAVY}; font-size:13px; font-weight:700; margin:0 0 6px 0;'>
                Binomial Distribution — B(n, p)
            </p>
            <p style='color:{MUTED}; font-size:11px;
                      font-family:"JetBrains Mono",monospace; margin:0 0 8px 0;'>
                P(X=k) = C(n,k) · pᵏ · (1−p)ⁿ⁻ᵏ
            </p>
            <p style='color:{NAVY}; font-size:12px; margin:0; line-height:1.8;'>
                <b>n</b> = {n_binom} games sampled &nbsp;|&nbsp;
                <b>p</b> = {p_mature:.4f} (M-rated proportion in dataset)<br>
                <b>Question:</b> In {n_binom} random games, how many are M-rated?<br>
                <b>Expected:</b> {n_binom * p_mature:.1f} &nbsp;|&nbsp;
                <b>P(X = {expected_k}) =
                <span style='color:{GOLD};'>{p_expected:.4f}</span></b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        batch_size  = st.slider("Poisson — Batch size (games):",
                                5, 30, 10, key="poisson_batch")
        lam_dynamic = float(p_blockbuster * batch_size)
        k_pois      = np.arange(0, min(batch_size + 1, 25))
        poisson_pmf = stats.poisson.pmf(k_pois, mu=lam_dynamic)
        mode_k      = int(lam_dynamic)

        fig, ax = styled_fig(7, 4)
        pois_colors = [GOLD if k == mode_k else SLATE for k in k_pois]
        ax.bar(k_pois, poisson_pmf, color=pois_colors,
               edgecolor=WHITE, linewidth=0.8, alpha=0.9, width=0.7)
        ax.axvline(lam_dynamic, color="#E74C3C", linewidth=2,
                   linestyle='--', alpha=0.8, label=f"λ = {lam_dynamic:.2f}")
        ax.set_xlabel("Blockbuster games in batch (k)", fontweight='bold', fontsize=10)
        ax.set_ylabel("P(X = k)", fontweight='bold', fontsize=10)
        ax.set_title(f"Poisson  Pois(λ={lam_dynamic:.2f})",
                     fontweight='bold', fontsize=13, pad=12)
        ax.legend(fontsize=8.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        p_mode = float(stats.poisson.pmf(mode_k, mu=lam_dynamic))
        st.markdown(f"""
        <div style='background:{WHITE}; border:1px solid #E2D9C8;
                    border-left:4px solid #E74C3C; border-radius:8px;
                    padding:1.2rem 1.4rem; margin-top:0.5rem;'>
            <p style='color:{NAVY}; font-size:13px; font-weight:700; margin:0 0 6px 0;'>
                Poisson Distribution — Pois(λ)
            </p>
            <p style='color:{MUTED}; font-size:11px;
                      font-family:"JetBrains Mono",monospace; margin:0 0 8px 0;'>
                P(X=k) = (λᵏ · e^−λ) / k!
            </p>
            <p style='color:{NAVY}; font-size:12px; margin:0; line-height:1.8;'>
                <b>λ</b> = {lam_dynamic:.4f} &nbsp;|&nbsp;
                <b>Batch</b> = {batch_size} games<br>
                <b>Question:</b> In {batch_size} games, how many sell &gt;1M?<br>
                <b>Blockbuster rate:</b> {p_blockbuster*100:.1f}% of all games &nbsp;|&nbsp;
                <b>P(X = {mode_k}) =
                <span style='color:{GOLD};'>{p_mode:.4f}</span></b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    spacer()

    # ══════════════════════════════════════════════════════════
    # SECTION 2 — CONDITIONAL PROBABILITY
    # ══════════════════════════════════════════════════════════
    section("Conditional Probability")

    st.markdown(f"""
    <p style='text-align:center; color:{MUTED}; margin-top:-1.5rem; margin-bottom:2rem;'>
        P(Event | Rating) — probability of a sales outcome given a specific ESRB rating
    </p>
    """, unsafe_allow_html=True)

    thresholds = [0.5, 1.0, 2.0, 5.0]
    ratings    = ["E", "E10+", "T", "M"]

    cond_rows = []
    for rating in ratings:
        group = df[df["Rating"] == rating]["Global_Sales"]
        n     = len(group)
        row   = {"Rating": rating, "Games": f"{n:,}"}
        for t in thresholds:
            p = float((group > t).mean())
            row[f"P(Sales > {t}M)"] = f"{p*100:.1f}%"
        cond_rows.append(row)

    cond_df = pd.DataFrame(cond_rows)
    st.dataframe(cond_df, hide_index=True, use_container_width=True)

    # Visual comparison for threshold 1M
    st.markdown("<br>", unsafe_allow_html=True)
    cp_cols = st.columns(4, gap="large")
    for i, rating in enumerate(ratings):
        group = df[df["Rating"] == rating]["Global_Sales"]
        p_val = float((group > 1.0).mean())
        color = RATING_COLORS.get(rating, GOLD)
        with cp_cols[i]:
            st.markdown(f"""
            <div style='background:{WHITE}; border:1px solid #E2D9C8;
                        border-top:4px solid {color}; border-radius:8px;
                        padding:1.2rem; text-align:center;'>
                <p style='color:{MUTED}; font-size:11px; margin:0;
                          text-transform:uppercase; letter-spacing:0.06em;'>
                    P(Sales &gt; 1M | {rating})
                </p>
                <p style='color:{NAVY}; font-size:1.8rem; font-weight:700;
                          margin:8px 0; font-family:"Playfair Display",serif;'>
                    {p_val*100:.1f}%
                </p>
                <p style='color:{MUTED}; font-size:11px; margin:0;'>
                    {int(p_val * len(group)):,} of {len(group):,} games
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:{BEIGE}; padding:14px 18px; border-radius:8px;
                border-left:3px solid {GOLD}; margin-top:1.2rem;'>
        <p style='color:{NAVY}; font-size:12px; margin:0; line-height:1.8;'>
            <b>How to read this:</b> P(Sales &gt; 1M | M) means
            "given the game is M-rated, what is the probability it sells more than 1M copies?"
            This is conditional probability — we restrict our sample to one rating group
            and measure the event probability within that group only.
        </p>
    </div>
    """, unsafe_allow_html=True)

    spacer()

    # ══════════════════════════════════════════════════════════
    # SECTION 3 — BAYES THEOREM
    # ══════════════════════════════════════════════════════════
    section("Bayes' Theorem")

    st.markdown(f"""
    <p style='text-align:center; color:{MUTED}; margin-top:-1.5rem; margin-bottom:2rem;'>
        If a game is a blockbuster, how likely is it M-rated? Bayes flips the question.
    </p>
    """, unsafe_allow_html=True)

    bayes_col1, bayes_col2 = st.columns([1, 1], gap="large")

    with bayes_col1:
        st.markdown(f"""
        <div style='background:{WHITE}; border:1px solid #E2D9C8;
                    border-left:4px solid {GOLD}; border-radius:8px;
                    padding:1.5rem; margin-bottom:1rem;'>
            <p style='color:{NAVY}; font-size:13px; font-weight:700;
                      margin:0 0 10px 0;'>Bayes Formula</p>
            <p style='color:{MUTED}; font-size:12px;
                      font-family:"JetBrains Mono",monospace;
                      margin:0 0 12px 0; line-height:2;'>
                P(A|B) = P(B|A) · P(A) / P(B)
            </p>
            <p style='color:{NAVY}; font-size:12px; margin:0; line-height:1.8;'>
                <b>Question:</b> Given a game sold &gt;1M copies,
                what is the probability it was M-rated?<br><br>
                <b>A</b> = Game is M-rated<br>
                <b>B</b> = Game sold &gt; 1M copies
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Threshold slider
        bayes_threshold = st.slider(
            "Sales threshold (millions):",
            0.5, 5.0, 1.0, 0.5, key="bayes_thresh")

        # Calculate all components
        p_m          = float((df["Rating"] == "M").mean())
        p_b_given_m  = float((df[df["Rating"] == "M"]["Global_Sales"] > bayes_threshold).mean())
        p_b          = float((df["Global_Sales"] > bayes_threshold).mean())
        p_m_given_b  = float((p_b_given_m * p_m) / p_b) if p_b > 0 else 0.0

        st.markdown(f"""
        <div style='background:{BEIGE}; padding:14px 18px; border-radius:8px;
                    border-left:3px solid {GOLD}; margin-top:0.5rem;'>
            <p style='color:{NAVY}; font-size:12px; margin:0; line-height:2;'>
                <b>P(A)</b> = P(M-rated) =
                <b style='color:{GOLD};'>{p_m:.4f}</b>
                ({p_m*100:.1f}%)<br>
                <b>P(B|A)</b> = P(Sales &gt; {bayes_threshold}M | M-rated) =
                <b style='color:{GOLD};'>{p_b_given_m:.4f}</b>
                ({p_b_given_m*100:.1f}%)<br>
                <b>P(B)</b> = P(Sales &gt; {bayes_threshold}M) =
                <b style='color:{GOLD};'>{p_b:.4f}</b>
                ({p_b*100:.1f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with bayes_col2:
        st.markdown(f"""
        <div style='background:{NAVY}; padding:2rem; border-radius:12px;
                    text-align:center; border:1px solid {GOLD}33;
                    margin-bottom:1rem;'>
            <p style='color:{MUTED}; font-size:13px; margin:0;'>
                P(M-rated | Sales &gt; {bayes_threshold}M)
            </p>
            <p style='color:{GOLD}; font-size:3.5rem; margin:12px 0;
                      font-weight:700; font-family:"Playfair Display",serif;'>
                {p_m_given_b*100:.1f}%
            </p>
            <p style='color:{BEIGE}; font-size:12px; margin:0;'>
                probability of being M-rated given high sales
            </p>
        </div>
        """, unsafe_allow_html=True)

        # All ratings via Bayes
        bayes_rows = []
        for rating in ratings:
            p_r         = float((df["Rating"] == rating).mean())
            p_b_given_r = float((df[df["Rating"] == rating]["Global_Sales"]
                                 > bayes_threshold).mean())
            p_r_given_b = float((p_b_given_r * p_r) / p_b) if p_b > 0 else 0.0
            bayes_rows.append({
                "Rating":         rating,
                "P(Rating)":      f"{p_r*100:.1f}%",
                f"P(Sales>{bayes_threshold}M|Rating)": f"{p_b_given_r*100:.1f}%",
                f"P(Rating|Sales>{bayes_threshold}M)": f"{p_r_given_b*100:.1f}%"
            })

        bayes_df = pd.DataFrame(bayes_rows)
        st.dataframe(bayes_df, hide_index=True, use_container_width=True)

        st.markdown(f"""
        <div style='background:{BEIGE}; padding:12px 16px; border-radius:8px;
                    border-left:3px solid {GOLD}; margin-top:0.8rem;'>
            <p style='color:{NAVY}; font-size:12px; margin:0; line-height:1.8;'>
                The last column answers: among all games that sold &gt;{bayes_threshold}M,
                what fraction belonged to each rating?
                This is the <b>posterior probability</b> — updated belief after
                observing the evidence (high sales).
            </p>
        </div>
        """, unsafe_allow_html=True)

    spacer()

    # ══════════════════════════════════════════════════════════
    # SECTION 4 — PROBABILITY CALCULATOR
    # ══════════════════════════════════════════════════════════
    section("Probability Calculator")

    st.markdown(f"""
    <p style='text-align:center; color:{MUTED}; margin-top:-1.5rem; margin-bottom:2rem;'>
        Using the Normal distribution fitted on global sales data.
    </p>
    """, unsafe_allow_html=True)

    calc_col1, calc_col2 = st.columns([1, 2], gap="large")

    with calc_col1:
        threshold = st.slider("P(Global Sales > X million):",
                              0.1, 10.0, 1.0, 0.1, key="prob_slider")

        prob_theoretical = float(max(0, min(1,
                            1 - stats.norm.cdf(threshold, mu_n, sigma_n))))
        prob_empirical   = float((df["Global_Sales"] > threshold).mean())
        count_above      = int((df["Global_Sales"] > threshold).sum())

        st.markdown(metric_card(
            "Theoretical P (Normal)",
            f"{prob_theoretical*100:.2f}%",
            f"P(X > {threshold}M) under Normal fit"
        ), unsafe_allow_html=True)

        st.markdown(metric_card(
            "Empirical P (Actual Data)",
            f"{prob_empirical*100:.2f}%",
            f"{count_above:,} of {len(df):,} games"
        ), unsafe_allow_html=True)

    with calc_col2:
        rows_prob = []
        for label, lo, hi in [
            ("P(Sales < 0.5M)",  0,   0.5),
            ("P(Sales 0.5–1M)",  0.5, 1.0),
            ("P(Sales 1M–5M)",   1.0, 5.0),
            ("P(Sales > 5M)",    5.0, 999),
        ]:
            if hi == 999:
                emp = float((df["Global_Sales"] > lo).mean())
                the = float(max(0, 1 - stats.norm.cdf(lo, mu_n, sigma_n)))
            else:
                emp = float(((df["Global_Sales"] >= lo) &
                             (df["Global_Sales"] < hi)).mean())
                the = float(max(0, stats.norm.cdf(hi, mu_n, sigma_n)
                                - stats.norm.cdf(lo, mu_n, sigma_n)))
            rows_prob.append({
                "Range":       label,
                "Empirical":   f"{emp*100:.2f}%",
                "Theoretical": f"{the*100:.2f}%",
                "Count":       f"{int(emp * len(df)):,}"
            })

        st.dataframe(pd.DataFrame(rows_prob),
                     hide_index=True, use_container_width=True)

        st.markdown(f"""
        <div style='background:{BEIGE}; padding:12px 16px; border-radius:8px;
                    border-left:3px solid {GOLD}; margin-top:0.8rem;'>
            <p style='color:{NAVY}; font-size:12px; margin:0; line-height:1.8;'>
                <b>Empirical</b> = actual proportion from {len(df):,} games<br>
                <b>Theoretical</b> = calculated from fitted
                Normal(μ={mu_n:.2f}, σ={sigma_n:.2f})<br>
                The gap confirms sales data is <b>right-skewed</b>,
                not perfectly normal.
            </p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 5 — Regression & Prediction
def show_regression(df):
    if df.empty:
        st.error("No data available.")
        return

    st.markdown(f"""
    <h1 style='font-family:"Playfair Display",serif; font-size:2.4rem;
               margin-bottom:2rem;'>Regression & Prediction</h1>
    """, unsafe_allow_html=True)

    violence_map = {"E": 0, "E10+": 1, "T": 2, "M": 3}
    df = df.copy()
    df["Violence_Level"] = df["Rating"].map(violence_map)

    multi_df = df[["Global_Sales", "Violence_Level", "Critic_Score", "Genre"]].dropna()
    genre_dummies = pd.get_dummies(multi_df["Genre"], prefix="Genre")
    multi_df_encoded = pd.concat(
        [multi_df[["Global_Sales", "Violence_Level", "Critic_Score"]], genre_dummies], axis=1)

    top_genres = df["Genre"].value_counts().head(6).index.tolist()
    genre_cols = [f"Genre_{g}" for g in top_genres if f"Genre_{g}" in multi_df_encoded.columns]
    feature_cols = ["Violence_Level", "Critic_Score"] + genre_cols

    X_multi = multi_df_encoded[feature_cols].values.astype(float)
    y_multi = multi_df_encoded["Global_Sales"].values.astype(float)

    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

    X_simple = X_multi[:, 1].reshape(-1, 1)
    simple_model = LinearRegression().fit(X_simple, y_multi)
    r2_simple = r2_score(y_multi, simple_model.predict(X_simple))
    rmse_simple = float(np.sqrt(mean_squared_error(y_multi, simple_model.predict(X_simple))))

    multi_model = LinearRegression().fit(X_multi, y_multi)
    r2_multi = r2_score(y_multi, multi_model.predict(X_multi))
    rmse_multi = float(np.sqrt(mean_squared_error(y_multi, multi_model.predict(X_multi))))
    v_coef = float(multi_model.coef_[0])
    c_coef = float(multi_model.coef_[1])

    y_pred_test_multi = multi_model.predict(Xm_te)
    residuals = ym_te - y_pred_test_multi

    # SECTION 1: SIMPLE REGRESSION
    section("Simple Regression: Critic Score to Global Sales")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        fig, ax = styled_fig(8, 5.5)
        ax.scatter(X_simple, y_multi, color=SLATE, alpha=0.15, s=30,
                   edgecolor=None, label=f"Games (n={len(X_simple):,})")
        x_line = np.linspace(float(X_simple.min()), float(X_simple.max()), 200).reshape(-1, 1)
        y_line = simple_model.predict(x_line)
        ax.plot(x_line, y_line, color=GOLD, linewidth=3, label="Regression Line", zorder=5)
        ax.fill_between(x_line.ravel(), y_line - rmse_simple, y_line + rmse_simple,
                        color=GOLD, alpha=0.12, label="Error Band")
        ax.set_xlabel("Critic Score (0-100)", fontweight="bold", fontsize=11)
        ax.set_ylabel("Global Sales (millions)", fontweight="bold", fontsize=11)
        ax.set_title("Critic Score vs Global Sales", fontweight="bold", fontsize=14, pad=15)
        ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        with st.container():
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(metric_card("R2 Score", f"{r2_simple:.4f}",
                                        f"Explains {r2_simple*100:.1f}%"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("RMSE", f"{rmse_simple:.3f}M",
                                        "Prediction error"), unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                st.markdown(metric_card("Slope", f"{simple_model.coef_[0]:.4f}",
                                        "Sales per score point"), unsafe_allow_html=True)
            with c4:
                st.markdown(metric_card("Intercept", f"{simple_model.intercept_:.3f}M",
                                        "Baseline"), unsafe_allow_html=True)

    spacer()

    # SECTION 2: INTERACTIVE PREDICTOR
    section("Interactive Predictor")

    col_p1, col_p2 = st.columns([1, 2], gap="large")

    with col_p1:
        user_score = st.slider("Critic Score:", 0, 100, 75, key="pred_score")
        predicted = float(max(0, simple_model.predict([[user_score]])[0]))
        nearby = y_multi[(X_simple.ravel() >= user_score - 5) & (X_simple.ravel() <= user_score + 5)]
        median_ref = float(np.median(nearby)) if len(nearby) > 0 else 0.0

        st.markdown(f"""
        <div class="callout" style='text-align:center;'>
            <h3>Predicted Sales</h3>
            <p style='color:{BEIGE}; font-size:2rem; margin:15px 0;'>
                <b style='color:{GOLD};'>{predicted:.2f}M copies</b>
            </p>
            <p style='color:{BEIGE}; font-size:0.85rem; margin:0;'>
                Score: {user_score}/100 | Median: {median_ref:.2f}M
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        if len(nearby) > 0:
            avg_sim = float(nearby.mean())
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:{WHITE}; border:1px solid #E2D9C8;
                        border-radius:8px; padding:2rem;'>
                <h4 style='color:{NAVY}; margin-top:0;'>Similar Games ({user_score-5} to {user_score+5})</h4>
                <p style='color:{NAVY}; line-height:2;'>
                    <b>Count:</b> {len(nearby):,}<br>
                    <b>Average sales:</b> {avg_sim:.3f}M<br>
                    <b>Median sales:</b> {median_ref:.3f}M<br>
                    <b>Prediction:</b> {predicted:.3f}M
                </p>
            </div>
            """, unsafe_allow_html=True)

    spacer()

    # SECTION 3: MULTIPLE REGRESSION
    section("Multiple Regression: Violence + Genre + Critic Score")

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(metric_card("R2 Score", f"{r2_multi:.4f}",
                                    f"Explains {r2_multi*100:.1f}%"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("RMSE", f"{rmse_multi:.3f}M",
                                    "Prediction error"), unsafe_allow_html=True)

        coef_data = []
        for name, coef in [("Violence Level", v_coef), ("Critic Score", c_coef)]:
            coef_data.append({"Predictor": name, "Coefficient": f"{coef:+.4f}",
                              "Direction": "Positive" if coef > 0 else "Negative"})
        for i, genre in enumerate(top_genres[:6]):
            if 2 + i < len(multi_model.coef_):
                coef = float(multi_model.coef_[2 + i])
                coef_data.append({"Predictor": f"Genre: {genre}",
                                  "Coefficient": f"{coef:+.4f}",
                                  "Direction": "Positive" if coef > 0 else "Negative"})
        st.dataframe(pd.DataFrame(coef_data), hide_index=True, use_container_width=True)

    with col_b:
        improvement = r2_multi - r2_simple

        st.markdown(f"""
        <div style='background:{BEIGE}; padding:1.5rem; border-radius:10px;
                    text-align:center; margin-bottom:1rem; border:1px solid #E2D9C8;'>
            <p style='color:{MUTED}; font-size:11px; margin:0;'>MODEL 1 - CRITIC SCORE ONLY</p>
            <p style='color:{NAVY}; font-size:2rem; margin:8px 0; font-weight:bold;'>R2 = {r2_simple:.4f}</p>
        </div>
        <p style='text-align:center; color:{GOLD}; font-size:1.05rem; margin:0.3rem 0;'>
            Adding Genre and Violence<br>
            <span style='font-size:11px;'>(+{improvement*100:.2f}% R2 change)</span>
        </p>
        <div style='background:{NAVY}; padding:1.5rem; border-radius:10px;
                    text-align:center; margin-top:0.8rem;'>
            <p style='color:{GOLD}; font-size:11px; margin:0;'>MODEL 2 - CRITIC + GENRE + VIOLENCE</p>
            <p style='color:{GOLD}; font-size:2rem; margin:8px 0; font-weight:bold;'>R2 = {r2_multi:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    spacer()

    # SECTION 4: PREDICT GAME SALES
    section("Predict Game Sales")

    col_p1, col_p2 = st.columns([1, 1], gap="large")

    with col_p1:
        violence_options = {"E (Everyone)": 0, "E10+": 1, "T (Teen)": 2, "M (Mature)": 3}
        violence_choice = st.selectbox("ESRB Rating:", list(violence_options.keys()))
        violence_val = violence_options[violence_choice]
        critic_val = st.slider("Critic Score:", 0, 100, 75, key="pred_multi")
        genre_choice = st.selectbox("Genre:", top_genres[:6])

        input_dict = {"Violence_Level": float(violence_val), "Critic_Score": float(critic_val)}
        for g in top_genres[:6]:
            input_dict[f"Genre_{g}"] = 1.0 if g == genre_choice else 0.0
        input_arr = pd.DataFrame([input_dict])[feature_cols].values.astype(float)
        pred_sales = float(max(0.0, multi_model.predict(input_arr)[0]))

    with col_p2:
        if pred_sales < 0.5:
            tier, tier_color, txt_color = "Low Sales", "#dc3545", "#FFFFFF"
        elif pred_sales < 2:
            tier, tier_color, txt_color = "Average", GOLD, NAVY
        elif pred_sales < 5:
            tier, tier_color, txt_color = "Strong", "#28a745", "#FFFFFF"
        else:
            tier, tier_color, txt_color = "Blockbuster", NAVY, GOLD

        st.markdown(f"""
        <div style='background:{NAVY}; padding:30px; border-radius:12px;
                    text-align:center; border:1px solid {GOLD}33; margin-top:1.5rem;'>
            <p style='color:{MUTED}; font-size:14px; margin:0;'>Predicted Global Sales</p>
            <p style='color:{GOLD}; font-size:48px; margin:10px 0;
                      font-weight:bold; font-family:"Playfair Display",serif;'>
                {pred_sales:.2f}M</p>
            <p style='color:{BEIGE}; font-size:13px; margin:0;'>
                {violence_choice} | {genre_choice} | Score: {critic_val}/100</p>
        </div>
        <div style='background:{tier_color}; padding:12px; border-radius:8px;
                    text-align:center; margin-top:10px;'>
            <p style='color:{txt_color}; font-size:16px; margin:0; font-weight:bold;'>{tier}</p>
        </div>
        """, unsafe_allow_html=True)

    spacer()

    # SECTION 5: RESIDUAL ANALYSIS
    section("Residual Analysis")

    rc1, rc2 = st.columns(2, gap="large")

    with rc1:
        fig, ax = styled_fig(7, 4.5)
        ax.scatter(y_pred_test_multi, residuals, color=SLATE, alpha=0.25, s=20, edgecolor=None)
        ax.axhline(0, color=GOLD, linewidth=2, linestyle="--", alpha=0.9, label="Zero")
        sorted_idx = np.argsort(y_pred_test_multi)
        z = np.polyfit(y_pred_test_multi[sorted_idx], residuals[sorted_idx], 2)
        ax.plot(y_pred_test_multi[sorted_idx], np.poly1d(z)(y_pred_test_multi[sorted_idx]),
                color=ACCENT, linewidth=2.5, alpha=0.8, label="Trend")
        ax.set_xlabel("Fitted Values (millions)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Residuals (millions)", fontweight="bold", fontsize=10)
        ax.set_title("Residuals vs Fitted", fontweight="bold", fontsize=13, pad=12)
        ax.legend(fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with rc2:
        r_mean = float(residuals.mean())
        r_std = float(residuals.std())
        fig, ax = styled_fig(7, 4.5)
        ax.hist(residuals, bins=50, color=SLATE, alpha=0.5,
                edgecolor=NAVY, linewidth=0.4, density=True, label="Residuals")
        x_r = np.linspace(float(residuals.min()), float(residuals.max()), 200)
        ax.plot(x_r, stats.norm.pdf(x_r, r_mean, r_std),
                color=GOLD, linewidth=2.5, label="Normal fit")
        ax.set_xlabel("Residuals (millions)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Density", fontweight="bold", fontsize=10)
        ax.set_title("Residual Distribution", fontweight="bold", fontsize=13, pad=12)
        ax.legend(fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# PAGE 6 — Conclusion
def show_conclusion(df):
    if df.empty:
        st.error("No data available.")
        return

    st.markdown(f"""
    <h1 style='font-family:"Playfair Display",serif; font-size:2.4rem;
               margin-bottom:2rem;'>Conclusion</h1>
    """, unsafe_allow_html=True)

    m_mean   = df[df["Rating"]=="M"]["Global_Sales"].mean()
    e_mean   = df[df["Rating"]=="E"]["Global_Sales"].mean()
    t_mean   = df[df["Rating"]=="T"]["Global_Sales"].mean()
    all_mean = df["Global_Sales"].mean()
    m_count  = (df["Rating"]=="M").sum()

    m_sales   = df[df["Rating"]=="M"]["Global_Sales"].values
    non_m     = df[df["Rating"]!="M"]["Global_Sales"].values
    t_stat, p_val = stats.ttest_ind(m_sales, non_m, equal_var=False)
    is_sig = p_val < 0.05

    # Top metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("M-Rated Avg Sales", f"{m_mean:.3f}M",
                                f"n = {m_count:,} games"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("E-Rated Avg Sales", f"{e_mean:.3f}M",
                                f"n = {(df['Rating']=='E').sum():,} games"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Overall Average", f"{all_mean:.3f}M",
                                f"Across {len(df):,} games"), unsafe_allow_html=True)
    with c4:
        diff_pct = ((m_mean - all_mean) / all_mean) * 100
        st.markdown(metric_card("M vs Overall",
                                f"{diff_pct:+.1f}%",
                                f"{'Higher' if diff_pct > 0 else 'Lower'} than average"),
                    unsafe_allow_html=True)

    spacer()

    # Verdict
    verdict_icon = "YES" if is_sig else "MIXED"
    st.markdown(f"""
    <div class="callout" style='text-align:center;'>
        <h3>Do Violent Video Games Sell More?</h3>
        <p style='color:{BEIGE}; font-size:1.5rem; margin:20px 0;'>
            <span class="verdict">{verdict_icon}</span>
        </p>
        <p style='color:{BEIGE}; font-size:0.95rem; line-height:1.8; margin:0;'>
            Our analysis of <b>{len(df):,} video games</b> shows M-rated games average
            <b style='color:{GOLD}'>{m_mean:.3f}M</b> in global sales, vs the overall
            average of {all_mean:.3f}M. This difference is
            <b style='color:{GOLD}'>
                {'statistically significant (p = ' + f'{p_val:.4f})' if is_sig
                 else 'not statistically significant'}
            </b>. However, a small number of blockbuster titles like GTA V
            significantly inflate M-rated averages — genre and franchise power
            are likely the true drivers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    spacer()

    # Key findings
    st.markdown(f"<h2 style='color:{NAVY}; text-align:center; margin-bottom:2rem;'>Key Findings</h2>",
                unsafe_allow_html=True)

    findings = [
        ("Sales & Ratings",
         f"M-rated games (mean={m_mean:.3f}M) "
         f"{'outperform' if m_mean > e_mean else 'do not outperform'} "
         f"E-rated games (mean={e_mean:.3f}M). Genre and franchise effects may explain this gap."),
        ("Statistical Significance",
         f"Welch's t-test {'confirms' if is_sig else 'does not confirm'} significant difference "
         f"(p={p_val:.4f}), suggesting the gap is "
         f"{'unlikely' if is_sig else 'possibly'} due to chance."),
        ("Critic Scores & Sales",
         "Positive but weak linear relationship (R² ≈ 0.06). "
         "Critical reception alone cannot reliably predict commercial success."),
        ("Distribution Pattern",
         "Sales data is heavily right-skewed. A small number of blockbusters dominate "
         "revenue regardless of rating category. Exponential fit describes this better."),
        ("Genre Confounding",
         "Action and Shooter genres — predominantly M-rated — dominate total sales. "
         "Genre effect may drive the rating-sales correlation more than violence itself."),
        ("Practical Implications",
         "ESRB rating alone is insufficient to predict sales. "
         "Marketing, franchise power, platform, and genre matter significantly more."),
    ]

    for i in range(0, len(findings), 3):
        cols = st.columns(3, gap="large")
        for j in range(3):
            if i + j < len(findings):
                title, content = findings[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div style='background:{WHITE};
                                border:1px solid #E2D9C8;
                                border-left:4px solid {GOLD};
                                border-radius:8px;
                                padding:1.8rem;
                                margin-bottom:1.5rem;
                                box-shadow:0 2px 4px rgba(0,0,0,0.05);'>
                        <h3 style='color:{NAVY}; margin-top:0;
                                   margin-bottom:1rem; font-size:1.1rem;'>
                            {title}
                        </h3>
                        <p style='color:{NAVY}; font-size:0.9rem;
                                  line-height:1.7; margin:0;'>
                            {content}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    spacer()

    # Limitations
    st.markdown(f"<h2 style='color:{NAVY}; text-align:center; margin-bottom:2rem;'>Limitations</h2>",
                unsafe_allow_html=True)

    limitations = [
        ("Data Coverage",
         "Dataset covers games up to 2016 only. Modern titles like Fortnite and "
         "Elden Ring are not included, which may change conclusions."),
        ("Rating System",
         "ESRB ratings are region-specific. Games sold in Japan or Europe use "
         "different rating systems (CERO, PEGI) not captured here."),
        ("Data Completeness",
         "Sales data may be incomplete for smaller or indie titles. "
         "Large franchises are better represented, causing selection bias."),
        ("Causation vs Correlation",
         "M-rated games selling more does not mean violence causes higher sales. "
         "Marketing budget and franchise size are likely the real drivers."),
    ]

    lc1, lc2 = st.columns(2, gap="large")
    for i, (title, text) in enumerate(limitations):
        with lc1 if i % 2 == 0 else lc2:
            st.markdown(f"""
            <div style='background:#fff3cd; padding:15px; border-radius:8px;
                        margin-bottom:10px; border-left:3px solid {GOLD};'>
                <p style='color:{NAVY}; font-size:13px; font-weight:600; margin:0 0 5px 0;'>
                    {title}
                </p>
                <p style='color:#2C2C2C; font-size:12px; margin:0;'>{text}</p>
            </div>
            """, unsafe_allow_html=True)

    spacer()

    # Statistical methods
    st.markdown(f"<h2 style='color:{NAVY}; text-align:center; margin-bottom:2rem;'>Statistical Methods Used</h2>",
                unsafe_allow_html=True)

    mc1, mc2 = st.columns(2, gap="large")

    methods_left = [
        ("Descriptive Statistics",
         "Mean, median, mode, standard deviation, confidence intervals"),
        ("Graphical Analysis",
         "Bar charts, box plots, pie charts, heatmaps, KDE"),
        ("Probability Theory",
         "Normal & exponential distribution fitting, probability calculations, Shapiro-Wilk test"),
    ]
    methods_right = [
        ("Hypothesis Testing",
         "Welch's two-sample t-test for mean comparison"),
        ("Regression Modeling",
         "Simple & multiple OLS Linear Regression with R², RMSE, residual analysis"),
        ("Python Libraries",
         "Streamlit, Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn"),
    ]

    for method, detail in methods_left:
        mc1.markdown(f"""
        <div style='background:{WHITE}; padding:1.2rem; border-radius:8px;
                    margin-bottom:1.2rem; border-left:3px solid {GOLD};'>
            <b style='color:{NAVY}; font-size:1rem;'>{method}</b>
            <div style='color:{MUTED}; font-size:0.9rem; margin-top:6px;'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    for method, detail in methods_right:
        mc2.markdown(f"""
        <div style='background:{WHITE}; padding:1.2rem; border-radius:8px;
                    margin-bottom:1.2rem; border-left:3px solid {GOLD};'>
            <b style='color:{NAVY}; font-size:1rem;'>{method}</b>
            <div style='color:{MUTED}; font-size:0.9rem; margin-top:6px;'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:{NAVY}; border:2px solid {GOLD}; border-radius:10px;
                padding:2rem; text-align:center; color:{BEIGE}; margin-top:2rem;'>
        <p style='font-size:1.1rem; margin:0; font-family:"Playfair Display",serif;'>
            Probability & Statistics Semester Project · Spring 2026
        </p>
        <p style='font-size:0.85rem; margin-top:10px; opacity:0.8;'>
            Data Source: Kaggle Video Game Sales with Ratings ·
            Python & Streamlit Analytics Platform
        </p>
    </div>
    <div class='footer'>
        Probability & Statistics — Spring 2026 · Pentagon Group
    </div>
    """, unsafe_allow_html=True)

# Main
def main():
    load_css()
    df = load_data()

    if df.empty:
        st.error("""
        ### Dataset Not Found
        Please ensure:
        1. The file `Video_Games_Sales_as_at_22_Dec_2016.csv` is in the same folder as `app.py`
        2. The file is not corrupted
        3. The file has the expected columns
        """)
        return

    page = show_sidebar(df)

    if   page == "Home":                        show_home(df)
    elif page == "Graphical Analysis":          show_graphs(df)
    elif page == "Descriptive Statistics":      show_descriptive(df)
    elif page == "Probability & Distributions": show_probability(df)
    elif page == "Regression & Prediction":     show_regression(df)
    elif page == "Conclusion":                  show_conclusion(df)

if __name__ == "__main__":
    main()