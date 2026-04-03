import os
import json
import streamlit as st

LOGO_PATH = 'BKAI_Logo.png'
USERS_FILE = 'users.json'

st.set_page_config(
    page_title='BKAI Crack Analysis Portal',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded'
)


def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


users = load_users()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'


def inject_styles():
    st.markdown(
        """
        <style>
        :root{
            --bg:#dfe7f3;
            --panel:#ffffff;
            --panel-2:#f8fbff;
            --line:#d8e1ef;
            --text:#172033;
            --muted:#6d7b95;
            --blue1:#4d92ff;
            --blue2:#2e6ee8;
            --blue3:#1f5bd3;
            --accent:#4fc3ff;
            --danger:#ea6b78;
            --success:#27ae60;
            --shadow:0 24px 60px rgba(34,55,100,.16);
        }

        .stApp{
            background:
                radial-gradient(circle at top left, rgba(255,255,255,.95), rgba(223,231,243,.78) 22%, transparent 45%),
                linear-gradient(180deg, #e9eff8 0%, #dfe7f3 100%);
        }

        html, body, [class*="css"]{
            font-family: Inter, Arial, Helvetica, sans-serif;
            color:var(--text);
        }

        .block-container{
            max-width: 1220px;
            padding-top: 24px;
            padding-bottom: 32px;
        }

        div[data-testid="stSidebar"]{
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
            border-right:1px solid var(--line);
        }

        .app-shell{
            border:1.5px solid #cfd9ea;
            border-radius:30px;
            overflow:hidden;
            box-shadow: var(--shadow);
            background: rgba(255,255,255,.38);
            backdrop-filter: blur(10px);
        }

        .hero{
            position:relative;
            overflow:hidden;
            padding:34px 40px 110px;
            background: linear-gradient(135deg, var(--blue1) 0%, var(--blue2) 52%, var(--blue3) 100%);
        }

        .hero::before{
            content:"";
            position:absolute;
            inset:auto -10% 48px -10%;
            height:64px;
            background: rgba(255,255,255,.11);
            border-radius:50%;
        }

        .hero::after{
            content:"";
            position:absolute;
            inset:auto -10% 14px -10%;
            height:96px;
            background: rgba(130,214,255,.20);
            border-radius:50%;
        }

        .hero-flex{
            position:relative;
            z-index:2;
            display:flex;
            align-items:center;
            gap:28px;
        }

        .hero-logo{
            flex:0 0 124px;
            width:124px;
            height:124px;
            border-radius:24px;
            background:#fff;
            display:flex;
            align-items:center;
            justify-content:center;
            box-shadow: 0 16px 36px rgba(19,43,93,.28);
            border:1px solid rgba(255,255,255,.85);
            overflow:hidden;
        }

        .hero-logo-fallback{
            font-weight:800;
            font-size:30px;
            color:var(--blue2);
            letter-spacing:.04em;
        }

        .hero-copy{color:#fff;}
        .hero-kicker{
            font-size:13px;
            letter-spacing:.24em;
            text-transform:uppercase;
            opacity:.92;
            font-weight:700;
            margin-bottom:10px;
        }

        .hero-title{
            font-size:58px;
            line-height:1.04;
            font-weight:800;
            margin:0;
            letter-spacing:-0.03em;
            text-shadow:0 8px 22px rgba(0,0,0,.08);
            max-width: 880px;
        }

        .hero-subtitle{
            margin-top:16px;
            font-size:19px;
            line-height:1.7;
            color:rgba(255,255,255,.93);
            max-width:850px;
        }

        .hero-badge{
            display:inline-flex;
            align-items:center;
            justify-content:center;
            margin-top:22px;
            padding:14px 30px;
            border-radius:999px;
            color:#fff;
            font-size:22px;
            font-weight:700;
            background: rgba(255,255,255,.10);
            border:1px solid rgba(255,255,255,.18);
            backdrop-filter: blur(8px);
            box-shadow: 0 10px 30px rgba(10,40,120,.18);
        }

        .login-card-wrap{
            position:relative;
            margin-top:-74px;
            padding:0 36px 36px;
            z-index:3;
        }

        .login-card{
            max-width:960px;
            margin:0 auto;
            background: linear-gradient(180deg, rgba(255,255,255,.97) 0%, rgba(245,248,255,.96) 100%);
            border:1px solid rgba(212,223,239,.95);
            border-radius:28px;
            box-shadow: 0 25px 60px rgba(31,58,120,.18);
            overflow:hidden;
            backdrop-filter: blur(14px);
        }

        .login-card-inner{
            padding:24px 34px 32px;
        }

        div[data-baseweb="tab-list"]{
            gap:22px;
            justify-content:center;
            border-bottom:1px solid #e4eaf5;
            padding:18px 10px 0;
        }

        button[data-baseweb="tab"]{
            background:transparent !important;
            padding:12px 4px 14px !important;
            border-radius:0 !important;
            font-size:17px !important;
            font-weight:600 !important;
            color:#7080a0 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"]{
            color:#1d4ed8 !important;
            border-bottom:3px solid #2e6ee8 !important;
        }

        .section-title{
            font-size:21px;
            font-weight:800;
            color:#202a40;
            margin:10px 0 14px;
        }

        .section-note{
            font-size:14px;
            color:#7a879f;
            margin-bottom:18px;
        }

        div[data-testid="stTextInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stTextArea"] label,
        div.stCheckbox label{
            font-size:14px !important;
            font-weight:600 !important;
            color:#33415c !important;
        }

        div[data-testid="stTextInput"] input{
            min-height:52px !important;
            border-radius:14px !important;
            border:1.5px solid #d8e1ef !important;
            background:#f8fbff !important;
            box-shadow:none !important;
            padding:0 16px !important;
            color:#1f2937 !important;
        }

        div[data-testid="stTextInput"] input:focus{
            border:1.5px solid #3b82f6 !important;
            background:#ffffff !important;
        }

        div.stButton > button{
            width:100%;
            min-height:50px;
            border:none;
            border-radius:14px;
            background: linear-gradient(135deg, #4f8cff 0%, #2b6ee9 60%, #1d5ddb 100%);
            color:#fff;
            font-size:18px;
            font-weight:700;
            box-shadow: 0 16px 30px rgba(43,110,233,.26);
            transition: all .22s ease;
        }

        div.stButton > button:hover{
            transform: translateY(-1px);
            box-shadow: 0 20px 36px rgba(43,110,233,.30);
        }

        .secondary-btn button{
            background: linear-gradient(135deg, #66b9ff 0%, #4a95ff 50%, #5bc1ff 100%) !important;
        }

        .form-sep{
            display:flex;
            align-items:center;
            gap:12px;
            margin:18px 0 14px;
            color:#93a1ba;
            font-size:12px;
            text-transform:uppercase;
            letter-spacing:.18em;
        }

        .form-sep::before,
        .form-sep::after{
            content:"";
            flex:1;
            height:1px;
            background:#e1e8f3;
        }

        .help-note{
            margin-top:18px;
            text-align:center;
            font-size:13px;
            color:#7a879f;
        }

        .dash-hero{
            padding:26px 30px;
            border-radius:26px;
            background: linear-gradient(135deg, #4f8cff 0%, #2563eb 55%, #1e4fc4 100%);
            color:#fff;
            box-shadow: var(--shadow);
            position:relative;
            overflow:hidden;
            margin-bottom:18px;
        }

        .dash-hero::after{
            content:"";
            position:absolute;
            right:-80px;
            bottom:-90px;
            width:260px;
            height:260px;
            border-radius:50%;
            background: rgba(255,255,255,.10);
        }

        .dash-title{
            font-size:34px;
            font-weight:800;
            margin-bottom:8px;
        }

        .dash-sub{
            font-size:16px;
            color:rgba(255,255,255,.92);
            max-width:760px;
        }

        .metric-card{
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border:1px solid #dde6f2;
            border-radius:22px;
            padding:22px;
            box-shadow: 0 16px 34px rgba(31,58,120,.08);
        }

        .metric-label{
            font-size:14px;
            color:#74829b;
            font-weight:600;
            margin-bottom:10px;
        }

        .metric-value{
            font-size:34px;
            font-weight:800;
            color:#1d4ed8;
            line-height:1;
        }

        .metric-foot{
            font-size:13px;
            color:#7b889f;
            margin-top:8px;
        }

        .content-card{
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            border:1px solid #dde6f2;
            border-radius:24px;
            padding:24px;
            box-shadow: 0 16px 34px rgba(31,58,120,.08);
            margin-top:16px;
        }

        .card-title{
            font-size:22px;
            font-weight:800;
            color:#1f2b44;
            margin-bottom:8px;
        }

        .card-sub{
            font-size:14px;
            color:#76839c;
            margin-bottom:16px;
        }

        .chip-row{
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            margin-top:10px;
        }

        .chip{
            display:inline-flex;
            align-items:center;
            padding:10px 14px;
            border-radius:999px;
            background:#eef5ff;
            border:1px solid #dbe7fb;
            color:#2e5ec9;
            font-size:13px;
            font-weight:700;
        }

        .feature-list{
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:14px;
        }

        .feature-item{
            padding:16px 18px;
            border-radius:18px;
            background:#f7fbff;
            border:1px solid #e0e9f6;
        }

        .feature-item strong{
            display:block;
            color:#22314d;
            font-size:15px;
            margin-bottom:6px;
        }

        .feature-item span{
            color:#6f7d97;
            font-size:13px;
            line-height:1.6;
        }

        .status-good{
            background:#ebfaf0;
            color:#198754;
            border:1px solid #c7efda;
            border-radius:14px;
            padding:12px 14px;
            font-size:14px;
            font-weight:700;
        }

        @media (max-width: 900px){
            .hero{
                padding:26px 22px 96px;
            }
            .hero-flex{
                flex-direction:column;
                align-items:flex-start;
            }
            .hero-title{
                font-size:40px;
            }
            .hero-subtitle{
                font-size:16px;
            }
            .hero-badge{
                font-size:18px;
            }
            .login-card-wrap{
                padding:0 14px 20px;
                margin-top:-64px;
            }
            .login-card-inner{
                padding:18px 18px 22px;
            }
            .feature-list{
                grid-template-columns:1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_auth_page():
    st.markdown('<div class="app-shell">', unsafe_allow_html=True)

    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown('<div class="hero-flex">', unsafe_allow_html=True)

    if os.path.exists(LOGO_PATH):
        st.markdown('<div class="hero-logo">', unsafe_allow_html=True)
        st.image(LOGO_PATH, width=100)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hero-logo"><div class="hero-logo-fallback">BKAI</div></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-copy">
            <div class="hero-kicker">BKAI CRACK ANALYSIS PORTAL</div>
            <h1 class="hero-title">AI-Based Concrete Crack Detection Platform</h1>
            <div class="hero-subtitle">
                Secure access to image-based crack detection, segmentation, reporting, and structural crack classification in one integrated interface.
            </div>
            <div class="hero-badge">Welcome to the system</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="login-card-wrap"><div class="login-card"><div class="login-card-inner">', unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(['Login', 'Register'])

    with tab_login:
        st.markdown('<div class="section-title">Sign in</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Access your BKAI workspace with your registered credentials.</div>', unsafe_allow_html=True)

        username = st.text_input('Username', key='login_user', placeholder='Enter username')
        password = st.text_input('Password', key='login_pass', type='password', placeholder='Enter password')
        st.checkbox('Stay logged in', key='stay_logged_in')

        login_btn = st.button('Log in with Credentials', key='login_button')
        st.markdown('<div class="form-sep">or</div>', unsafe_allow_html=True)
        badge_col = st.container()
        with badge_col:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            badge_btn = st.button('Log in with Badge', key='badge_button')
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="help-note">Forgot your registered account or password? Contact the BKAI analysis portal team.</div>', unsafe_allow_html=True)

        if login_btn:
            if not username or not password:
                st.error('Please enter both username and password.')
            elif username in users and users.get(username) == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error('Invalid username or password.')

        if badge_btn:
            st.info('Badge login is a demo placeholder in this version.')

    with tab_register:
        st.markdown('<div class="section-title">Create account</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Register a new BKAI account to access the crack analysis workspace.</div>', unsafe_allow_html=True)

        new_user = st.text_input('Username', key='reg_user', placeholder='Choose a username')
        new_email = st.text_input('Email', key='reg_email', placeholder='Enter your email')
        new_pass = st.text_input('Password', key='reg_pass', type='password', placeholder='Create a password')
        new_pass2 = st.text_input('Confirm Password', key='reg_pass2', type='password', placeholder='Re-enter password')

        register_btn = st.button('Create Account', key='register_button')

        if register_btn:
            if not new_user or not new_email or not new_pass or not new_pass2:
                st.warning('Please complete all required fields.')
            elif '@' not in new_email or '.' not in new_email:
                st.error('Please enter a valid email address.')
            elif new_user in users:
                st.error('This username already exists.')
            elif new_pass != new_pass2:
                st.error('Passwords do not match.')
            elif len(new_pass) < 6:
                st.error('Password must be at least 6 characters long.')
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success('Account created successfully. You can now log in.')

    st.markdown('</div></div></div>', unsafe_allow_html=True)


def dashboard_page():
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=110)
        st.markdown(f'### {st.session_state.username}')
        st.caption('BKAI workspace active')
        st.radio(
            'Navigation',
            options=['Dashboard', 'Image Analyzer', 'Reports', 'User Profile'],
            key='nav_radio'
        )
        st.markdown('---')
        st.slider('Confidence threshold', 0.0, 1.0, 0.35, 0.05)
        st.selectbox('Project mode', ['Concrete Inspection', 'Research Demo', 'Site Monitoring'])
        if st.button('Log out'):
            st.session_state.authenticated = False
            st.session_state.username = ''
            st.rerun()

    st.markdown(
        """
        <div class="dash-hero">
            <div class="dash-title">BKAI Smart Concrete Inspection Workspace</div>
            <div class="dash-sub">
                Centralize crack detection, image analysis, reporting, and research support in one modern interface designed for engineers, researchers, and project teams.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Images processed</div>
                <div class="metric-value">1,248</div>
                <div class="metric-foot">Across research and inspection cases</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Average confidence</div>
                <div class="metric-value">94.7%</div>
                <div class="metric-foot">Based on validated crack predictions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Generated reports</div>
                <div class="metric-value">326</div>
                <div class="metric-foot">PDF reports exported successfully</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Image Analyzer</div>
                <div class="card-sub">Upload concrete images to perform crack detection, segmentation, and report generation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader('Upload concrete images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if uploaded:
            st.markdown('<div class="status-good">Images uploaded successfully. Ready for AI analysis.</div>', unsafe_allow_html=True)
            st.write(f'{len(uploaded)} file(s) selected.')
            st.button('Run AI Analysis')

        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Feature Modules</div>
                <div class="card-sub">Main functional areas available in this full web layout.</div>
                <div class="feature-list">
                    <div class="feature-item"><strong>Crack Detection</strong><span>Detect visible cracks from uploaded concrete images with AI-assisted prediction.</span></div>
                    <div class="feature-item"><strong>Instance Segmentation</strong><span>Visualize crack regions with overlay masks for inspection and reporting workflows.</span></div>
                    <div class="feature-item"><strong>PDF Reporting</strong><span>Create structured analysis summaries for academic, field, or management use.</span></div>
                    <div class="feature-item"><strong>Stage 2 Knowledge</strong><span>Map observed cracks to structural components and common cause categories.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Current Session</div>
                <div class="card-sub">Overview of the active user environment.</div>
                <div class="chip-row">
                    <div class="chip">User: Active</div>
                    <div class="chip">Portal: Online</div>
                    <div class="chip">Model: Ready</div>
                    <div class="chip">Reports: Enabled</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Recommended Workflow</div>
                <div class="card-sub">Suggested step-by-step usage inside the BKAI system.</div>
                <div class="feature-list" style="grid-template-columns:1fr;">
                    <div class="feature-item"><strong>Step 1 — Upload images</strong><span>Select one or multiple concrete images from your local device.</span></div>
                    <div class="feature-item"><strong>Step 2 — Run analysis</strong><span>Start the AI pipeline to detect and segment visible cracks.</span></div>
                    <div class="feature-item"><strong>Step 3 — Review results</strong><span>Inspect metrics, severity, overlays, and structural interpretation.</span></div>
                    <div class="feature-item"><strong>Step 4 — Export outputs</strong><span>Download summary reports and use them in research or inspection records.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


inject_styles()

if st.session_state.authenticated:
    dashboard_page()
else:
    show_auth_page()
