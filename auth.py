import streamlit as st
import json
import os
import bcrypt

USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        default = {
            "admin": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        }
        with open(USERS_FILE, "w") as f:
            json.dump(default, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def verify_password(plain, hashed):
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def hash_password(plain):
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def show_login_page():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background: #080c14 !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ── Auth wrapper ── */
    .auth-outer {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #080c14;
        padding: 40px 16px;
    }
    .auth-card {
        width: 100%;
        max-width: 400px;
        background: #0d1625;
        border: 1px solid #1a2d4a;
        border-radius: 24px;
        padding: 40px 36px 32px;
        position: relative;
        overflow: hidden;
    }
    .auth-card::before {
        content: '';
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(220,38,38,0.10) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    /* ── Logo ── */
    .auth-logo-ring {
        width: 64px; height: 64px; border-radius: 50%;
        background: rgba(239,68,68,0.10);
        border: 1.5px solid rgba(239,68,68,0.28);
        display: flex; align-items: center; justify-content: center;
        font-size: 30px;
        margin: 0 auto 18px;
    }
    .auth-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem; font-weight: 800;
        color: #f0f4ff; text-align: center; margin-bottom: 4px;
    }
    .auth-subtitle {
        font-size: 0.82rem; color: #4a6285;
        text-align: center; margin-bottom: 10px;
    }
    .auth-badge {
        display: flex; align-items: center; justify-content: center; gap: 6px;
        width: fit-content; margin: 0 auto 24px;
        background: rgba(34,197,94,0.10);
        border: 0.5px solid rgba(34,197,94,0.3);
        border-radius: 20px; padding: 4px 12px;
        font-size: 0.75rem; color: #22c55e; font-weight: 500;
    }

    /* ── Fields ── */
    .auth-label {
        display: block;
        font-size: 0.75rem; font-weight: 500; letter-spacing: 0.06em;
        text-transform: uppercase; color: #4a6285; margin-bottom: 6px;
    }
    div[data-testid="stTextInput"] input {
        background: #060a12 !important;
        border: 1px solid #1a2d4a !important;
        border-radius: 10px !important;
        color: #c9d9f0 !important;
        font-size: 0.9rem !important;
        padding: 10px 14px !important;
        transition: border-color 0.2s !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.12) !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: #2a4a7a !important; }
    div[data-testid="stTextInput"] label {
        font-size: 0.75rem !important; color: #4a6285 !important;
        font-weight: 500 !important; letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }

    /* ── Buttons ── */
    div[data-testid="stButton"] > button {
        background: #1d4ed8 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 0.9rem !important; font-weight: 600 !important;
        padding: 12px 0 !important;
        width: 100% !important;
        transition: background 0.2s !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: #2563eb !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: #060a12 !important;
        border: 1px solid #1a2d4a !important;
        color: #6b8aad !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        border-color: #2a4a7a !important;
    }

    /* ── Tab pills ── */
    div[data-testid="stTabs"] > div:first-child {
        background: #060a12;
        border-radius: 10px;
        padding: 3px;
        border: 1px solid #1a2d4a;
        gap: 2px;
    }
    button[data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        color: #4a6285 !important;
        background: transparent !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #1a2f50 !important;
        color: #60a5fa !important;
    }

    /* ── Divider ── */
    .auth-divider {
        display: flex; align-items: center; gap: 12px;
        margin: 16px 0; color: #2a4a7a; font-size: 0.8rem;
    }
    .auth-divider-line { flex: 1; height: 1px; background: #1a2d4a; }

    /* ── Footer ── */
    .auth-footer {
        text-align: center; margin-top: 22px;
        font-size: 0.75rem; color: #2a4a7a; line-height: 1.8;
    }
    .auth-footer a { color: #3b82f6; text-decoration: none; }

    /* ── Alert overrides ── */
    div[data-testid="stAlert"] {
        background: rgba(239,68,68,0.07) !important;
        border: 1px solid rgba(239,68,68,0.2) !important;
        border-radius: 10px !important;
        color: #fca5a5 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Centered card layout ──
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        st.markdown('<div class="auth-logo-ring">🫀</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">CVD Intelligence Hub</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Cardiovascular Risk Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-badge">🛡 Secure · Medical Grade</div>', unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["  Sign In  ", "  Register  "])

        users = load_users()

        # ── LOGIN TAB ──
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

            st.markdown('<div style="text-align:right;margin-top:-8px;margin-bottom:12px;"><span style="font-size:0.78rem;color:#3b82f6;cursor:pointer;">Forgot password?</span></div>', unsafe_allow_html=True)

            if st.button("Sign In →", use_container_width=True, key="btn_login"):
                if not username or not password:
                    st.error("Please fill in both fields.")
                elif username not in users:
                    st.error("❌ Username not found.")
                elif not verify_password(password, users[username]):
                    st.error("❌ Incorrect password.")
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.rerun()

            st.markdown("""
            <div class="auth-divider">
              <div class="auth-divider-line"></div>
              <span>or</span>
              <div class="auth-divider-line"></div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔵  Continue with Google", use_container_width=True, key="btn_google", type="secondary"):
                st.info("Google sign-in not configured yet. Use username/password.")

        # ── REGISTER TAB ──
        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)
            new_name = st.text_input("Full Name", placeholder="e.g. Dr. Sharma", key="reg_name")
            new_user = st.text_input("Username", placeholder="Choose a username", key="reg_user")
            new_pass = st.text_input("Password", type="password", placeholder="Min 6 characters", key="reg_pass")
            confirm  = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="reg_confirm")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True, key="btn_register"):
                if not new_name or not new_user or not new_pass or not confirm:
                    st.error("Please fill in all fields.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pass != confirm:
                    st.error("❌ Passwords do not match.")
                elif new_user in users:
                    st.error("❌ Username already taken. Choose another.")
                else:
                    users[new_user] = hash_password(new_pass)
                    save_users(users)
                    st.success(f"✅ Account created! Go to Sign In tab, {new_name}.")

        st.markdown("""
        <div class="auth-footer">
          ⚕️ For authorised medical personnel only<br>
          <a href="#">Privacy Policy</a> · <a href="#">Terms of Use</a>
        </div>
        """, unsafe_allow_html=True)


def require_auth():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        show_login_page()
        st.stop()