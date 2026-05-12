import streamlit as st
import json
import os
import bcrypt

USERS_FILE = "users.json"


# =========================
# USER STORAGE
# =========================
def load_users():
    if not os.path.exists(USERS_FILE):
        default = {
            "admin": bcrypt.hashpw(
                "admin123".encode(),
                bcrypt.gensalt()
            ).decode()
        }

        with open(USERS_FILE, "w") as f:
            json.dump(default, f)

    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def verify_password(plain, hashed):
    return bcrypt.checkpw(
        plain.encode(),
        hashed.encode()
    )


def hash_password(plain):
    return bcrypt.hashpw(
        plain.encode(),
        bcrypt.gensalt()
    ).decode()


# =========================
# LOGIN PAGE
# =========================
def show_login_page():

    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background: #f4f8fc !important;
        color: #111827 !important;
    }

    .stApp {
        background: linear-gradient(to bottom right, #eef5ff, #f8fbff);
    }

    /* Hide Streamlit Menu */
    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 100% !important;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Card */
    .auth-card {
        background: white;
        padding: 45px 40px;
        border-radius: 24px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.08);
        border: 1px solid #dbeafe;
    }

    /* Logo */
    .auth-logo {
        width: 75px;
        height: 75px;
        border-radius: 50%;
        background: #eff6ff;
        border: 2px solid #bfdbfe;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
        font-size: 34px;
    }

    /* Titles */
    .auth-title {
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-top: 18px;
        margin-bottom: 6px;
    }

    .auth-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    /* Badge */
    .auth-badge {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #15803d;
        width: fit-content;
        margin: auto;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 30px;
    }

    /* Inputs */
    div[data-testid="stTextInput"] input {
        background: white !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #111827 !important;
        padding: 12px 14px !important;
        font-size: 0.95rem !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    }

    div[data-testid="stTextInput"] label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.04em;
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: 0.2s;
    }

    div[data-testid="stButton"] > button:hover {
        background: #1d4ed8 !important;
    }

    /* Tabs */
    div[data-testid="stTabs"] > div:first-child {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 4px;
    }

    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        color: #2563eb !important;
    }

    /* Divider */
    .divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 18px 0;
        color: #94a3b8;
        font-size: 0.8rem;
    }

    .divider-line {
        flex: 1;
        height: 1px;
        background: #e2e8f0;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 28px;
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.8;
    }

    .footer a {
        color: #2563eb;
        text-decoration: none;
    }

    </style>
    """, unsafe_allow_html=True)

    users = load_users()

    # CENTER CARD
    left, center, right = st.columns([1, 1.2, 1])

    with center:

        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="auth-logo">🫀</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="auth-title">CVD Intelligence Hub</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="auth-subtitle">Cardiovascular Risk Prediction Platform</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="auth-badge">🛡 Secure Medical Access</div>',
            unsafe_allow_html=True
        )

        # TABS
        tab1, tab2 = st.tabs(["Sign In", "Register"])

        # ================= LOGIN =================
        with tab1:

            username = st.text_input(
                "Username",
                placeholder="Enter username",
                key="login_user"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
                key="login_pass"
            )

            st.markdown(
                """
                <div style="text-align:right;margin-top:-8px;margin-bottom:12px;">
                <span style="font-size:0.8rem;color:#2563eb;">
                Forgot password?
                </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Sign In",
                use_container_width=True,
                key="login_btn"
            ):

                if not username or not password:
                    st.error("Please fill all fields.")

                elif username not in users:
                    st.error("Username not found.")

                elif not verify_password(password, users[username]):
                    st.error("Incorrect password.")

                else:
                    st.success("Login successful!")

                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username

                    st.rerun()

            st.markdown("""
            <div class="divider">
                <div class="divider-line"></div>
                or
                <div class="divider-line"></div>
            </div>
            """, unsafe_allow_html=True)

            st.button(
                "Continue with Google",
                use_container_width=True,
                key="google"
            )

        # ================= REGISTER =================
        with tab2:

            full_name = st.text_input(
                "Full Name",
                placeholder="Enter full name",
                key="reg_name"
            )

            new_user = st.text_input(
                "Username",
                placeholder="Choose username",
                key="reg_user"
            )

            new_pass = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 6 characters",
                key="reg_pass"
            )

            confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter password",
                key="reg_confirm"
            )

            if st.button(
                "Create Account",
                use_container_width=True,
                key="register_btn"
            ):

                if not full_name or not new_user or not new_pass or not confirm:
                    st.error("Please fill all fields.")

                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")

                elif new_pass != confirm:
                    st.error("Passwords do not match.")

                elif new_user in users:
                    st.error("Username already exists.")

                else:
                    users[new_user] = hash_password(new_pass)
                    save_users(users)

                    st.success(
                        f"Account created successfully, {full_name}!"
                    )

        st.markdown("""
        <div class="footer">
        ⚕️ For authorised medical personnel only<br>
        <a href="#">Privacy Policy</a> ·
        <a href="#">Terms of Use</a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# AUTH CHECK
# =========================
def require_auth():

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        show_login_page()
        st.stop()