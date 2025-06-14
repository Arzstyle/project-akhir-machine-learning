import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings("ignore")

# Konten panduan penggunaan untuk setiap input field
field_tooltips = {
    "gender": "Pilih jenis kelamin pelanggan: Male atau Female.",
    "senior_citizen": "Apakah pelanggan berusia 65 tahun atau lebih? Pilih Ya (1) atau Tidak (0).",
    "partner": "Apakah pelanggan memiliki pasangan? Pilih Ya atau Tidak.",
    "dependents": "Apakah pelanggan memiliki tanggungan? Pilih Ya atau Tidak.",
    "phone_service": "Apakah pelanggan menggunakan layanan telepon? Pilih Ya atau Tidak.",
    "multiple_lines": "Apakah pelanggan memiliki multiple lines? Pilih Ya, Tidak, atau No phone service jika tidak menggunakan layanan telepon.",
    "internet_service": "Jenis layanan internet yang digunakan pelanggan: DSL, Fiber optic, atau No jika tidak menggunakan layanan internet.",
    "online_security": "Apakah pelanggan menggunakan layanan keamanan online? Pilih Ya, Tidak, atau No internet service jika tidak menggunakan layanan internet.",
    "online_backup": "Apakah pelanggan menggunakan layanan backup online? Pilih Ya, Tidak, atau No internet service jika tidak menggunakan layanan internet.",
    "device_protection": "Apakah pelanggan menggunakan layanan proteksi perangkat? Pilih Ya, Tidak, atau No internet service jika tidak menggunakan layanan internet.",
    "tech_support": "Apakah pelanggan menggunakan layanan dukungan teknis? Pilih Ya, Tidak, atau No internet service jika tidak menggunakan layanan internet.",
    "streaming_tv": "Apakah pelanggan menggunakan layanan streaming TV? Pilih Ya, Tidak, atau No internet service jika tidak menggunakan layanan internet.",
    "streaming_movies": "Apakah pelanggan menggunakan layanan streaming film? Pilih Ya, Tidak, atau No internet service jika tidak menggunakan layanan internet.",
    "contract": "Jenis kontrak langganan pelanggan: Month-to-month, One year, atau Two year.",
    "paperless_billing": "Apakah pelanggan menggunakan tagihan elektronik (paperless billing)? Pilih Ya atau Tidak.",
    "payment_method": "Metode pembayaran yang digunakan pelanggan: Electronic check, Mailed check, Bank transfer (automatic), atau Credit card (automatic).",
    "tenure": "Lama berlangganan pelanggan dalam bulan (nilai antara 0 hingga 100).",
    "monthly_charges": "Biaya bulanan pelanggan dalam USD (nilai antara 0.0 hingga 200.0).",
    "total_charges": "Total biaya yang telah dibayar pelanggan dalam USD (nilai antara 0.0 hingga 10000.0)."
}

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .churn-high {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .churn-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_and_preprocess_data():
    """Memuat dataset dan melakukan pra-pemrosesan awal."""
    try:
        df = pd.read_csv("demo-streamlit/data-customer-churn.csv")
        
        # Menangani nilai kosong di TotalCharges
        df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float)
        df.dropna(subset=["TotalCharges"], inplace=True)
        
        # Menangani duplikasi customerID
        if df.duplicated(subset=["customerID"]).any():
            df.drop_duplicates(subset=["customerID"], keep="first", inplace=True)
        
        # Hapus kolom customerID
        df.drop("customerID", axis=1, inplace=True)
        
        return df
    except FileNotFoundError:
        st.error("File 'data-customer-churn.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan aplikasi.")
        return None

# Fungsi untuk melatih model
@st.cache_resource
def train_model(df):
    """Melatih model Logistic Regression dengan pipeline."""
    # Memisahkan fitur dan target
    X = df.drop("Churn", axis=1)
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Identifikasi kolom numerik dan kategorikal
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    
    # Buat preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])
    
    # Buat pipeline model
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="liblinear", random_state=42, class_weight="balanced"))
    ])
    
    # Latih model
    model_pipeline.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    return model_pipeline, metrics, X_test, y_test, y_pred, y_pred_proba, numerical_features, categorical_features

# Fungsi untuk mendapatkan feature importance
def get_feature_importance(model_pipeline, numerical_features, categorical_features):
    """Mendapatkan feature importance dari model."""
    # Dapatkan koefisien
    coefficients = model_pipeline.named_steps["classifier"].coef_[0]
    
    # Dapatkan nama fitur setelah encoding
    ohe = model_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    try:
        encoded_categorical_features = ohe.get_feature_names_out(categorical_features)
    except AttributeError:
        encoded_categorical_features = ohe.get_feature_names(categorical_features)
    
    feature_names = list(numerical_features) + list(encoded_categorical_features)
    
    # Buat DataFrame
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Abs_Coefficient": np.abs(coefficients)
    })
    
    return feature_importance_df.sort_values(by="Abs_Coefficient", ascending=False)

# Fungsi untuk prediksi individual
def predict_churn(model_pipeline, input_data):
    """Melakukan prediksi churn untuk input data."""
    prediction = model_pipeline.predict(input_data)[0]
    probability = model_pipeline.predict_proba(input_data)[0]
    
    return prediction, probability

# Fungsi untuk menjelaskan prediksi
def explain_prediction(input_data, feature_importance_df, numerical_features, categorical_features):
    """Memberikan penjelasan tentang prediksi."""
    explanations = []
    
    # Analisis fitur numerik
    for feature in numerical_features:
        value = input_data[feature].iloc[0]
        coef_row = feature_importance_df[feature_importance_df["Feature"] == feature]
        if not coef_row.empty:
            coef = coef_row["Coefficient"].iloc[0]
            if abs(coef) > 0.1:  # Hanya fitur dengan koefisien signifikan
                if coef > 0:
                    explanations.append(f"• {feature}: {value} (meningkatkan risiko churn)")
                else:
                    explanations.append(f"• {feature}: {value} (menurunkan risiko churn)")
    
    # Analisis fitur kategorikal
    for feature in categorical_features:
        value = input_data[feature].iloc[0]
        # Cari fitur yang sesuai setelah one-hot encoding
        matching_features = feature_importance_df[feature_importance_df["Feature"].str.contains(f"{feature}_{value}", na=False)]
        if not matching_features.empty:
            coef = matching_features["Coefficient"].iloc[0]
            if abs(coef) > 0.1:
                if coef > 0:
                    explanations.append(f"• {feature}: {value} (meningkatkan risiko churn)")
                else:
                    explanations.append(f"• {feature}: {value} (menurunkan risiko churn)")
    
    return explanations

# Sidebar untuk navigasi
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Dashboard Statistik Data", "Prediksi Churn", "About"]
)

# Muat data
df = load_and_preprocess_data()

if df is not None:
    # Latih model
    model_pipeline, metrics, X_test, y_test, y_pred, y_pred_proba, numerical_features, categorical_features = train_model(df)
    feature_importance_df = get_feature_importance(model_pipeline, numerical_features, categorical_features)

    # HALAMAN 1: Dashboard Statistik Data
    if page == "Dashboard Statistik Data":
        st.markdown("<div class=\"main-header\">📊 Dashboard Statistik Data</div>", unsafe_allow_html=True)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
            st.metric("Total Pelanggan", f"{len(df):,}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            churn_count = len(df[df["Churn"] == "Yes"])
            st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
            st.metric("Pelanggan Churn", f"{churn_count:,}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            churn_rate = (churn_count / len(df)) * 100
            st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
            st.metric("Tingkat Churn", f"{churn_rate:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            avg_monthly_charges = df["MonthlyCharges"].mean()
            st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
            st.metric("Rata-rata Biaya Bulanan", f"${avg_monthly_charges:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Distribusi Churn
        st.markdown("<div class=\"sub-header\">Distribusi Churn Pelanggan</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            churn_counts = df["Churn"].value_counts()
            fig_pie = px.pie(
                values=churn_counts.values,
                names=["Tidak Churn", "Churn"],
                title="Distribusi Churn",
                color_discrete_sequence=["#2ecc71", "#e74c3c"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=["Tidak Churn", "Churn"],
                y=churn_counts.values,
                title="Jumlah Pelanggan per Status Churn",
                color=["Tidak Churn", "Churn"],
                color_discrete_sequence=["#2ecc71", "#e74c3c"]
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>💡 Insight:</strong> Dari total pelanggan, sekitar {:.1f}% melakukan churn. 
        Ini menunjukkan adanya tantangan dalam retensi pelanggan yang perlu ditangani.
        </div>
        """.format(churn_rate), unsafe_allow_html=True)
        
        # Analisis fitur numerik
        st.markdown("<div class=\"sub-header\">Analisis Fitur Numerik</div>", unsafe_allow_html=True)
        
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(numeric_cols):
                    feature = numeric_cols[i + j]
                    with col:
                        fig = px.histogram(
                            df, 
                            x=feature, 
                            color="Churn",
                            title=f"Distribusi {feature}",
                            color_discrete_sequence=["#2ecc71", "#e74c3c"],
                            opacity=0.7
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis fitur kategorikal
        st.markdown("<div class=\"sub-header\">Analisis Fitur Kategorikal</div>", unsafe_allow_html=True)
        
        categorical_cols = ["Contract", "PaymentMethod", "InternetService", "gender"]
        
        for i in range(0, len(categorical_cols), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(categorical_cols):
                    feature = categorical_cols[i + j]
                    with col:
                        # Hitung proporsi churn per kategori
                        churn_by_category = df.groupby(feature)["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
                        churn_by_category.columns = [feature, "Churn_Rate"]
                        
                        fig = px.bar(
                            churn_by_category,
                            x=feature,
                            y="Churn_Rate",
                            title=f"Tingkat Churn berdasarkan {feature}",
                            color="Churn_Rate",
                            color_continuous_scale="Reds"
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

    # HALAMAN 2: Prediksi Churn
    elif page == "Prediksi Churn":
        st.markdown("<div class=\"main-header\">🔮 Prediksi Churn Pelanggan</div>", unsafe_allow_html=True)
        
        # Tambahkan penjelasan singkat tentang fungsi fitur prediksi churn
        st.markdown("""
        <div class="info-box">
        Gunakan fitur ini untuk memprediksi kemungkinan pelanggan akan berhenti berlangganan. 
        Hasil prediksi didasarkan pada model machine learning dan dapat membantu mengambil keputusan strategis.
        </div>
        """, unsafe_allow_html=True)

        # Form input
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Informasi Demografis")
                gender = st.selectbox("Gender", ["Male", "Female"], help=field_tooltips["gender"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak", help=field_tooltips["senior_citizen"])
                partner = st.selectbox("Partner", ["Yes", "No"], format_func=lambda x: "Ya" if x == "Yes" else "Tidak", help=field_tooltips["partner"])
                dependents = st.selectbox("Dependents", ["Yes", "No"], format_func=lambda x: "Ya" if x == "Yes" else "Tidak", help=field_tooltips["dependents"])
                
                st.subheader("Informasi Layanan")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], format_func=lambda x: "Ya" if x == "Yes" else "Tidak", help=field_tooltips["phone_service"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], help=field_tooltips["multiple_lines"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], help=field_tooltips["internet_service"])
            
            with col2:
                st.subheader("Layanan Tambahan")
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], help=field_tooltips["online_security"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], help=field_tooltips["online_backup"])
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], help=field_tooltips["device_protection"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], help=field_tooltips["tech_support"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], help=field_tooltips["streaming_tv"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], help=field_tooltips["streaming_movies"])
            
            with col3:
                st.subheader("Informasi Kontrak & Pembayaran")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help=field_tooltips["contract"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], format_func=lambda x: "Ya" if x == "Yes" else "Tidak", help=field_tooltips["paperless_billing"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ], help=field_tooltips["payment_method"])
                
                st.subheader("Informasi Finansial")
                tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12, help=field_tooltips["tenure"])
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01, help=field_tooltips["monthly_charges"])
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=0.01, help=field_tooltips["total_charges"])
            
            submitted = st.form_submit_button("🔍 Prediksi Churn", use_container_width=True)
            
            if submitted:
                # Buat DataFrame input
                input_data = pd.DataFrame({
                    "gender": [gender],
                    "SeniorCitizen": [senior_citizen],
                    "Partner": [partner],
                    "Dependents": [dependents],
                    "tenure": [tenure],
                    "PhoneService": [phone_service],
                    "MultipleLines": [multiple_lines],
                    "InternetService": [internet_service],
                    "OnlineSecurity": [online_security],
                    "OnlineBackup": [online_backup],
                    "DeviceProtection": [device_protection],
                    "TechSupport": [tech_support],
                    "StreamingTV": [streaming_tv],
                    "StreamingMovies": [streaming_movies],
                    "Contract": [contract],
                    "PaperlessBilling": [paperless_billing],
                    "PaymentMethod": [payment_method],
                    "MonthlyCharges": [monthly_charges],
                    "TotalCharges": [total_charges]
                })
                
                # Prediksi
                prediction, probability = predict_churn(model_pipeline, input_data)
                churn_prob = probability[1] * 100
                
                # Tampilkan hasil
                st.markdown("---")
                st.markdown("<div class=\"sub-header\">🎯 Hasil Prediksi</div>", unsafe_allow_html=True)
                
                # Tambahkan penjelasan untuk output tabel
                st.markdown("""
                <div class="info-box">
                Tabel di bawah menampilkan hasil prediksi kemungkinan churn berdasarkan input yang Anda berikan. 
                Kolom probabilitas menunjukkan tingkat kemungkinan pelanggan akan churn, 
                sedangkan label hasil memberikan interpretasi langsung (Churn/Tidak Churn).
                </div>
                """, unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-result churn-high">
                    ⚠️ RISIKO TINGGI - Pelanggan kemungkinan akan CHURN<br>
                    Probabilitas Churn: {churn_prob:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-result churn-low">
                    ✅ RISIKO RENDAH - Pelanggan kemungkinan TIDAK akan churn<br>
                    Probabilitas Churn: {churn_prob:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                # Penjelasan prediksi
                st.markdown("<div class=\"sub-header\">💡 Penjelasan Prediksi</div>", unsafe_allow_html=True)
                
                explanations = explain_prediction(input_data, feature_importance_df, numerical_features, categorical_features)
                
                if explanations:
                    st.markdown("**Faktor-faktor yang mempengaruhi prediksi:**")
                    for explanation in explanations[:5]:  # Tampilkan 5 faktor teratas
                        st.markdown(explanation)
                else:
                    st.markdown("Tidak ada faktor signifikan yang teridentifikasi.")
                
                # Rekomendasi
                st.markdown("<div class=\"sub-header\">📋 Rekomendasi Tindakan</div>", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown("""
                    <div class="info-box">
                    <strong>🚨 Tindakan Segera Diperlukan:</strong><br>
                    • Hubungi pelanggan untuk memahami kekhawatiran mereka<br>
                    • Tawarkan diskon atau promosi khusus<br>
                    • Pertimbangkan upgrade layanan atau kontrak jangka panjang<br>
                    • Tingkatkan kualitas customer service<br>
                    • Evaluasi kepuasan pelanggan secara berkala
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                    <strong>✅ Pelanggan Stabil:</strong><br>
                    • Pertahankan kualitas layanan yang baik<br>
                    • Tawarkan layanan tambahan yang relevan<br>
                    • Berikan program loyalitas<br>
                    • Minta feedback untuk perbaikan berkelanjutan
                    </div>
                    """, unsafe_allow_html=True)

    # HALAMAN 3: About
    elif page == "About":
        st.markdown("<div class=\"main-header\">ℹ️ Tentang Aplikasi</div>", unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 Pemodelan Prediksi Churn dengan Regresi Logistik
        
        ### Studi Kasus pada Pelanggan Berbasis Langganan
        
        Aplikasi ini dikembangkan untuk membantu perusahaan telekomunikasi dalam memprediksi dan mencegah churn pelanggan 
        menggunakan teknik machine learning.
        
        ### 🔍 Apa itu Churn?
        
        **Customer Churn** adalah istilah yang digunakan untuk menggambarkan pelanggan yang berhenti menggunakan 
        produk atau layanan perusahaan. Dalam konteks telekomunikasi, churn terjadi ketika pelanggan:
        - Membatalkan langganan
        - Beralih ke kompetitor
        - Tidak memperpanjang kontrak
        
        ### 💼 Mengapa Churn Penting untuk Bisnis?
        
        1. **Biaya Akuisisi Tinggi**: Mendapatkan pelanggan baru 5-25x lebih mahal daripada mempertahankan yang ada
        2. **Revenue Loss**: Kehilangan pelanggan berarti kehilangan pendapatan berkelanjutan
        3. **Competitive Advantage**: Perusahaan dengan churn rate rendah memiliki keunggulan kompetitif
        4. **Customer Lifetime Value**: Pelanggan loyal memberikan nilai jangka panjang yang lebih tinggi
        
        ### 🤖 Model yang Digunakan
        
        **Logistic Regression** dipilih karena:
        - Mudah diinterpretasi dan dijelaskan
        - Memberikan probabilitas prediksi
        - Performa yang baik untuk klasifikasi biner
        - Dapat mengidentifikasi faktor-faktor penting
        
        ### 📊 Sumber Data
        
        Dataset yang digunakan berisi informasi pelanggan telekomunikasi dengan fitur:
        - **Demografis**: Gender, usia, status keluarga
        - **Layanan**: Jenis layanan yang digunakan
        - **Kontrak**: Jenis kontrak dan metode pembayaran
        - **Finansial**: Biaya bulanan dan total
        
        ### 🎯 Tujuan Aplikasi
        
        1. **Identifikasi Risiko**: Mendeteksi pelanggan dengan risiko churn tinggi
        2. **Analisis Faktor**: Memahami faktor-faktor yang mempengaruhi churn
        3. **Tindakan Preventif**: Memberikan rekomendasi untuk mencegah churn
        4. **Monitoring**: Memantau tren dan pola churn
        
        ### 📈 Manfaat Bisnis
        
        - **Proactive Retention**: Tindakan pencegahan sebelum pelanggan churn
        - **Resource Optimization**: Fokus upaya retensi pada pelanggan berisiko tinggi
        - **Revenue Protection**: Melindungi pendapatan dari kehilangan pelanggan
        - **Strategic Planning**: Data-driven decision making untuk strategi bisnis
        """)
        
        # Model Performance
        st.markdown("<div class=\"sub-header\">📊 Performa Model</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics["accuracy"]:.3f}")
            st.metric("Precision", f"{metrics["precision"]:.3f}")
        
        with col2:
            st.metric("Recall", f"{metrics["recall"]:.3f}")
            st.metric("F1-Score", f"{metrics["f1"]:.3f}")
        
        with col3:
            st.metric("ROC AUC", f"{metrics["roc_auc"]:.3f}")
        
        st.markdown("""
        <div class="info-box">
        <strong>📝 Catatan Performa:</strong> Model telah dilatih dan divalidasi menggunakan teknik cross-validation 
        untuk memastikan generalisasi yang baik pada data baru.
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("Tidak dapat memuat data. Pastikan file 'data-customer-churn.csv' tersedia.")
