import os
import io
import base64
import numpy as np
from PIL import Image as PILImage
from flask import Flask, render_template_string, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.secret_key = 'super_secret_key_anda'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('preprocessed', exist_ok=True)

MODEL_PATH = 'mobilenetv2_endtoend.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model MobileNetV2 berhasil dimuat!")
else:
    model = None
    print("Model h5 tidak ditemukan!")

CLASS_NAMES = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

CLASS_META = {
    'COVID': {
        'color': '#EF4444', 'bg': '#FEF2F2',
        'desc': 'Terdeteksi indikasi infeksi spesifik virus COVID-19 pada jaringan paru-paru.',
        'badge': 'KRITIS'
    },
    'Lung Opacity': {
        'color': '#F59E0B', 'bg': '#FFFBEB',
        'desc': 'Terdeteksi area opasitas atau kekeruhan non-spesifik pada paru-paru.',
        'badge': 'WASPADA'
    },
    'Normal': {
        'color': '#10B981', 'bg': '#ECFDF5',
        'desc': 'Kondisi paru-paru terlihat normal tanpa tanda infeksi atau opasitas akut.',
        'badge': 'NORMAL'
    },
    'Viral Pneumonia': {
        'color': '#3B82F6', 'bg': '#EFF6FF',
        'desc': 'Terdeteksi pola infeksi paru-paru yang khas dari pneumonia virus.',
        'badge': 'PERHATIAN'
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def img_to_b64(pil_img, fmt='JPEG'):
    buf = io.BytesIO()
    pil_img.convert('RGB').save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ─── SHARED NAV SNIPPET ───
def make_nav(active):
    dash_color = 'white' if active == 'dashboard' else '#64748B'
    dash_bg    = 'rgba(16,185,129,0.12)' if active == 'dashboard' else 'transparent'
    det_color  = 'white' if active == 'detection' else '#64748B'
    det_bg     = 'rgba(16,185,129,0.12)' if active == 'detection' else 'transparent'
    return f"""
<nav style="background:#0A0F1E;border-bottom:1px solid rgba(255,255,255,0.07);padding:0 32px;height:64px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;">
    <div style="display:flex;align-items:center;gap:12px;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#10B981,#059669);border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:700;color:white;font-size:14px;">C</div>
        <span style="font-size:17px;font-weight:700;color:#F1F5F9;">Cov<span style="color:#10B981;">Detect</span></span>
        <span style="font-size:11px;background:rgba(16,185,129,0.12);color:#10B981;border:1px solid rgba(16,185,129,0.25);padding:2px 8px;border-radius:5px;font-weight:600;">MobileNetV2</span>
    </div>
    <div style="display:flex;align-items:center;gap:4px;">
        <a href="/dashboard" style="padding:8px 16px;border-radius:8px;font-size:14px;font-weight:500;text-decoration:none;color:{dash_color};background:{dash_bg};">Dashboard</a>
        <a href="/detection" style="padding:8px 16px;border-radius:8px;font-size:14px;font-weight:500;text-decoration:none;color:{det_color};background:{det_bg};">Deteksi</a>
        <a href="/logout" style="margin-left:8px;padding:8px 16px;border-radius:8px;font-size:14px;font-weight:500;text-decoration:none;background:rgba(239,68,68,0.1);color:#FCA5A5;border:1px solid rgba(239,68,68,0.2);">Keluar</a>
    </div>
</nav>"""


LOGIN_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CovDetect — Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
        body{font-family:'Inter',sans-serif;background:#0A0F1E;min-height:100vh;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;}
        .bg-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(16,185,129,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(16,185,129,0.04) 1px,transparent 1px);background-size:40px 40px;}
        .card{position:relative;z-index:10;background:rgba(15,23,42,0.95);border:1px solid rgba(255,255,255,0.08);border-radius:24px;padding:48px 44px;width:100%;max-width:440px;backdrop-filter:blur(12px);}
        .logo-wrap{display:flex;align-items:center;gap:12px;margin-bottom:32px;}
        .logo-icon{width:44px;height:44px;background:linear-gradient(135deg,#10B981,#059669);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;color:white;font-weight:700;}
        .logo-text{font-size:20px;font-weight:700;color:#F1F5F9;}.logo-text span{color:#10B981;}
        .badge-version{margin-left:auto;font-size:11px;font-weight:600;background:rgba(16,185,129,0.15);color:#10B981;border:1px solid rgba(16,185,129,0.3);padding:3px 8px;border-radius:6px;}
        h1{font-size:28px;font-weight:700;color:#F1F5F9;}.subtitle{font-size:14px;color:#64748B;margin-top:8px;margin-bottom:32px;line-height:1.5;}
        .flash-msg{padding:12px 16px;border-radius:10px;font-size:13px;margin-bottom:20px;}
        .flash-danger{background:rgba(239,68,68,0.1);color:#FCA5A5;border:1px solid rgba(239,68,68,0.2);}
        .flash-success{background:rgba(16,185,129,0.1);color:#6EE7B7;border:1px solid rgba(16,185,129,0.2);}
        .form-group{margin-bottom:20px;}
        label{display:block;font-size:13px;font-weight:500;color:#94A3B8;margin-bottom:8px;}
        .input-wrap{position:relative;}
        .input-icon{position:absolute;left:14px;top:50%;transform:translateY(-50%);color:#475569;font-size:16px;}
        input[type=text],input[type=password]{width:100%;background:rgba(30,41,59,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 14px 14px 42px;font-size:14px;font-family:inherit;color:#F1F5F9;outline:none;transition:border-color 0.2s;}
        input:focus{border-color:rgba(16,185,129,0.5);}
        input::placeholder{color:#475569;}
        .btn-submit{width:100%;background:linear-gradient(135deg,#10B981,#059669);color:white;border:none;border-radius:12px;padding:14px;font-size:15px;font-weight:600;font-family:inherit;cursor:pointer;margin-top:8px;}
        .hint{margin-top:24px;padding:14px 16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;font-size:12px;color:#475569;}
        .hint code{color:#64748B;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;}
    </style>
</head>
<body>
    <div class="bg-grid"></div>
    <div class="card">
        <div class="logo-wrap">
            <div class="logo-icon">C</div>
            <div class="logo-text">Cov<span>Detect</span></div>
            <div class="badge-version">v2.0</div>
        </div>
        <h1>Selamat datang kembali</h1>
        <p class="subtitle">Masuk untuk mengakses sistem klasifikasi radiologi berbasis AI</p>
        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}{% for category, message in messages %}
          <div class="flash-msg flash-{{ category }}">{{ message }}</div>
          {% endfor %}{% endif %}
        {% endwith %}
        <form action="/login" method="POST">
            <div class="form-group">
                <label>Username</label>
                <div class="input-wrap">
                    <span class="input-icon">👤</span>
                    <input type="text" name="username" placeholder="Masukkan username" required>
                </div>
            </div>
            <div class="form-group">
                <label>Password</label>
                <div class="input-wrap">
                    <span class="input-icon">🔒</span>
                    <input type="password" name="password" placeholder="Masukkan password" required>
                </div>
            </div>
            <button type="submit" class="btn-submit">Masuk ke Sistem →</button>
        </form>
        <div class="hint">Demo: username <code>admin</code> &nbsp;/&nbsp; password <code>covid2026</code></div>
    </div>
</body>
</html>
"""


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CovDetect — Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
        body{font-family:'Inter',sans-serif;background:#060B18;color:#CBD5E1;min-height:100vh;}
        a{text-decoration:none;}
        .page{max-width:1200px;margin:0 auto;padding:40px 32px;}
        .hero{background:linear-gradient(135deg,#0D1B2A 0%,#0F2337 50%,#0A1628 100%);border:1px solid rgba(16,185,129,0.15);border-radius:20px;padding:48px;margin-bottom:32px;position:relative;overflow:hidden;}
        .hero-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(16,185,129,0.1);color:#10B981;border:1px solid rgba(16,185,129,0.25);padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;margin-bottom:20px;}
        .hero h1{font-size:36px;font-weight:700;color:#F1F5F9;line-height:1.2;max-width:600px;margin-bottom:16px;}
        .hero p{font-size:15px;color:#64748B;max-width:520px;line-height:1.7;margin-bottom:28px;}
        .btn-primary{display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,#10B981,#059669);color:white;padding:14px 28px;border-radius:12px;font-size:15px;font-weight:600;}
        .stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px;}
        .stat-card{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:24px;}
        .stat-icon{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;margin-bottom:16px;}
        .stat-label{font-size:12px;color:#475569;font-weight:500;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em;}
        .stat-value{font-size:24px;font-weight:700;color:#F1F5F9;}
        .stat-sub{font-size:12px;color:#64748B;margin-top:4px;}
        .section-title{font-size:18px;font-weight:600;color:#F1F5F9;margin-bottom:16px;}
        .class-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px;}
        .class-card{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:20px;}
        .class-badge{display:inline-block;padding:3px 10px;border-radius:6px;font-size:10px;font-weight:700;letter-spacing:0.06em;margin-bottom:12px;}
        .class-name{font-size:16px;font-weight:600;color:#F1F5F9;margin-bottom:8px;}
        .class-desc{font-size:13px;color:#64748B;line-height:1.5;}
        .info-section{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:24px;}
        .info-row{display:flex;align-items:flex-start;gap:16px;padding:16px 0;border-bottom:1px solid rgba(255,255,255,0.05);}
        .info-row:last-child{border-bottom:none;padding-bottom:0;}
        .info-dot{width:8px;height:8px;border-radius:50%;margin-top:5px;flex-shrink:0;}
        .info-label{font-size:14px;font-weight:500;color:#94A3B8;min-width:160px;}
        .info-value{font-size:14px;color:#CBD5E1;}
    </style>
</head>
<body>
{{ nav | safe }}
<div class="page">
    <div class="hero">
        <div class="hero-badge">🧠 Deep Learning · Transfer Learning</div>
        <h1>Analisis Citra Radiologi Paru-paru Berbasis AI</h1>
        <p>Sistem terintegrasi menggunakan arsitektur MobileNetV2 yang dilatih dengan teknik transfer learning untuk mengklasifikasikan kondisi paru-paru dari citra X-ray secara presisi dan otomatis.</p>
        <a href="/detection" class="btn-primary">Mulai Deteksi Sekarang →</a>
    </div>
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-icon" style="background:rgba(16,185,129,0.12);">🧬</div><div class="stat-label">Arsitektur Model</div><div class="stat-value" style="font-size:18px;">MobileNetV2</div><div class="stat-sub">Transfer Learning</div></div>
        <div class="stat-card"><div class="stat-icon" style="background:rgba(59,130,246,0.12);">📐</div><div class="stat-label">Dimensi Input</div><div class="stat-value">224px</div><div class="stat-sub">× 224 pixel (RGB)</div></div>
        <div class="stat-card"><div class="stat-icon" style="background:rgba(168,85,247,0.12);">🏷️</div><div class="stat-label">Total Kelas</div><div class="stat-value">4 Kelas</div><div class="stat-sub">COVID, Normal, dll.</div></div>
        <div class="stat-card"><div class="stat-icon" style="background:rgba(245,158,11,0.12);">📁</div><div class="stat-label">Format Didukung</div><div class="stat-value" style="font-size:18px;">JPG / PNG</div><div class="stat-sub">Citra X-ray dada</div></div>
    </div>
    <div class="section-title">Kategori Klasifikasi</div>
    <div class="class-grid">
        <div class="class-card"><div class="class-badge" style="background:rgba(239,68,68,0.12);color:#EF4444;">KRITIS</div><div class="class-name" style="color:#EF4444;">COVID-19</div><div class="class-desc">Indikasi infeksi spesifik virus COVID-19 pada jaringan paru-paru pasien.</div></div>
        <div class="class-card"><div class="class-badge" style="background:rgba(245,158,11,0.12);color:#F59E0B;">WASPADA</div><div class="class-name" style="color:#F59E0B;">Lung Opacity</div><div class="class-desc">Area kekeruhan atau opasitas non-spesifik, bisa efusi cairan atau radang.</div></div>
        <div class="class-card"><div class="class-badge" style="background:rgba(16,185,129,0.12);color:#10B981;">NORMAL</div><div class="class-name" style="color:#10B981;">Normal</div><div class="class-desc">Kondisi paru-paru sehat tanpa tanda infeksi atau kelainan yang terdeteksi.</div></div>
        <div class="class-card"><div class="class-badge" style="background:rgba(59,130,246,0.12);color:#3B82F6;">PERHATIAN</div><div class="class-name" style="color:#3B82F6;">Viral Pneumonia</div><div class="class-desc">Pola infeksi khas dari pneumonia yang disebabkan oleh virus selain COVID-19.</div></div>
    </div>
    <div class="section-title">Spesifikasi Teknis</div>
    <div class="info-section">
        <div class="info-row"><div class="info-dot" style="background:#10B981;"></div><div class="info-label">Model</div><div class="info-value">MobileNetV2 dengan bobot ImageNet (Fine-tuned)</div></div>
        <div class="info-row"><div class="info-dot" style="background:#3B82F6;"></div><div class="info-label">Preprocessing</div><div class="info-value">Normalisasi MobileNetV2 (nilai piksel [-1, 1])</div></div>
        <div class="info-row"><div class="info-dot" style="background:#A855F7;"></div><div class="info-label">Ukuran Input</div><div class="info-value">224 × 224 piksel, 3 channel warna (RGB)</div></div>
        <div class="info-row"><div class="info-dot" style="background:#F59E0B;"></div><div class="info-label">Output</div><div class="info-value">Probabilitas Softmax untuk 4 kelas penyakit paru-paru</div></div>
    </div>
</div>
</body>
</html>
"""


DETECTION_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CovDetect — Upload Gambar</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
        body{font-family:'Inter',sans-serif;background:#060B18;color:#CBD5E1;min-height:100vh;}
        a{text-decoration:none;}
        .page{max-width:900px;margin:0 auto;padding:40px 32px;}
        .page-header{margin-bottom:32px;}
        .page-header h1{font-size:28px;font-weight:700;color:#F1F5F9;}
        .page-header p{font-size:14px;color:#64748B;margin-top:6px;line-height:1.6;}

        /* Step indicator */
        .steps{display:flex;align-items:center;gap:0;margin-bottom:36px;}
        .step{display:flex;align-items:center;gap:10px;}
        .step-num{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;border:2px solid;}
        .step-label{font-size:13px;font-weight:500;}
        .step.active .step-num{background:#10B981;border-color:#10B981;color:white;}
        .step.active .step-label{color:#F1F5F9;}
        .step.done .step-num{background:rgba(16,185,129,0.2);border-color:#10B981;color:#10B981;}
        .step.done .step-label{color:#64748B;}
        .step.pending .step-num{background:transparent;border-color:rgba(255,255,255,0.15);color:#475569;}
        .step.pending .step-label{color:#475569;}
        .step-line{flex:1;height:1px;background:rgba(255,255,255,0.08);margin:0 12px;min-width:40px;}

        .panel{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.07);border-radius:20px;padding:36px;}
        .upload-zone{border:2px dashed rgba(255,255,255,0.1);border-radius:16px;padding:64px 24px;text-align:center;cursor:pointer;transition:all 0.2s;position:relative;background:rgba(255,255,255,0.02);}
        .upload-zone:hover,.upload-zone.dragging{border-color:rgba(16,185,129,0.5);background:rgba(16,185,129,0.04);}
        .upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
        .upload-icon{width:72px;height:72px;background:rgba(16,185,129,0.1);border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:32px;margin:0 auto 20px;}
        .upload-title{font-size:17px;font-weight:600;color:#F1F5F9;margin-bottom:8px;}
        .upload-sub{font-size:13px;color:#475569;line-height:1.5;}
        .file-selected{display:none;align-items:center;gap:12px;padding:14px 18px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:12px;margin-top:20px;}
        .file-selected.show{display:flex;}
        .file-name{font-size:13px;color:#10B981;font-weight:500;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
        .btn-next{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;background:linear-gradient(135deg,#10B981,#059669);color:white;border:none;border-radius:12px;padding:14px;font-size:15px;font-weight:600;font-family:inherit;cursor:pointer;margin-top:24px;opacity:0.4;pointer-events:none;transition:opacity 0.2s;}
        .btn-next.ready{opacity:1;pointer-events:auto;}
    </style>
</head>
<body>
{{ nav | safe }}
<div class="page">
    <div class="page-header">
        <h1>🔬 Analisis Citra Radiologi</h1>
        <p>Ikuti 3 tahap: upload gambar → preprocessing → deteksi .</p>
    </div>

    <div class="steps">
        <div class="step active">
            <div class="step-num">1</div>
            <div class="step-label">Upload Gambar</div>
        </div>
        <div class="step-line"></div>
        <div class="step pending">
            <div class="step-num">2</div>
            <div class="step-label">Preprocessing</div>
        </div>
        <div class="step-line"></div>
        <div class="step pending">
            <div class="step-num">3</div>
            <div class="step-label">Deteksi </div>
        </div>
    </div>

    <div class="panel">
        <form action="/preprocessing" method="POST" enctype="multipart/form-data" id="uploadForm">
            <div class="upload-zone" id="dropZone">
                <input type="file" name="file" id="fileInput" accept=".jpg,.jpeg,.png" required>
                <div class="upload-icon">🩻</div>
                <div class="upload-title">Pilih atau Drop Gambar X-Ray di Sini</div>
                <div class="upload-sub">Format yang didukung: JPG, JPEG, PNG<br>Gambar akan dapat diedit ukurannya sebelum dianalisis</div>
            </div>
            <div class="file-selected" id="fileSelected">
                <span style="font-size:20px;">🖼️</span>
                <span class="file-name" id="fileName">—</span>
                <span style="font-size:12px;color:#475569;white-space:nowrap;" id="fileMeta"></span>
            </div>
            <button type="submit" class="btn-next" id="btnNext">
                Lanjut ke Preprocessing →
            </button>
        </form>
    </div>
</div>
<script>
    const fi = document.getElementById('fileInput');
    const fs = document.getElementById('fileSelected');
    const fn = document.getElementById('fileName');
    const fm = document.getElementById('fileMeta');
    const btn = document.getElementById('btnNext');
    const dz = document.getElementById('dropZone');

    function handleFile(f) {
        fn.textContent = f.name;
        fm.textContent = (f.size/1024).toFixed(1) + ' KB';
        fs.classList.add('show');
        btn.classList.add('ready');
    }
    fi.addEventListener('change', e => { if(e.target.files[0]) handleFile(e.target.files[0]); });
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragging'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('dragging'));
    dz.addEventListener('drop', e => {
        e.preventDefault(); dz.classList.remove('dragging');
        if(e.dataTransfer.files[0]){ fi.files = e.dataTransfer.files; handleFile(e.dataTransfer.files[0]); }
    });
</script>
</body>
</html>
"""


PREPROCESSING_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CovDetect — Preprocessing</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
        body{font-family:'Inter',sans-serif;background:#060B18;color:#CBD5E1;min-height:100vh;}
        a{text-decoration:none;}
        .page{max-width:1200px;margin:0 auto;padding:40px 32px;}
        .page-header{margin-bottom:32px;}
        .page-header h1{font-size:28px;font-weight:700;color:#F1F5F9;}
        .page-header p{font-size:14px;color:#64748B;margin-top:6px;line-height:1.6;}

        .steps{display:flex;align-items:center;gap:0;margin-bottom:36px;}
        .step{display:flex;align-items:center;gap:10px;}
        .step-num{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;border:2px solid;}
        .step-label{font-size:13px;font-weight:500;}
        .step.active .step-num{background:#10B981;border-color:#10B981;color:white;}
        .step.active .step-label{color:#F1F5F9;}
        .step.done .step-num{background:rgba(16,185,129,0.2);border-color:#10B981;color:#10B981;}
        .step.done .step-label{color:#64748B;}
        .step.pending .step-num{background:transparent;border-color:rgba(255,255,255,0.15);color:#475569;}
        .step.pending .step-label{color:#475569;}
        .step-line{flex:1;height:1px;background:rgba(255,255,255,0.08);margin:0 12px;min-width:40px;}

        .main-grid{display:grid;grid-template-columns:1fr 380px;gap:24px;align-items:start;}

        /* Preview panel */
        .preview-panel{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.07);border-radius:20px;padding:28px;}
        .panel-label{font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:16px;}
        .img-compare{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;}
        .img-box{border-radius:12px;overflow:hidden;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.06);position:relative;}
        .img-box img{width:100%;display:block;object-fit:contain;max-height:280px;}
        .img-box-label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.65);font-size:11px;color:#94A3B8;padding:6px 10px;font-weight:500;}
        .img-box-badge{position:absolute;top:8px;left:8px;font-size:10px;font-weight:700;padding:3px 8px;border-radius:5px;letter-spacing:0.05em;}

        /* Metadata strip */
        .meta-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:8px;}
        .meta-item{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;}
        .meta-key{font-size:10px;color:#475569;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;}
        .meta-val{font-size:15px;font-weight:600;color:#F1F5F9;}

        /* Control panel */
        .ctrl-panel{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.07);border-radius:20px;padding:28px;display:flex;flex-direction:column;gap:24px;}
        .ctrl-section{border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:20px;}
        .ctrl-section:last-of-type{border-bottom:none;padding-bottom:0;}
        .ctrl-title{font-size:13px;font-weight:600;color:#94A3B8;margin-bottom:14px;display:flex;align-items:center;gap:8px;}
        .ctrl-row{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
        .ctrl-row:last-child{margin-bottom:0;}
        .ctrl-lbl{font-size:13px;color:#64748B;min-width:52px;}
        .ctrl-input{flex:1;-webkit-appearance:none;appearance:none;height:4px;border-radius:2px;background:rgba(255,255,255,0.1);outline:none;cursor:pointer;}
        .ctrl-input::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#10B981;cursor:pointer;}
        .ctrl-input::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#10B981;cursor:pointer;border:none;}
        .ctrl-val{font-size:13px;font-weight:600;color:#F1F5F9;min-width:52px;text-align:right;}

        .preset-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
        .preset-btn{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 12px;font-size:12px;font-weight:500;color:#94A3B8;cursor:pointer;font-family:inherit;transition:all 0.15s;text-align:left;}
        .preset-btn:hover{background:rgba(16,185,129,0.08);border-color:rgba(16,185,129,0.3);color:#10B981;}
        .preset-btn.active{background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.4);color:#10B981;}
        .preset-btn .ps-size{font-size:11px;color:#475569;margin-top:2px;}
        .preset-btn.active .ps-size{color:rgba(16,185,129,0.6);}

        .info-callout{background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.15);border-radius:10px;padding:12px 14px;font-size:12px;color:#93C5FD;line-height:1.6;}
        .info-callout strong{color:#60A5FA;}

        .action-row{display:flex;flex-direction:column;gap:10px;}
        .btn-analyze{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;background:linear-gradient(135deg,#10B981,#059669);color:white;border:none;border-radius:12px;padding:14px;font-size:15px;font-weight:600;font-family:inherit;cursor:pointer;transition:opacity 0.2s;}
        .btn-analyze:hover{opacity:0.9;}
        .btn-back{display:flex;align-items:center;justify-content:center;gap:6px;width:100%;background:transparent;border:1px solid rgba(255,255,255,0.1);color:#64748B;border-radius:12px;padding:12px;font-size:14px;font-weight:500;font-family:inherit;cursor:pointer;text-decoration:none;}
        .btn-back:hover{background:rgba(255,255,255,0.04);color:#94A3B8;}

        .loading-overlay{display:none;position:fixed;inset:0;background:rgba(6,11,24,0.9);z-index:200;flex-direction:column;align-items:center;justify-content:center;gap:16px;}
        .loading-overlay.active{display:flex;}
        .spinner{width:44px;height:44px;border:3px solid rgba(16,185,129,0.2);border-top-color:#10B981;border-radius:50%;animation:spin 0.8s linear infinite;}
        @keyframes spin{to{transform:rotate(360deg);}}
        .loading-text{font-size:14px;color:#64748B;}
        .loading-sub{font-size:12px;color:#334155;}
    </style>
</head>
<body>
{{ nav | safe }}

<div class="loading-overlay" id="loadingOverlay">
    <div class="spinner"></div>
    <div class="loading-text">Menjalankan analisis MobileNetV2...</div>
    <div class="loading-sub">Mohon tunggu, model sedang memproses gambar</div>
</div>

<div class="page">
    <div class="page-header">
        <h1>🎛️ Preprocessing Gambar</h1>
        <p>Atur ukuran gambar sebelum dianalisis. Bandingkan gambar asli dan hasil preprocessing secara langsung.</p>
    </div>

    <div class="steps">
        <div class="step done">
            <div class="step-num">✓</div>
            <div class="step-label">Upload Gambar</div>
        </div>
        <div class="step-line"></div>
        <div class="step active">
            <div class="step-num">2</div>
            <div class="step-label">Preprocessing</div>
        </div>
        <div class="step-line"></div>
        <div class="step pending">
            <div class="step-num">3</div>
            <div class="step-label">Deteksi </div>
        </div>
    </div>

    <div class="main-grid">

        <!-- Kiri: preview gambar -->
        <div class="preview-panel">
            <div class="panel-label">📸 Perbandingan Gambar</div>
            <div class="img-compare">
                <div class="img-box">
                    <div class="img-box-badge" style="background:rgba(100,116,139,0.8);color:#CBD5E1;">ASLI</div>
                    <img id="imgOriginal" src="data:image/jpeg;base64,{{ original_b64 }}" alt="Gambar Asli">
                    <div class="img-box-label">{{ orig_w }} × {{ orig_h }} px</div>
                </div>
                <div class="img-box">
                    <div class="img-box-badge" style="background:rgba(16,185,129,0.8);color:white;">PREVIEW</div>
                    <img id="imgPreview" src="data:image/jpeg;base64,{{ preview_b64 }}" alt="Preview Preprocessing">
                    <div class="img-box-label" id="previewLabel">{{ target_w }} × {{ target_h }} px</div>
                </div>
            </div>

            <div class="meta-strip">
                <div class="meta-item">
                    <div class="meta-key">Ukuran Asli</div>
                    <div class="meta-val">{{ orig_w }}×{{ orig_h }}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-key">Ukuran Target</div>
                    <div class="meta-val" id="metaTarget">{{ target_w }}×{{ target_h }}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-key">Ukuran File</div>
                    <div class="meta-val">{{ file_kb }} KB</div>
                </div>
            </div>
        </div>

        <!-- Kanan: kontrol -->
        <div class="ctrl-panel">

            <div class="ctrl-section">
                <div class="ctrl-title">📐 Atur Ukuran Target</div>
                <div class="ctrl-row">
                    <span class="ctrl-lbl">Lebar</span>
                    <input type="range" class="ctrl-input" id="sliderW" min="32" max="512" step="8" value="{{ target_w }}">
                    <span class="ctrl-val"><span id="valW">{{ target_w }}</span>px</span>
                </div>
                <div class="ctrl-row">
                    <span class="ctrl-lbl">Tinggi</span>
                    <input type="range" class="ctrl-input" id="sliderH" min="32" max="512" step="8" value="{{ target_h }}">
                    <span class="ctrl-val"><span id="valH">{{ target_h }}</span>px</span>
                </div>
                <div style="margin-top:10px;display:flex;align-items:center;gap:8px;">
                    <input type="checkbox" id="lockAspect" style="accent-color:#10B981;cursor:pointer;">
                    <label for="lockAspect" style="font-size:12px;color:#64748B;cursor:pointer;">Kunci rasio aspek</label>
                </div>
            </div>

            <div class="ctrl-section">
                <div class="ctrl-title">⚡ Preset Ukuran</div>
                <div class="preset-grid">
                    <button class="preset-btn" onclick="applyPreset(64,64,this)">
                        Kecil
                        <div class="ps-size">64 × 64 px</div>
                    </button>
                    <button class="preset-btn" onclick="applyPreset(128,128,this)">
                        Sedang
                        <div class="ps-size">128 × 128 px</div>
                    </button>
                    <button class="preset-btn active" onclick="applyPreset(224,224,this)">
                        MobileNetV2 ★
                        <div class="ps-size">224 × 224 px</div>
                    </button>
                    <button class="preset-btn" onclick="applyPreset(256,256,this)">
                        Besar
                        <div class="ps-size">256 × 256 px</div>
                    </button>
                    <button class="preset-btn" onclick="applyPreset(320,320,this)">
                        HD Kecil
                        <div class="ps-size">320 × 320 px</div>
                    </button>
                    <button class="preset-btn" onclick="applyPreset(512,512,this)">
                        HD Penuh
                        <div class="ps-size">512 × 512 px</div>
                    </button>
                </div>
            </div>

            <div class="ctrl-section">
                <div class="info-callout">
                    <strong>⚠️ Catatan penting:</strong> Model MobileNetV2 hanya menerima input <strong>224 × 224 px</strong>. Jika Anda memilih ukuran lain, sistem akan otomatis meresize ulang ke 224×224 saat deteksi berjalan.
                </div>
            </div>

            <div class="action-row">
                <form action="/analyze" method="POST" id="analyzeForm">
                    <input type="hidden" name="filename" value="{{ filename }}">
                    <input type="hidden" name="target_w" id="formW" value="{{ target_w }}">
                    <input type="hidden" name="target_h" id="formH" value="{{ target_h }}">
                    <button type="submit" class="btn-analyze" onclick="document.getElementById('loadingOverlay').classList.add('active')">
                        🧠 Jalankan Deteksi  →
                    </button>
                </form>
                <a href="/detection" class="btn-back">← Ganti Gambar</a>
            </div>

        </div>
    </div>
</div>

<script>
    const origW = {{ orig_w }};
    const origH = {{ orig_h }};
    const origB64 = "{{ original_b64 }}";
    const sliderW = document.getElementById('sliderW');
    const sliderH = document.getElementById('sliderH');
    const valW = document.getElementById('valW');
    const valH = document.getElementById('valH');
    const formW = document.getElementById('formW');
    const formH = document.getElementById('formH');
    const previewLabel = document.getElementById('previewLabel');
    const metaTarget = document.getElementById('metaTarget');
    const imgPreview = document.getElementById('imgPreview');
    const lockAspect = document.getElementById('lockAspect');
    let debounceTimer = null;

    function updatePreview(w, h) {
        valW.textContent = w;
        valH.textContent = h;
        formW.value = w;
        formH.value = h;
        previewLabel.textContent = w + ' × ' + h + ' px';
        metaTarget.textContent = w + '×' + h;

        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => fetchPreview(w, h), 300);
    }

    function fetchPreview(w, h) {
        fetch('/api/preview', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: '{{ filename }}', w: w, h: h})
        })
        .then(r => r.json())
        .then(data => {
            if(data.b64) {
                imgPreview.src = 'data:image/jpeg;base64,' + data.b64;
            }
        });
    }

    sliderW.addEventListener('input', function() {
        const w = parseInt(this.value);
        if(lockAspect.checked) {
            const h = Math.round(w * origH / origW / 8) * 8;
            sliderH.value = Math.min(512, Math.max(32, h));
        }
        updatePreview(parseInt(sliderW.value), parseInt(sliderH.value));
    });

    sliderH.addEventListener('input', function() {
        const h = parseInt(this.value);
        if(lockAspect.checked) {
            const w = Math.round(h * origW / origH / 8) * 8;
            sliderW.value = Math.min(512, Math.max(32, w));
        }
        updatePreview(parseInt(sliderW.value), parseInt(sliderH.value));
    });

    function applyPreset(w, h, el) {
        sliderW.value = w;
        sliderH.value = h;
        updatePreview(w, h);
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        el.classList.add('active');
    }
</script>
</body>
</html>
"""


RESULT_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CovDetect — Hasil Deteksi</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
        body{font-family:'Inter',sans-serif;background:#060B18;color:#CBD5E1;min-height:100vh;}
        a{text-decoration:none;}
        .page{max-width:1200px;margin:0 auto;padding:40px 32px;}
        .page-header{margin-bottom:32px;}
        .page-header h1{font-size:28px;font-weight:700;color:#F1F5F9;}
        .page-header p{font-size:14px;color:#64748B;margin-top:6px;}

        .steps{display:flex;align-items:center;gap:0;margin-bottom:36px;}
        .step{display:flex;align-items:center;gap:10px;}
        .step-num{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;border:2px solid;}
        .step-label{font-size:13px;font-weight:500;}
        .step.done .step-num{background:rgba(16,185,129,0.2);border-color:#10B981;color:#10B981;}
        .step.done .step-label{color:#64748B;}
        .step.active .step-num{background:#10B981;border-color:#10B981;color:white;}
        .step.active .step-label{color:#F1F5F9;}
        .step-line{flex:1;height:1px;background:rgba(255,255,255,0.08);margin:0 12px;min-width:40px;}

        .result-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;}
        .panel{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.07);border-radius:20px;padding:28px;}
        .panel-label{font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:16px;}

        .img-box{border-radius:12px;overflow:hidden;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.06);position:relative;margin-bottom:16px;}
        .img-box img{width:100%;display:block;object-fit:contain;max-height:300px;}
        .img-box-label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.65);font-size:11px;color:#94A3B8;padding:6px 10px;}

        .meta-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
        .meta-item{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;}
        .meta-key{font-size:10px;color:#475569;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;}
        .meta-val{font-size:14px;font-weight:600;color:#F1F5F9;}

        .result-main{text-align:center;padding:16px 0 24px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:24px;}
        .result-icon{width:80px;height:80px;border-radius:20px;display:flex;align-items:center;justify-content:center;font-size:36px;margin:0 auto 16px;}
        .result-badge{display:inline-block;padding:4px 12px;border-radius:8px;font-size:11px;font-weight:700;letter-spacing:0.08em;margin-bottom:10px;}
        .result-label{font-size:32px;font-weight:700;margin-bottom:12px;}
        .conf-wrap{display:inline-flex;align-items:baseline;gap:6px;padding:8px 20px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;}
        .conf-lbl{font-size:12px;color:#64748B;}
        .conf-val{font-size:22px;font-weight:700;}
        .result-desc{font-size:13px;color:#64748B;margin-top:14px;line-height:1.6;}

        .prob-title{font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;}
        .prob-row{margin-bottom:12px;}
        .prob-hdr{display:flex;justify-content:space-between;margin-bottom:6px;}
        .prob-name{font-size:13px;font-weight:500;color:#94A3B8;}
        .prob-pct{font-size:13px;font-weight:600;}
        .prob-bg{height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;}
        .prob-fill{height:100%;border-radius:3px;}

        .disclaimer{margin-top:20px;padding:12px 16px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:10px;font-size:12px;color:#92400E;line-height:1.5;}
        .disclaimer strong{color:#F59E0B;}
        .btn-new{display:flex;align-items:center;justify-content:center;gap:8px;background:linear-gradient(135deg,#10B981,#059669);color:white;border:none;border-radius:12px;padding:13px 24px;font-size:14px;font-weight:600;font-family:inherit;cursor:pointer;text-decoration:none;margin-top:20px;}
    </style>
</head>
<body>
{{ nav | safe }}
<div class="page">
    <div class="page-header">
        <h1>✅ Hasil Deteksi </h1>
        <p>Analisis selesai. Berikut hasil klasifikasi citra X-ray menggunakan MobileNetV2.</p>
    </div>

    <div class="steps">
        <div class="step done"><div class="step-num">✓</div><div class="step-label">Upload Gambar</div></div>
        <div class="step-line"></div>
        <div class="step done"><div class="step-num">✓</div><div class="step-label">Preprocessing</div></div>
        <div class="step-line"></div>
        <div class="step active"><div class="step-num">3</div><div class="step-label">Deteksi </div></div>
    </div>

    <div class="result-grid">
        <!-- Kiri: gambar -->
        <div class="panel">
            <div class="panel-label">🩻 Gambar yang Dianalisis</div>
            <div class="img-box">
                <img src="data:image/jpeg;base64,{{ processed_b64 }}" alt="Gambar preprocessed">
                <div class="img-box-label">Setelah preprocessing: {{ proc_w }} × {{ proc_h }} px → dianalisis sebagai 224×224</div>
            </div>
            <div class="meta-strip">
                <div class="meta-item"><div class="meta-key">Ukuran Asli</div><div class="meta-val">{{ orig_w }}×{{ orig_h }}</div></div>
                <div class="meta-item"><div class="meta-key">Ukuran Preprocessing</div><div class="meta-val">{{ proc_w }}×{{ proc_h }}</div></div>
                <div class="meta-item"><div class="meta-key">Input Model</div><div class="meta-val">224×224</div></div>
            </div>
            <a href="/detection" class="btn-new">🔄 Analisis Gambar Baru</a>
        </div>

        <!-- Kanan: hasil -->
        <div class="panel">
            <div class="panel-label">📊 Hasil Diagnosis </div>
            {% set meta = class_meta.get(result, {}) %}
            <div class="result-main">
                <div class="result-icon" style="background:{{ meta.get('bg','#1e293b') }};">
                    {{ '🦠' if result=='COVID' else ('☁️' if result=='Lung Opacity' else ('✅' if result=='Normal' else '⚠️')) }}
                </div>
                <div class="result-badge" style="background:{{ meta.get('bg','#1e293b') }};color:{{ meta.get('color','#94A3B8') }};">{{ meta.get('badge','—') }}</div>
                <div class="result-label" style="color:{{ meta.get('color','#F1F5F9') }};">{{ result }}</div>
                <div class="conf-wrap">
                    <span class="conf-lbl">Kepercayaan</span>
                    <span class="conf-val" style="color:{{ meta.get('color','#10B981') }};">{{ confidence }}</span>
                </div>
                <div class="result-desc">{{ meta.get('desc','') }}</div>
            </div>

            <div class="prob-title">Distribusi Probabilitas Semua Kelas</div>
            {% for class_name, prob in raw_preds %}
            {% set is_best = class_name == result %}
            <div class="prob-row">
                <div class="prob-hdr">
                    <span class="prob-name" style="{{ 'color:#F1F5F9;font-weight:600;' if is_best else '' }}">{{ class_name }}</span>
                    <span class="prob-pct" style="color:{{ class_meta.get(class_name,{}).get('color','#64748B') }};">{{ "%.1f"|format(prob * 100) }}%</span>
                </div>
                <div class="prob-bg">
                    <div class="prob-fill" style="width:{{ prob * 100 }}%;background:{{ class_meta.get(class_name,{}).get('color','#64748B') }};"></div>
                </div>
            </div>
            {% endfor %}

            <div class="disclaimer">
                <strong>⚠️ Perhatian Medis:</strong> Hasil ini merupakan output sistem AI dan bukan diagnosis medis resmi. Selalu konsultasikan hasil dengan dokter atau tenaga ahli radiologi.
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""


# ─── ROUTES ───

@app.route('/')
def index():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'covid2026':
            session['logged_in'] = True
            flash('Selamat datang kembali, Tenaga Medis!', 'success')
            return redirect(url_for('dashboard'))
        flash('Username atau password salah.', 'danger')
    return render_template_string(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template_string(DASHBOARD_HTML, nav=make_nav('dashboard'))

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template_string(DETECTION_HTML, nav=make_nav('detection'))

@app.route('/preprocessing', methods=['POST'])
def preprocessing():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files or request.files['file'].filename == '':
        flash('Tidak ada file dipilih.', 'warning')
        return redirect(url_for('detection'))

    file = request.files['file']
    if not allowed_file(file.filename):
        flash('Format file tidak didukung.', 'warning')
        return redirect(url_for('detection'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Buka gambar original
    pil_orig = PILImage.open(filepath).convert('RGB')
    orig_w, orig_h = pil_orig.size

    # Default target: 224x224
    target_w, target_h = 224, 224
    pil_preview = pil_orig.resize((target_w, target_h), PILImage.LANCZOS)

    orig_b64    = img_to_b64(pil_orig)
    preview_b64 = img_to_b64(pil_preview)
    file_kb     = round(os.path.getsize(filepath) / 1024, 1)

    return render_template_string(PREPROCESSING_HTML,
        nav=make_nav('detection'),
        filename=filename,
        original_b64=orig_b64,
        preview_b64=preview_b64,
        orig_w=orig_w, orig_h=orig_h,
        target_w=target_w, target_h=target_h,
        file_kb=file_kb
    )

@app.route('/api/preview', methods=['POST'])
def api_preview():
    """API endpoint: real-time preview gambar setelah resize."""
    if 'logged_in' not in session:
        return jsonify({'error': 'unauthorized'}), 401

    data = request.get_json()
    filename = secure_filename(data.get('filename', ''))
    w = max(32, min(512, int(data.get('w', 224))))
    h = max(32, min(512, int(data.get('h', 224))))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'file not found'}), 404

    pil_img  = PILImage.open(filepath).convert('RGB')
    resized  = pil_img.resize((w, h), PILImage.LANCZOS)
    b64      = img_to_b64(resized)
    return jsonify({'b64': b64, 'w': w, 'h': h})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    filename = secure_filename(request.form.get('filename', ''))
    target_w = max(32, min(512, int(request.form.get('target_w', 224))))
    target_h = max(32, min(512, int(request.form.get('target_h', 224))))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('File tidak ditemukan.', 'warning')
        return redirect(url_for('detection'))

    # Simpan info dimensi asli
    pil_orig = PILImage.open(filepath).convert('RGB')
    orig_w, orig_h = pil_orig.size

    # Resize sesuai pilihan user untuk preview di hasil
    pil_proc = pil_orig.resize((target_w, target_h), PILImage.LANCZOS)
    processed_b64 = img_to_b64(pil_proc)

    if model is None:
        return render_template_string(RESULT_HTML,
            nav=make_nav('detection'),
            result="Model Error", confidence="0%",
            raw_preds=[], class_meta=CLASS_META,
            processed_b64=processed_b64,
            orig_w=orig_w, orig_h=orig_h,
            proc_w=target_w, proc_h=target_h)

    # Model selalu butuh 224x224 — resize ke sini untuk inferensi
    img_model = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img_model)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions   = model.predict(img_array)
    best_idx      = np.argmax(predictions[0])
    result_label  = CLASS_NAMES[best_idx]
    confidence    = f"{predictions[0][best_idx] * 100:.1f}%"

    return render_template_string(RESULT_HTML,
        nav=make_nav('detection'),
        result=result_label,
        confidence=confidence,
        raw_preds=list(zip(CLASS_NAMES, predictions[0])),
        class_meta=CLASS_META,
        processed_b64=processed_b64,
        orig_w=orig_w, orig_h=orig_h,
        proc_w=target_w, proc_h=target_h)

if __name__ == '__main__':
    app.run(debug=True, port=5000)