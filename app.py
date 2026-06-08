import streamlit as st
from model import (
    load_dataset, train_pca, save_model, load_model_file,
    recognize, compare_two, compare_then_vs_now
)
import os

st.set_page_config(page_title="Deteksi Kemiripan Wajah")
st.title("Deteksi Kemiripan Wajah")
st.caption("Menggunakan metode PCA/SVD (Eigenfaces)")

# ── Load atau latih model ────────────────────────────────────────────
@st.cache_resource
def get_model():
    if os.path.exists("model.pkl"):
        pca, X_pca, labels = load_model_file("model.pkl")
        return pca, X_pca, labels, "dimuat dari file", []
    elif os.path.exists("dataset"):
        X, labels, gagal = load_dataset("dataset")
        if len(X) == 0:
            return None, None, None, "dataset kosong", []
        pca, X_pca = train_pca(X)
        save_model(pca, X_pca, labels)
        return pca, X_pca, labels, "dilatih dari dataset", gagal
    else:
        return None, None, None, "tidak ada dataset", []

pca, X_pca, labels, status, gagal = get_model()

if pca is None:
    st.error("Dataset tidak ditemukan. Pastikan folder dataset ada di proyek.")
    st.stop()

if st.button("🔄 Retrain Model"):
    if os.path.exists("model.pkl"):
        os.remove("model.pkl")
    st.cache_resource.clear()
    st.rerun()

orang_list = sorted(set(labels))
st.success(
    f"Model siap! ({status}) — "
    f"{len(labels)} foto dari {len(orang_list)} orang: "
    f"{', '.join(orang_list)}"
)

if gagal:
    with st.expander(f"⚠️ {len(gagal)} foto wajah tidak terdeteksi otomatis"):
        for f in gagal:
            st.write(f"• {f}")
        st.caption("Foto ini tetap dipakai tapi mungkin kurang akurat.")

# ── Pilih fitur ──────────────────────────────────────────────────────
st.divider()
fitur = st.radio("Pilih fitur:", [
    "Kenali Wajah dari Database",
    "Bandingkan Dua Wajah",
    "Foto Masa Kecil vs Sekarang"       # ← ini yang tadi kurang
])
threshold = st.slider("Threshold kemiripan", 0.0, 1.0, 0.75, 0.01)

# ── Fitur 1: Kenali wajah ────────────────────────────────────────────
if fitur == "Kenali Wajah dari Database":
    st.subheader("Siapa orang ini?")
    st.caption("Upload foto, sistem akan mencari siapa orangnya dari database.")
    foto = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])

    if foto:
        st.image(foto, caption="Foto yang diupload", width=250)
        if st.button("Kenali Sekarang"):
            with st.spinner("Sedang memproses..."):
                nama, skor, detected, top3 = recognize(
                    foto.read(), pca, X_pca, labels, threshold
                )
            if not detected:
                st.warning("⚠️ Wajah tidak terdeteksi otomatis. Hasil mungkin kurang akurat.")
            else:
                st.info("✅ Wajah berhasil terdeteksi.")
            st.metric("Skor Kemiripan Tertinggi", f"{skor:.4f}")
            if nama != "Tidak dikenal":
                st.success(f"Wajah dikenali sebagai: **{nama}**")
            else:
                st.warning("Wajah tidak dikenal. Coba turunkan threshold.")
            st.write("**Top 3 kandidat:**")
            for i, (kandidat, nilai) in enumerate(top3, 1):
                bar_color = "🟢" if nilai >= threshold else "🔴"
                st.write(f"{i}. {bar_color} **{kandidat}** — skor: `{nilai:.4f}`")

# ── Fitur 2: Bandingkan dua wajah ────────────────────────────────────
elif fitur == "Bandingkan Dua Wajah":
    st.subheader("Apakah dua wajah ini sama?")
    st.caption("Upload dua foto untuk dibandingkan satu sama lain.")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Foto Wajah 1", type=["jpg", "jpeg", "png"])
    with col2:
        file2 = st.file_uploader("Foto Wajah 2", type=["jpg", "jpeg", "png"])

    if file1 and file2:
        col1.image(file1, use_container_width=True)
        col2.image(file2, use_container_width=True)
        if st.button("Bandingkan Sekarang"):
            with st.spinner("Sedang memproses..."):
                skor, hasil, det1, det2 = compare_two(
                    file1.read(), file2.read(), pca, threshold
                )
            if not det1:
                st.warning("⚠️ Wajah 1 tidak terdeteksi otomatis.")
            if not det2:
                st.warning("⚠️ Wajah 2 tidak terdeteksi otomatis.")
            st.metric("Skor Kemiripan", f"{skor:.4f}")
            if hasil == "Mirip":
                st.success("✅ Wajah MIRIP!")
            else:
                st.error("❌ Wajah TIDAK mirip.")

# ── Fitur 3: Foto masa kecil vs sekarang ─────────────────────────────
elif fitur == "Foto Masa Kecil vs Sekarang":
    st.subheader("Apakah ini orang yang sama?")
    st.caption(
        "Upload foto masa kecil dan foto sekarang. "
        "Sistem akan menebak apakah keduanya adalah orang yang sama."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📷 Foto Masa Kecil**")
        foto_lama = st.file_uploader(
            "Upload foto masa kecil",
            type=["jpg", "jpeg", "png"],
            key="foto_lama"
        )
    with col2:
        st.markdown("**📷 Foto Sekarang**")
        foto_baru = st.file_uploader(
            "Upload foto sekarang",
            type=["jpg", "jpeg", "png"],
            key="foto_baru"
        )

    threshold_waktu = st.slider(
        "Threshold (lebih rendah karena wajah berubah seiring waktu)",
        0.0, 1.0, 0.70, 0.01,
        key="threshold_waktu"
    )

    if foto_lama and foto_baru:
        col1.image(foto_lama, caption="Masa kecil", use_container_width=True)
        col2.image(foto_baru, caption="Sekarang", use_container_width=True)

        if st.button("Analisis Sekarang"):
            with st.spinner("Sedang menganalisis..."):
                (score, label_lama, skor_lama,
                 label_baru, skor_baru,
                 det_lama, det_baru, orang_sama) = compare_then_vs_now(
                    foto_lama.read(), foto_baru.read(),
                    pca, X_pca, labels,
                    threshold_waktu
                )

            col1.caption("✅ Wajah terdeteksi" if det_lama else "⚠️ Wajah tidak terdeteksi otomatis")
            col2.caption("✅ Wajah terdeteksi" if det_baru else "⚠️ Wajah tidak terdeteksi otomatis")

            st.divider()

            if orang_sama:
                st.success("✅ Kemungkinan besar ini adalah **orang yang sama!**")
            else:
                st.error("❌ Kemungkinan ini adalah **orang yang berbeda.**")

            st.metric("Skor kemiripan langsung (masa kecil vs sekarang)", f"{score:.4f}")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Foto masa kecil paling mirip dengan:**")
                if skor_lama >= threshold_waktu:
                    st.success(f"**{label_lama}** (skor: `{skor_lama:.4f}`)")
                else:
                    st.warning(f"Tidak dikenal (skor tertinggi: `{skor_lama:.4f}`)")
            with col2:
                st.markdown("**Foto sekarang paling mirip dengan:**")
                if skor_baru >= threshold_waktu:
                    st.success(f"**{label_baru}** (skor: `{skor_baru:.4f}`)")
                else:
                    st.warning(f"Tidak dikenal (skor tertinggi: `{skor_baru:.4f}`)")

            st.divider()
            with st.expander("ℹ️ Bagaimana sistem memutuskan?"):
                cond1 = score >= threshold_waktu
                cond2 = (
                    label_lama == label_baru and
                    skor_lama >= threshold_waktu and
                    skor_baru >= threshold_waktu
                )
                st.write("Sistem menganggap **orang yang sama** apabila salah satu kondisi ini terpenuhi:")
                st.write(
                    f"1. Skor langsung antar dua foto ≥ threshold: "
                    f"{'✅' if cond1 else '❌'} ({score:.4f} vs {threshold_waktu})"
                )
                st.write(
                    f"2. Keduanya dikenali sebagai orang yang sama di database: "
                    f"{'✅' if cond2 else '❌'} ({label_lama} vs {label_baru})"
                )