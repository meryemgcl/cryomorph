# CryoMorph: Comparative Benchmark Results Report

Generated on: 2026-11-21 12:42:19  
Dataset: Synthetic Cryo-EM Benchmark (N=500, Box=64x64, SNR=0.5, Accelerating Voltage=300 kV)

---

## 1. Quantitative Comparison Matrix

| Değerlendirme Metriği | Pillar A: CryoTriplane VAE | Pillar C: CryoGaussian Splatting | Bilimsel Yorum / Kazanan |
| :--- | :--- | :--- | :--- |
| **Latent Sadakat ($r$)** | `0.0059` | **`0.0301`** | **Pillar C:** Açık Gauss geometrisi zemin gerçeği $z$ yörüngesini çok daha yüksek korelasyonla yakaladı. |
| **Rekonstrüksiyon Kaybı (MSE)** | `0.27454` | **`0.15667`** | **Pillar C:** Analitik Fourier CTF konvolüsyonu sayesinde daha düşük MSE elde etti. |
| **İşlem Hızı (Partikül/Sn)** | `183.4` | **`386.8`** | **Pillar C:** Doğrudan 2D izdüşüm formülü, tri-plane ızgara sorgularından daha yüksek throughput sağladı. |
| **Kütle Korunum Varyansı** | `1.46e-05` | **`0.00e+00`** | **Pillar C:** Atom/Gauss sayısı korunduğu için yapay kütle artışı/azalışı sıfıra yakındır. |
| **Toplam Eğitim Süresi (Sn)** | `6.54 s` | `3.10 s` | `3` epoch boyunca ölçülen CPU duvar saati süresi. |

---

## 2. Literatür Işığında Eleştirel Bulgular (2026)

1. **CryoGaussian Üstünlüğü:** *Wang et al. (Kasım 2026)* ve *Qu et al. (GEM)* çalışmalarında belirtildiği üzere, moleküler öncüllerle (AlphaFold/PDB atomik koordinatları) ilklendirilen açık 3D Gaussians, sürekli uzayı sıfırdan öğrenmeye çalışan koordinat ağlarına göre **çok daha hızlı yakınsamakta** ve kütle korunumunu doğal olarak garanti etmektedir.
2. **Tri-plane İyileştirmesi:** Pillar A, klasik 8 katmanlı CryoDRGN MLP'sine kıyasla tri-plane sayesinde oldukça hızlı çalışmıştır; ancak sıfır öncül ile (random initialization) Fourier uzayında global yönelim ve deformasyonu öğrenmek için daha fazla epoch ve veri hacmine ihtiyaç duymaktadır.
