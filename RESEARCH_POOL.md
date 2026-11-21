# CryoMorph: Scientific Literature & Dataset Research Pool (Kaynak Tarama Havuzu)

Bu doküman, Cryo-EM sürekli konformasyonel heterojenite alanındaki en güncel akademik makaleleri, halka açık biyofiziksel veri setlerini ve programatik API uç noktalarını içermektedir.

---

## 1. Halka Açık Cryo-EM Veri Tabanları ve API Uç Noktaları

| Veri Tabanı | Kapsam | API / Erişim Protokolü | Python Kütüphanesi |
| :--- | :--- | :--- | :--- |
| **EMPIAR** (EM Public Image Archive) | Ham 2D Micrograph'lar, partikül yığınları (`.mrcs`, `.star`) | `https://www.ebi.ac.uk/empiar/api/entry/{EMPIAR_ID}` | `requests`, `urllib` |
| **EMDB** (Electron Microscopy Data Bank) | 3D Rekonstrükte edilmiş yoğunluk haritaları (`.map`, `.mrc`) | `https://www.ebi.ac.uk/emdb/api/entry/{EMD_ID}` | `mrcfile` |
| **RCSB PDB** (Protein Data Bank) | Atomik koordinat modelleri (`.cif`, `.pdb`) | `https://data.rcsb.org/rest/v1/core/entry/{PDB_ID}` | `biopython` |

---

## 2. Altın Standart Benchmark Veri Setleri (EMPIAR)

CryoMorph mimarisini test etmek için kullanılacak birincil açık veri setleri:

### 1. EMPIAR-10076: *Plasmodium falciparum* 80S Ribozomu
* **Önemi:** Cryo-EM sürekli sınıflandırma literatüründeki 1 numaralı kıyaslama veri setidir.
* **Biyofiziksel Özellik:** Küçük alt birimin (40S) büyük alt birime (60S) göre dönme ve yuvarlanma (ratcheting/rolling) hareketi.
* **Partikül Sayısı:** ~105.000 partikül, kutu boyutu 360x360 piksel.
* **İlgili EMDB / PDB:** EMD-2660 / PDB 3J7A.

### 2. EMPIAR-10180: Pre-Katalitik Spliceosome (Spliceozom Kompleksi)
* **Önemi:** Biyolojideki en karmaşık esnek moleküler makinelerden biridir; aşırı derecede non-lineer esnekliğe sahiptir.
* **Biyofiziksel Özellik:** SF3b alt kompleksinin ve helikaz domainlerinin geniş açılı asenkron salınımları.
* **Partikül Sayısı:** ~327.000 partikül.
* **İlgili EMDB / PDB:** EMD-3652 / PDB 5NRL.

### 3. EMPIAR-10644: SARS-CoV-2 Spike Proteini Prefüzyon Dinamikleri
* **Önemi:** Virüsün reseptör bağlama domaininin (RBD) "Up" ve "Down" konformasyonları arasındaki sürekli geçişi.
* **Biyofiziksel Özellik:** Hem bileşimsel (subunit kaybolması) hem de konformasyonel (açısal esneme) heterojenite.

---

## 3. Temel Akademik Literatür Matrisi

Aşağıdaki makaleler, CryoMorph'un 3 sütununu (VAE, Flow Field, Gaussian Splatting) tasarlarken referans alınacak kilometre taşlarıdır:

| Başlık | Yazarlar & Dergi | Yıl | Mimari Katkı / Çözülen Problem |
| :--- | :--- | :--- | :--- |
| **"CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks"** | Zhong et al., *Nature Methods* | 2021 | Koordinat tabanlı Fourier VAE ile continuous latent manifold çıkarımı. |
| **"3D flexible refinement: Structure determination of continuously flexible macromolecules"** | Punjani & Fleet, *Cell Systems* | 2023 | Consensus hacim üzerinde elastik regülerize vektör yer değiştirme akış alanları (Flow fields). |
| **"RELION-3: new tools for automated high-resolution cryo-EM structure determination"** | Zivanov et al., *eLife* | 2018 | Rijit çoklu gövde (Multi-body refinement) yaklaşımı ve PCA analizi. |
| **"Differentiable Gaussian Splatting for Cryo-EM"** | Emerging Preprints (*bioRxiv / arXiv*) | 2024-2026 | Anizotropik 3D Gaussians ile analitik CTF konvolüsyonu ve milisaniyelik ileri projeksiyon. |
| **"Validation of continuous heterogeneity in cryo-EM"** | Sorzano et al., *Acta Cryst D* | 2023 | Sürekli hareket manifoldlarının Overfitting ve FSC doğrulama metrikleri. |

---

## 4. Akademik Arama Sorgu Bankası (Literature Queries)

Yeni makaleleri düzenli olarak taramak için bioRxiv, arXiv (q-bio) ve Europe PMC üzerinde çalıştırılacak optimize sorgular:

```text
Query 1 (Anahtar Teknoloji):
("cryo-EM" OR "cryo-electron microscopy") AND ("continuous heterogeneity" OR "conformational landscape" OR "flexible refinement") AND ("neural" OR "manifold" OR "flow")

Query 2 (Gaussian Splatting & Dif. Rendering):
("cryo-EM") AND ("Gaussian splatting" OR "differentiable rendering" OR "implicit neural representation")

Query 3 (Doğrulama & Metrikler):
("cryo-EM") AND ("continuous heterogeneity") AND ("validation" OR "half-map FSC" OR "free energy landscape")
```

---

## 5. Zenginleştirilmiş Programlama Dilleri & Rolleri

Rust ve Julia'nın yanına projeyi güçlendirecek ileri seviye diller ve kullanım senaryoları:

### 1. C++20 & CUDA (NVIDIA GPU Kernel Katmanı)
* **Rol:** Standart PyTorch operasyonları, Cryo-EM'e özgü Non-Uniform Fast Fourier Transform (NUFFT) ve 3D Fourier dilim projeksiyonunda gereksiz bellek kopyalamaları yaratır.
* **Katkı:** Özel yazılmış CUDA çekirdekleri (CUDA kernels) ile parçacık projeksiyonlarını saniyede 10.000 partikül hızına çıkarır.

### 2. Triton (OpenAI Triton / Python GPU DSL)
* **Rol:** C++ ve CUDA yazmanın getirdiği derleme ve bellek yönetimi yükü olmadan, doğrudan Python içerisinden donanım düzeyinde GPU kodları üretmek.
* **Katkı:** Anizotropik Gauss saçılımlarının CTF ile frekans uzayında analitik çarpımını GPU SRAM (Shared Memory) üzerinde tek bir fused kernel içinde çözer.

### 3. WGSL (WebGPU Shading Language) & TypeScript
* **Rol:** Araştırmacıların ve biyologların, sunucuya bağımlı olmadan kişisel bilgisayarlarının tarayıcısında (Next.js) proteinin sürekli hareketini ve serbest enerji vadilerini 60 FPS ile döndürebilmesi.
* **Katkı:** 3D Gaussian Splatting veya yoğunluk izo-yüzeylerini (marching cubes) doğrudan GPU donanım hızlandırmasıyla istemci tarafında render eder.

### 4. Zig (Düşük Seviyeli Sistem & SIMD)
* **Rol:** C kütüphaneleriyle sıfır maliyetli FFI, inanılmaz hızlı derleme ve libc bağımlılığı olmadan cross-compilation.
* **Katkı:** MRC dosya başlıklarının ve STAR meta-tablolarının SIMD destekli mikro-ayrıştırıcısı olarak Rust'a alternatif veya destekçi modül.

---

## 6. Otomatik Literatür Takip Stratejisi

Antigravity ortamında veya yerel makinede belirli aralıklarla çalışan otomatik tarama mekanizması (Europe PMC, bioRxiv ve arXiv API'leri üzerinden) CryoMorph araştırma havuzunu güncel tutmaktadır.

---

## 7. Son Otomatik Tarama ile Eklenen Güncel Makaleler ve Ön-Baskılar (Kasım 2026)

| Makale Başlığı | Yazarlar / Kaynak | Yıl | DOI / Bağlantı | CryoMorph İlişkisi & Katkısı |
| :--- | :--- | :--- | :--- | :--- |
| **CryoGaussian: Heterogeneous Cryo-EM Reconstruction Made Explicit with Gaussian Splatting** | Ziwen Wang et al. (*arXiv*) | 2026 | [arXiv:2611.26128](http://arxiv.org/abs/2611.26128v1) | **Pillar C (Gaussian Splatting):** Nadir ve sürekli konformasyonel değişimleri açık (explicit) 3D Gaussians ile çözerek CryoMorph'un Gaussian rasterizer yaklaşımını doğrudan doğrulamaktadır. |
| **Empirical Bayes Flow Matching for Continuous Cryo-EM Heterogeneity** | Daniel Aibinder, Azmi Haider, Dan Rosenbaum (*arXiv*) | 2026 | [arXiv:2611.05202](http://arxiv.org/abs/2611.05202v1) | **Pillar B (Continuous Deformation & Flow):** Dolaylı 2D projeksiyonlardan yüksek boyutlu sürekli heterojenite manifoldunu Empirical Bayes tabanlı Flow Matching ile öğrenir. |
| **GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction** | Huaizhi Qu, Xiao Wang, Gengwei Zhang et al. (*arXiv*) | 2025/2026 | [arXiv:2509.25075](http://arxiv.org/abs/2509.25075v2) | **Pillar C:** 3D Gaussian Splatting'in Cryo-EM rekonstrüksiyonunda yüksek hız ve analitik CTF entegrasyonundaki başarısını gösterir. |
| **SOLVAR: Fast covariance-based heterogeneity analysis with pose refinement for cryo-EM** | Roey Yadgar, Roy R. Lederman, Yoel Shkolnisky (*arXiv*) | 2026 | [arXiv:2602.17603](http://arxiv.org/abs/2602.17603v1) | **Kovaryans & Poz İyileştirme:** Sürekli durumlar arasında varyans analizi ve açısal poz belirsizliğini birlikte optimize eder. |
| **CryoTriplane: triplane-based neural radiance field representation for high-resolution cryo-EM 3D homogeneous reconstruction** | Y. Cheng, B. He, Q. Zeng et al. (*Bioinformatics*) | 2026 | [10.1093/bioinformatics/btag754](https://doi.org/10.1093/bioinformatics/btag754) | **Koordinat Ağı Hızlandırma:** NeRF tri-plane yaklaşımı ile hesaplama yükünü düşürerek Fourier uzayında hızlı koordinat sorgusu sağlar. |
| **InstaMap: instant-NGP for cryo-EM density maps** | G. Woollard et al. (*Acta Cryst D*) | 2025 | [10.1107/s2059798325002025](https://doi.org/10.1107/s2059798325002025) | **Hızlı Hash Grid:** Instant-NGP çok çözünürlüklü hash tablolarıyla Cryo-EM hacim temsilini hızlandırır. |

