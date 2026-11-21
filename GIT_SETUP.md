# CryoMorph: GitHub Kurulum & Antigravity Entegrasyon Rehberi

Bu rehber, projenin yerel Git deposunu oluşturmayı, GitHub hesabınızda (`meryemgcl`) yayınlamayı ve Antigravity ortamına bağlamayı adım adım açıklar.

---

## 1. Yerel Git Deposunu Başlatma & İlk Commit

PowerShell veya Komut İstemcisinde aşağıdaki komutları sırasıyla çalıştırın:

```powershell
# 1. Proje dizinine geçin
cd C:\Users\HP\Desktop\projeler\cryomorph

# 2. Git deposunu başlatın
git init

# 3. Tüm dosyaları sahneye ekleyin
git add .

# 4. İlk mimarî commit'i atın
git commit -m "feat: initial commit with polyglot architecture, 3-pillar benchmarks, and research pool"

# 5. Ana dalı main olarak ayarlayın
git branch -M main
```

---

## 2. GitHub Üzerinde Yeni Repo Açma & Bağlama

1. Tarayıcınızda [https://github.com/new](https://github.com/new) adresine gidin.
2. **Repository name** alanına: `cryomorph` yazın.
3. Açıklama (Description) kısmına:
   `Continuous Macromolecular Conformational Heterogeneity Engine for Cryo-EM (Rust, Julia, PyTorch, WebGPU)`
4. **Public** seçin. *"Initialize this repository with a README"* kutucuğunu **boş bırakın** (çünkü yerelde zaten hazırladık).
5. **Create repository** butonuna tıklayın.
6. Açılan sayfadaki uzak repo adresini yerel deponuza bağlayıp push edin:

```powershell
git remote add origin https://github.com/meryemgcl/cryomorph.git
git push -u origin main
```

---

## 3. Antigravity Üzerinde Projeye Bağlanma

Antigravity IDE'sinde bu projeyi doğrudan açmak veya workspaces listesine eklemek için:
* Antigravity sol üst menüsünden **File > Open Folder...** seçeneğine tıklayın.
* `C:\Users\HP\Desktop\projeler\cryomorph` klasörünü seçin.
* Tüm dokümantasyon (`README.md`, `ARCHITECTURE.md`, `RESEARCH_POOL.md`) ve `notebooks/` dizini doğrudan Antigravity dosya gezgininde yer alacaktır.

---

## 4. Sürekli Güncel Literatür & Haber Taraması

Bu alandaki yeni makaleler, model yaklaşımları (özellikle bioRxiv ve Nature Methods) çıktıkça:
* `notebooks/cryomorph_literature_and_data_explorer.ipynb` dosyasındaki 2. hücreyi çalıştırarak yeni yayınları doğrudan DataFrame olarak çekebilirsiniz.
* Antigravity içindeyken *"Cryo-EM continuous heterogeneity alanında bu ay çıkan yeni makaleleri tara ve RESEARCH_POOL.md dosyasına ekle"* komutunu vermeniz yeterlidir.
