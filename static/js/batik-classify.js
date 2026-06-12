document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const uploadForm = document.getElementById('uploadForm');
    const spinner = document.getElementById('spinner');
    const errorBox = document.getElementById('errorBox');
    const resultSection = document.getElementById('resultSection');

    let selectedFile = null;

    function setSelectedFile(file) {
        if (!file) return;
        const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/jpg'];
        if (!allowed.includes(file.type)) {
            showError('Format file tidak didukung. Gunakan JPG, JPEG, PNG, atau BMP.');
            return;
        }
        selectedFile = file;
        analyzeBtn.disabled = false;
        hideError();

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadArea.innerHTML = `
                <div class="preview-wrap">
                    <img src="${e.target.result}" alt="Preview">
                    <p>${file.name}</p>
                </div>
            `;
        };
        reader.readAsDataURL(file);
    }

    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(evt => {
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
    });

    uploadArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files[0]) {
            fileInput.files = files;
            setSelectedFile(files[0]);
        }
    });

    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        analyzeBtn.disabled = true;
        hideError();
        resultSection.classList.add('hidden');
        resetUploadArea();
    });

    function resetUploadArea() {
        uploadArea.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p><strong>Klik untuk memilih gambar</strong> atau seret &amp; lepas di sini</p>
            <div class="upload-hint">Format: JPG, JPEG, PNG, BMP &middot; Maks. 8MB</div>
            <input type="file" id="fileInput" name="image" accept=".jpg,.jpeg,.png,.bmp">
        `;
        // re-bind new fileInput
        const newInput = document.getElementById('fileInput');
        newInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                setSelectedFile(e.target.files[0]);
            }
        });
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.classList.remove('hidden');
    }

    function hideError() {
        errorBox.classList.add('hidden');
        errorBox.textContent = '';
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        hideError();
        resultSection.classList.add('hidden');
        spinner.classList.remove('hidden');
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            if (!res.ok || data.error) {
                throw new Error(data.error || 'Terjadi kesalahan saat memproses gambar.');
            }

            renderResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            spinner.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    function renderResult(data) {
        document.getElementById('predLabel').textContent = data.prediction.pretty.replace(/_/g, ' ');
        document.getElementById('predConfidence').textContent =
            `Tingkat keyakinan: ${data.prediction.confidence}%`;

        document.getElementById('imgOriginal').src = data.uploaded_image;
        document.getElementById('imgGray').src = data.gray_image;
        document.getElementById('imgQuant').src = data.quant_image;

        const top5List = document.getElementById('top5List');
        top5List.innerHTML = '';
        data.top5.forEach((item, idx) => {
            const div = document.createElement('div');
            div.className = 'top5-item';
            div.innerHTML = `
                <span>${idx + 1}. ${item.pretty.replace(/_/g, ' ')}</span>
                <div class="top5-bar-wrap">
                    <div class="top5-bar" style="width: ${item.prob}%"></div>
                </div>
                <span class="top5-pct">${item.prob}%</span>
            `;
            top5List.appendChild(div);
        });

        const tbody = document.getElementById('glcmTableBody');
        tbody.innerHTML = '';
        data.glcm_features.forEach(feat => {
            const tr = document.createElement('tr');
            const angles = data.angle_labels.map(a =>
                `<td class="numeric">${feat.per_angle[a].toFixed(4)}</td>`
            ).join('');
            tr.innerHTML = `
                <td class="feature-name">${feat.name}</td>
                <td class="numeric">${feat.mean.toFixed(4)}</td>
                ${angles}
            `;
            tbody.appendChild(tr);
        });

        resultSection.classList.remove('hidden');
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
