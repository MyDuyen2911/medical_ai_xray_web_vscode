const API_URL = "http://127.0.0.1:5000/predict";

const STATUS_COLORS = {
  success: "#1e9e6d",
  warning: "#c2790a",
  danger: "#d1414d",
  neutral: "#1456c7"
};

const GAUGE_CIRCUMFERENCE = 339.3;

const navLinks = document.querySelectorAll(".nav-link");
const pages = document.querySelectorAll(".page");

const imageInput = document.getElementById("imageInput");
const uploadBox = document.getElementById("uploadBox");
const previewImage = document.getElementById("previewImage");
const previewFrame = document.getElementById("previewFrame");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");

const resultEmptyState = document.getElementById("resultEmptyState");
const resultContent = document.getElementById("resultContent");
const resultImageBox = document.getElementById("resultImageBox");
const predictedClass = document.getElementById("predictedClass");
const confidenceScore = document.getElementById("confidenceScore");
const resultStatusBadge = document.getElementById("resultStatusBadge");
const probabilityList = document.getElementById("probabilityList");
const resultCardMain = document.querySelector(".result-card-main");
const gaugeCircle = document.getElementById("gaugeCircle");
const viewXaiBtn = document.getElementById("viewXaiBtn");

const xaiEmptyState = document.getElementById("xaiEmptyState");
const xaiContent = document.getElementById("xaiContent");
const xaiPredictedClass = document.getElementById("xaiPredictedClass");
const xaiConfidence = document.getElementById("xaiConfidence");
const xaiInputPreview = document.getElementById("xaiInputPreview");
const xaiComparisonImage = document.getElementById("xaiComparisonImage");
const xaiStripCard = document.getElementById("xaiStripCard");
const gradcamImage = document.getElementById("gradcamImage");
const gradcamppImage = document.getElementById("gradcamppImage");
const igImage = document.getElementById("igImage");
const occlusionImage = document.getElementById("occlusionImage");

const historyTableBody = document.getElementById("historyTableBody");
const historyEmpty = document.getElementById("historyEmpty");
const searchInput = document.getElementById("historySearchInput");
const filterButtons = document.querySelectorAll(".filter");
const toastContainer = document.getElementById("toastContainer");

let uploadedImageData = null;
let uploadedFile = null;
let currentFilter = "Tất cả";
let currentAnalysis = null;
let historyData = [];

function showToast(message, type = "neutral") {
  if (!toastContainer) {
    alert(message);
    return;
  }

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(8px)";
    setTimeout(() => toast.remove(), 200);
  }, 3200);
}

function showPage(pageId) {
  pages.forEach(page => page.classList.remove("active-page"));

  const targetPage = document.getElementById(pageId);
  if (targetPage) {
    targetPage.classList.add("active-page");
  }

  navLinks.forEach(link => {
    link.classList.toggle("active", link.dataset.page === pageId);
  });

  window.scrollTo({ top: 0, behavior: "smooth" });
}

navLinks.forEach(link => {
  link.addEventListener("click", () => showPage(link.dataset.page));
});

document.querySelectorAll("[data-goto]").forEach(el => {
  el.addEventListener("click", () => showPage(el.dataset.goto));
});

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = event => resolve(event.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function handleSelectedFile(file) {
  if (!file) return;

  if (!file.type || !file.type.startsWith("image/")) {
    showToast("Vui lòng chọn một tệp ảnh hợp lệ JPG, PNG hoặc JPEG.", "warning");
    return;
  }

  uploadedFile = file;
  uploadedImageData = await readFileAsDataURL(file);

  fileName.textContent = file.name;
  previewImage.src = uploadedImageData;
  previewImage.style.display = "block";

  const placeholder = previewFrame?.querySelector(".frame-placeholder");
  if (placeholder) {
    placeholder.style.display = "none";
  }
}

imageInput?.addEventListener("change", function () {
  handleSelectedFile(this.files[0]);
});

["dragenter", "dragover"].forEach(eventName => {
  uploadBox?.addEventListener(eventName, event => {
    event.preventDefault();
    uploadBox.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach(eventName => {
  uploadBox?.addEventListener(eventName, event => {
    event.preventDefault();
    uploadBox.classList.remove("drag-active");
  });
});

uploadBox?.addEventListener("drop", event => {
  const file = event.dataTransfer?.files?.[0];
  handleSelectedFile(file);
});

function setAnalyzeButtonLoading(isLoading) {
  if (!analyzeBtn) return;

  analyzeBtn.disabled = isLoading;
  analyzeBtn.textContent = isLoading ? "Đang phân tích..." : "Phân tích ảnh";
}

analyzeBtn?.addEventListener("click", async () => {
  if (!uploadedFile || !uploadedImageData) {
    showToast("Vui lòng chọn ảnh X-quang trước khi phân tích.", "warning");
    return;
  }

  setAnalyzeButtonLoading(true);

  try {
    const result = await predictWithBackend(uploadedFile);
    const normalized = normalizeBackendResult(result);

    currentAnalysis = normalized;
    updateResultPage(normalized);
    updateXaiPage(normalized);
    addHistory(normalized);

    showPage("result");
    showToast("Phân tích ảnh thành công.", "success");
  } catch (error) {
    console.error(error);
    showToast(`Không phân tích được ảnh: ${error.message}`, "danger");
  } finally {
    setAnalyzeButtonLoading(false);
  }
});

async function predictWithBackend(file) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("file", file);

  const response = await fetch(API_URL, {
    method: "POST",
    body: formData
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.error || "Backend không phản hồi thành công.");
  }

  return data;
}

function normalizeBackendResult(data) {
  const rawPrediction = data.prediction || data.class || data.label || "---";
  const prediction = formatLabel(rawPrediction);

  let confidence = Number(data.confidence ?? data.probability ?? data.score ?? 0);
  if (confidence <= 1) {
    confidence *= 100;
  }

  const probabilities = normalizeProbabilities(data.probabilities || data.probs || {});

  const rawXai = data.xai_images || data.heatmaps || {};
  console.log("Backend response:", data);
  console.log("XAI images:", data.xai_images);
  const xaiImages = {
    gradcam: rawXai.gradcam || rawXai["grad_cam"] || rawXai["Grad-CAM"] || "",
    gradcampp: rawXai.gradcampp || rawXai.gradcam_plus || rawXai["grad_cam_plus"] || rawXai["Grad-CAM++"] || "",
    integratedGradients: rawXai.integrated_gradients || rawXai.integratedGradients || rawXai.ig || "",
    occlusion: rawXai.occlusion || rawXai.occlusion_sensitivity || ""
  };

  const originalImage = data.original_image || data.input_image || uploadedImageData || "";
  const comparisonImage = data.xai_comparison || data.xai_comparison_image || data.comparison_image || "";

  return {
    id: Date.now(),
    fileName: uploadedFile?.name || "xray-image",
    date: new Date().toLocaleDateString("vi-VN"),
    model: data.model || "ResNet101",
    prediction,
    confidence,
    probabilities,
    status: getStatusText(prediction),
    statusClass: getStatusClass(prediction),
    originalImage,
    comparisonImage,
    xaiImages
  };
}

function normalizeProbabilities(probabilities) {
  const normalized = {
    "COVID-19": 0,
    Normal: 0,
    Pneumonia: 0,
    Tuberculosis: 0
  };

  Object.entries(probabilities).forEach(([key, value]) => {
    let label = formatLabel(key);
    let numberValue = Number(value);

    if (numberValue <= 1) {
      numberValue *= 100;
    }

    if (normalized[label] !== undefined) {
      normalized[label] = numberValue;
    }
  });

  return normalized;
}

function formatLabel(label) {
  const upperLabel = String(label || "").toUpperCase();

  if (upperLabel === "COVID19" || upperLabel === "COVID-19") return "COVID-19";
  if (upperLabel === "NORMAL") return "Normal";
  if (upperLabel === "PNEUMONIA") return "Pneumonia";
  if (upperLabel === "TUBERCULOSIS") return "Tuberculosis";

  return label || "---";
}

function getStatusText(label) {
  if (label === "Normal") return "Bình thường";
  if (label === "Pneumonia" || label === "COVID-19") return "Cần theo dõi";
  if (label === "Tuberculosis") return "Cần kiểm tra thêm";

  return "---";
}

function getStatusClass(label) {
  if (label === "Normal") return "success";
  if (label === "Pneumonia" || label === "COVID-19") return "warning";
  if (label === "Tuberculosis") return "danger";

  return "neutral";
}

function updateResultPage(result) {
  resultEmptyState?.classList.add("hidden");
  resultContent?.classList.remove("hidden");

  predictedClass.textContent = result.prediction;
  confidenceScore.textContent = `${result.confidence.toFixed(1)}%`;

  resultStatusBadge.textContent = `Trạng thái: ${result.status}`;
  resultStatusBadge.className = `badge ${result.statusClass}`;

  if (resultCardMain) {
    resultCardMain.style.setProperty(
      "--status-color",
      STATUS_COLORS[result.statusClass] || STATUS_COLORS.neutral
    );
  }

  updateGauge(result.confidence);

  resultImageBox.innerHTML = `
    <img src="${result.originalImage}" alt="Ảnh X-quang đầu vào">
  `;

  probabilityList.innerHTML = Object.entries(result.probabilities)
    .map(([probLabel, value]) => `
      <div class="prob-row">
        <span>${probLabel}</span>
        <div class="progress"><b style="width:${value}%"></b></div>
        <strong>${value.toFixed(1)}%</strong>
      </div>
    `)
    .join("");
}

function updateGauge(confidence) {
  if (!gaugeCircle) return;

  const clamped = Math.max(0, Math.min(Number(confidence) || 0, 100));
  const offset = GAUGE_CIRCUMFERENCE * (1 - clamped / 100);
  gaugeCircle.style.strokeDashoffset = String(offset);
}

function updateXaiPage(result) {
  xaiEmptyState?.classList.add("hidden");
  xaiContent?.classList.remove("hidden");

  xaiPredictedClass.textContent = result.prediction;
  xaiConfidence.textContent = `${result.confidence.toFixed(1)}%`;

  xaiInputPreview.innerHTML = `
    <img src="${result.originalImage}" alt="Ảnh X-quang đang được giải thích">
  `;

  if (result.comparisonImage) {
    xaiStripCard?.classList.remove("hidden");
    xaiComparisonImage.src = result.comparisonImage;
  } else {
    xaiStripCard?.classList.add("hidden");
  }

  setXaiImage(gradcamImage, result.xaiImages.gradcam);
  setXaiImage(gradcamppImage, result.xaiImages.gradcampp);
  setXaiImage(igImage, result.xaiImages.integratedGradients);
  setXaiImage(occlusionImage, result.xaiImages.occlusion);
}

function setXaiImage(imgElement, src) {
  if (!imgElement) return;

  const card = imgElement.closest(".xai-card");

  if (src) {
    imgElement.src = src;
    card?.classList.remove("unavailable");
  } else {
    imgElement.removeAttribute("src");
    card?.classList.add("unavailable");
  }
}

viewXaiBtn?.addEventListener("click", () => {
  if (!currentAnalysis) {
    showToast("Vui lòng phân tích ảnh trước khi xem giải thích XAI.", "warning");
    return;
  }

  updateXaiPage(currentAnalysis);
  showPage("xai");
});

function addHistory(result) {
  historyData.unshift(result);
  renderHistory();
}

function renderHistory() {
  if (!historyTableBody) return;

  historyEmpty?.classList.toggle("hidden", historyData.length > 0);

  const keyword = searchInput ? searchInput.value.toLowerCase().trim() : "";

  const filtered = historyData.filter(item => {
    const matchFilter = currentFilter === "Tất cả" || item.prediction === currentFilter;
    const matchKeyword = [
      item.fileName,
      item.date,
      item.model,
      item.prediction,
      item.status
    ].join(" ").toLowerCase().includes(keyword);

    return matchFilter && matchKeyword;
  });

  historyTableBody.innerHTML = filtered.map((item, index) => `
    <tr>
      <td>${String(index + 1).padStart(2, "0")}</td>
      <td><img class="history-thumb" src="${item.originalImage}" alt="Ảnh lịch sử"></td>
      <td>${item.date}</td>
      <td>${item.model}</td>
      <td>${item.prediction}</td>
      <td class="mono">${item.confidence.toFixed(1)}%</td>
      <td><span class="badge ${item.statusClass}">${item.status}</span></td>
      <td><button class="table-btn history-view-btn" data-id="${item.id}">Xem</button></td>
    </tr>
  `).join("");

  document.querySelectorAll(".history-view-btn").forEach(button => {
    button.addEventListener("click", () => {
      const selected = historyData.find(item => String(item.id) === button.dataset.id);

      if (!selected) return;

      currentAnalysis = selected;
      updateResultPage(selected);
      updateXaiPage(selected);
      showPage("result");
    });
  });
}

filterButtons.forEach(button => {
  button.addEventListener("click", () => {
    filterButtons.forEach(btn => btn.classList.remove("active"));
    button.classList.add("active");

    currentFilter = button.textContent.trim();
    renderHistory();
  });
});

searchInput?.addEventListener("input", renderHistory);
renderHistory();
