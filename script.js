const API_URL = "http://127.0.0.1:5000/predict";

const viewXaiBtn = document.getElementById("viewXaiBtn");
const xaiPredictedClass = document.getElementById("xaiPredictedClass");
const xaiConfidence = document.getElementById("xaiConfidence");
const xaiInputPreview = document.getElementById("xaiInputPreview");
const xaiEmptyBanner = document.getElementById("xaiEmptyBanner");

let lastPredictionResult = null;

const navLinks = document.querySelectorAll(".nav-link");
const pages = document.querySelectorAll(".page");

const imageInput = document.getElementById("imageInput");
const uploadBox = document.getElementById("uploadBox");
const previewImage = document.getElementById("previewImage");
const previewFrame = document.getElementById("previewFrame");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultImageBox = document.getElementById("resultImageBox");

const predictedClass = document.getElementById("predictedClass");
const confidenceScore = document.getElementById("confidenceScore");
const resultStatusBadge = document.getElementById("resultStatusBadge");
const probabilityList = document.getElementById("probabilityList");
const resultEmptyState = document.getElementById("resultEmptyState");
const resultContent = document.getElementById("resultContent");
const resultCardMain = document.querySelector(".result-card-main");
const gaugeCircle = document.getElementById("gaugeCircle");

const historyTableBody = document.getElementById("historyTableBody");
const searchInput = document.querySelector(".search-input");
const filterButtons = document.querySelectorAll(".filter");
const toastContainer = document.getElementById("toastContainer");

const GAUGE_CIRCUMFERENCE = 339.3;
const STATUS_COLORS = {
  success: "#1e9e6d",
  warning: "#c2790a",
  danger: "#d1414d",
  neutral: "#1456c7"
};

let uploadedImageData = null;
let uploadedFile = null;
let currentFilter = "Tất cả";

const demoResults = [
  {
    prediction: "Pneumonia",
    confidence: 94.2,
    probabilities: {
      "COVID-19": 3.1,
      Normal: 1.4,
      Pneumonia: 94.2,
      Tuberculosis: 1.3
    }
  },
  {
    prediction: "Normal",
    confidence: 96.1,
    probabilities: {
      "COVID-19": 1.2,
      Normal: 96.1,
      Pneumonia: 1.9,
      Tuberculosis: 0.8
    }
  },
  {
    prediction: "Tuberculosis",
    confidence: 89.7,
    probabilities: {
      "COVID-19": 2.0,
      Normal: 3.8,
      Pneumonia: 4.5,
      Tuberculosis: 89.7
    }
  },
  {
    prediction: "COVID-19",
    confidence: 91.4,
    probabilities: {
      "COVID-19": 91.4,
      Normal: 2.8,
      Pneumonia: 4.1,
      Tuberculosis: 1.7
    }
  }
];

function showToast(message, type = "neutral") {
  if (!toastContainer) {
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
  pages.forEach((page) => {
    page.classList.remove("active-page");
  });

  const targetPage = document.getElementById(pageId);

  if (targetPage) {
    targetPage.classList.add("active-page");
  }

  navLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.page === pageId);
  });
}

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    showPage(link.dataset.page);
  });
});

document.querySelectorAll("[data-goto]").forEach((el) => {
  el.addEventListener("click", () => {
    showPage(el.dataset.goto);
  });
});

function handleSelectedFile(file) {
  if (!file) {
    return;
  }

  if (!file.type || !file.type.startsWith("image/")) {
    showToast("Vui lòng chọn một tệp ảnh hợp lệ (JPG, PNG).", "warning");
    return;
  }

  uploadedFile = file;
  fileName.textContent = file.name;

  const reader = new FileReader();

  reader.onload = function (event) {
    uploadedImageData = event.target.result;

    previewImage.src = uploadedImageData;
    previewImage.style.display = "block";

    const placeholder = previewFrame.querySelector(".frame-placeholder");
    if (placeholder) {
      placeholder.style.display = "none";
    }

    resultImageBox.innerHTML = `
      <img
        src="${uploadedImageData}"
        alt="Uploaded X-ray"
        style="width:100%;height:100%;object-fit:contain;background:#111827;border-radius:18px;"
      >
    `;
  };

  reader.readAsDataURL(file);
}

imageInput?.addEventListener("change", function () {
  handleSelectedFile(this.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadBox?.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadBox.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadBox?.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadBox.classList.remove("drag-active");
  });
});

uploadBox?.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  handleSelectedFile(file);
});

analyzeBtn?.addEventListener("click", async () => {
  if (!uploadedFile || !uploadedImageData) {
    showToast("Vui lòng chọn ảnh X-quang trước khi phân tích.", "warning");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Đang phân tích...";

  try {
    const result = await predictWithBackend(uploadedFile);

    updateResult(result, "backend");
    addHistory(result);
    showPage("result");
  } catch (error) {
    console.warn("Không gọi được backend, chuyển sang demo:", error);

    const demoResult =
      demoResults[Math.floor(Math.random() * demoResults.length)];

    updateResult(demoResult, "demo");
    addHistory(demoResult);
    showPage("result");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Phân tích ảnh";
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

  if (!response.ok) {
    throw new Error("Backend không phản hồi thành công.");
  }

  const data = await response.json();

  return normalizeBackendResult(data);
}

function normalizeBackendResult(data) {
  let prediction = data.prediction || data.class || data.label || "---";

  let confidence =
    data.confidence ?? data.probability ?? data.score ?? 0;

  confidence = Number(confidence);

  if (confidence <= 1) {
    confidence = confidence * 100;
  }

  const probabilities = normalizeProbabilities(
    data.probabilities || data.probs || {}
  );

  const xaiImages = data.xai_images || data.heatmaps || null;

  return {
    prediction,
    confidence,
    probabilities,
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
    let label = key;
    let numberValue = Number(value);

    if (numberValue <= 1) {
      numberValue = numberValue * 100;
    }

    const upperKey = key.toUpperCase();

    if (upperKey === "COVID19" || upperKey === "COVID-19") {
      label = "COVID-19";
    }

    if (upperKey === "NORMAL") {
      label = "Normal";
    }

    if (upperKey === "PNEUMONIA") {
      label = "Pneumonia";
    }

    if (upperKey === "TUBERCULOSIS") {
      label = "Tuberculosis";
    }

    if (normalized[label] !== undefined) {
      normalized[label] = numberValue;
    }
  });

  return normalized;
}

function updateResult(result, source = "demo") {
  const label = formatLabel(result.prediction);
  const confidence = Number(result.confidence || 0);
  const probabilities = normalizeProbabilities(result.probabilities || {});
  const statusClass = getStatusClass(label);
  const status = getStatusText(label);

  predictedClass.textContent = label;
  confidenceScore.textContent = `${confidence.toFixed(1)}%`;

  if (source === "backend") {
    resultStatusBadge.textContent = `Trạng thái: ${status}`;
  } else {
    resultStatusBadge.textContent = `Trạng thái: ${status} (demo)`;
  }

  resultStatusBadge.className = `badge ${statusClass}`;

  if (gaugeCircle) {
    const clampedConfidence = Math.max(0, Math.min(confidence, 100));
    const offset = GAUGE_CIRCUMFERENCE * (1 - clampedConfidence / 100);
    gaugeCircle.style.strokeDashoffset = String(offset);
  }

  if (resultCardMain) {
    resultCardMain.style.setProperty(
      "--status-color",
      STATUS_COLORS[statusClass] || STATUS_COLORS.neutral
    );
  }

  probabilityList.innerHTML = Object.entries(probabilities)
    .map(([probLabel, value]) => {
      return `
        <div class="prob-row">
          <span>${probLabel}</span>
          <div class="progress">
            <b style="width: ${value}%"></b>
          </div>
          <strong>${value.toFixed(1)}%</strong>
        </div>
      `;
    })
    .join("");

  lastPredictionResult = {
    label,
    confidence,
    probabilities
  };

  resultEmptyState?.classList.add("hidden");
  resultContent?.classList.remove("hidden");

  updateXaiInfo(label, confidence, result.xaiImages);
}

function addHistory(result) {
  if (!historyTableBody) {
    return;
  }

  const label = formatLabel(result.prediction);
  const confidence = Number(result.confidence || 0);
  const status = getStatusText(label);

  const today = new Date().toLocaleDateString("vi-VN");

  const imageCell = uploadedImageData
    ? `<img src="${uploadedImageData}" alt="X-ray thumbnail" style="width:58px;height:58px;object-fit:cover;border-radius:12px;background:#111827;">`
    : `<div class="thumb-xray"><svg class="scan-illustration tiny"><use href="#scan-base"></use></svg></div>`;

  const row = document.createElement("tr");

  row.innerHTML = `
    <td>00</td>
    <td>${imageCell}</td>
    <td>${today}</td>
    <td>ResNet101</td>
    <td>${label}</td>
    <td class="mono">${confidence.toFixed(1)}%</td>
    <td>
      <span class="badge ${getStatusClass(label)}">${status}</span>
    </td>
    <td><button class="table-btn">Xem</button></td>
  `;

  historyTableBody.prepend(row);
  renumberHistory();
}

function renumberHistory() {
  if (!historyTableBody) {
    return;
  }

  const rows = historyTableBody.querySelectorAll("tr");
  const total = rows.length;

  rows.forEach((row, position) => {
    const sttCell = row.children[0];
    if (sttCell) {
      sttCell.textContent = String(total - position).padStart(2, "0");
    }
  });
}

function formatLabel(label) {
  const upperLabel = String(label || "").toUpperCase();

  if (upperLabel === "COVID19" || upperLabel === "COVID-19") {
    return "COVID-19";
  }

  if (upperLabel === "NORMAL") {
    return "Normal";
  }

  if (upperLabel === "PNEUMONIA") {
    return "Pneumonia";
  }

  if (upperLabel === "TUBERCULOSIS") {
    return "Tuberculosis";
  }

  return label || "---";
}

function getStatusText(label) {
  if (label === "Normal") {
    return "Bình thường";
  }

  if (label === "Pneumonia" || label === "COVID-19") {
    return "Cần theo dõi";
  }

  if (label === "Tuberculosis") {
    return "Cần kiểm tra thêm";
  }

  return "---";
}

function getStatusClass(label) {
  if (label === "Normal") {
    return "success";
  }

  if (label === "Pneumonia" || label === "COVID-19") {
    return "warning";
  }

  if (label === "Tuberculosis") {
    return "danger";
  }

  return "neutral";
}

filterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    filterButtons.forEach((btn) => btn.classList.remove("active"));
    button.classList.add("active");

    currentFilter = button.textContent.trim();
    filterHistory();
  });
});

searchInput?.addEventListener("input", () => {
  filterHistory();
});

function filterHistory() {
  if (!historyTableBody) {
    return;
  }

  const keyword = searchInput
    ? searchInput.value.toLowerCase().trim()
    : "";

  const rows = historyTableBody.querySelectorAll("tr");

  rows.forEach((row) => {
    const rowText = row.textContent.toLowerCase();
    const resultText = row.children[4]?.textContent.trim() || "";

    const matchKeyword = rowText.includes(keyword);
    const matchFilter =
      currentFilter === "Tất cả" || resultText === currentFilter;

    row.style.display = matchKeyword && matchFilter ? "" : "none";
  });
}

function updateXaiInfo(label, confidence, xaiImages) {
  if (xaiPredictedClass) {
    xaiPredictedClass.textContent = label;
  }

  if (xaiConfidence) {
    xaiConfidence.textContent = `${confidence.toFixed(1)}%`;
  }

  if (xaiInputPreview && uploadedImageData) {
    xaiInputPreview.innerHTML = `
      <img
        src="${uploadedImageData}"
        alt="XAI input image"
        style="width:100%;height:100%;object-fit:contain;background:#111827;border-radius:18px;"
      >
    `;
  }

  updateXaiHeatmaps(xaiImages);

  xaiEmptyBanner?.classList.add("hidden");
}

const heatmapConfig = [
  { id: "heatmapGradcam", key: "gradcam" },
  { id: "heatmapGradcamPlus", key: "gradcam_plus" },
  { id: "heatmapIntegrated", key: "integrated_gradients" },
  { id: "heatmapOcclusion", key: "occlusion" }
];

function updateXaiHeatmaps(xaiImages) {
  heatmapConfig.forEach(({ id, key }) => {
    const container = document.getElementById(id);
    if (!container) {
      return;
    }

    const photo = container.querySelector(".heatmap-photo");
    const icon = container.querySelector(".scan-illustration");
    const tag = container.querySelector(".illustrative-tag");
    const realImage = xaiImages && xaiImages[key];

    if (realImage) {
      // Backend trả về ảnh heatmap đã tính toán thật cho kỹ thuật này.
      if (photo) {
        photo.src = realImage;
        photo.style.display = "block";
      }
      icon?.style.setProperty("display", "none");
      tag?.classList.add("hidden");
    } else if (uploadedImageData) {
      // Chưa có heatmap thật từ backend: hiển thị ảnh X-quang vừa tải lên
      // làm nền, có gắn nhãn "Minh họa" để không gây hiểu nhầm đây là bản đồ
      // được tính toán thật.
      if (photo) {
        photo.src = uploadedImageData;
        photo.style.display = "block";
      }
      icon?.style.setProperty("display", "none");
      tag?.classList.remove("hidden");
    } else {
      // Chưa có ảnh nào được tải lên: giữ icon minh họa mặc định.
      if (photo) {
        photo.style.display = "none";
      }
      icon?.style.setProperty("display", "block");
      tag?.classList.add("hidden");
    }
  });
}

viewXaiBtn?.addEventListener("click", () => {
  if (!lastPredictionResult) {
    showToast("Vui lòng phân tích ảnh trước khi xem giải thích XAI.", "warning");
    return;
  }

  showPage("xai");
});