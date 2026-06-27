const API_BASE = "http://127.0.0.1:5000";
const API_URL  = `${API_BASE}/predict`;

const TOKEN_KEY    = "mediai_token";
const USERNAME_KEY = "mediai_username";

const STATUS_COLORS = { success:"#1e9e6d", warning:"#c2790a", danger:"#d1414d", neutral:"#1456c7" };
const GAUGE_CIRCUMFERENCE = 339.3;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const authScreen   = document.getElementById("authScreen");
const appShell     = document.getElementById("appShell");
const loginForm    = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const authTabs     = document.querySelectorAll(".auth-tab");

const logoutBtn         = document.getElementById("logoutBtn");
const currentUsernameEl = document.getElementById("currentUsername");
const historyUsernameLabel = document.getElementById("historyUsernameLabel");

const navLinks = document.querySelectorAll(".nav-link");
const pages    = document.querySelectorAll(".page");

const imageInput  = document.getElementById("imageInput");
const uploadBox   = document.getElementById("uploadBox");
const previewImage = document.getElementById("previewImage");
const previewFrame = document.getElementById("previewFrame");
const fileName    = document.getElementById("fileName");
const analyzeBtn  = document.getElementById("analyzeBtn");

const resultEmptyState  = document.getElementById("resultEmptyState");
const resultContent     = document.getElementById("resultContent");
const resultImageBox    = document.getElementById("resultImageBox");
const predictedClass    = document.getElementById("predictedClass");
const confidenceScore   = document.getElementById("confidenceScore");
const resultStatusBadge = document.getElementById("resultStatusBadge");
const probabilityList   = document.getElementById("probabilityList");
const resultCardMain    = document.querySelector(".result-card-main");
const gaugeCircle       = document.getElementById("gaugeCircle");
const viewXaiBtn        = document.getElementById("viewXaiBtn");
const exportReportBtn   = document.getElementById("exportReportBtn");

const xaiEmptyState     = document.getElementById("xaiEmptyState");
const xaiContent        = document.getElementById("xaiContent");
const xaiPredictedClass = document.getElementById("xaiPredictedClass");
const xaiConfidence     = document.getElementById("xaiConfidence");
const xaiInputPreview   = document.getElementById("xaiInputPreview");
const xaiComparisonImage = document.getElementById("xaiComparisonImage");
const xaiStripCard      = document.getElementById("xaiStripCard");
const gradcamImage      = document.getElementById("gradcamImage");
const gradcamppImage    = document.getElementById("gradcamppImage");
const igImage           = document.getElementById("igImage");
const occlusionImage    = document.getElementById("occlusionImage");

const historyTableBody = document.getElementById("historyTableBody");
const historyEmpty     = document.getElementById("historyEmpty");
const searchInput      = document.getElementById("historySearchInput");
const filterButtons    = document.querySelectorAll(".filter");
const toastContainer   = document.getElementById("toastContainer");

const profileBadge    = document.getElementById("profileBadge");
const profileUsername = document.getElementById("profileUsername");
const profileJoined   = document.getElementById("profileJoined");
const profileTotal    = document.getElementById("profileTotal");
const profileCovid    = document.getElementById("profileCovid");
const profileNormal   = document.getElementById("profileNormal");
const profilePneumonia = document.getElementById("profilePneumonia");
const profileTB       = document.getElementById("profileTB");
const changePwForm    = document.getElementById("changePwForm");

const printOverlay = document.getElementById("printOverlay");
const printReport  = document.getElementById("printReport");

// ── State ─────────────────────────────────────────────────────────────────────
let authToken      = localStorage.getItem(TOKEN_KEY) || null;
let currentUsername = localStorage.getItem(USERNAME_KEY) || null;
let uploadedImageData = null;
let uploadedFile   = null;
let currentFilter  = "Tất cả";
let currentAnalysis = null;
let historyData    = [];

// ── Helpers ───────────────────────────────────────────────────────────────────
function authHeaders() {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

function showToast(message, type = "neutral") {
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

function formatDate(isoString) {
  const d = new Date(isoString);
  if (isNaN(d)) return isoString;
  return `${d.toLocaleDateString("vi-VN")} ${d.toLocaleTimeString("vi-VN", { hour:"2-digit", minute:"2-digit" })}`;
}

// ── Auth ──────────────────────────────────────────────────────────────────────
function setSession(token, username) {
  authToken = token;
  currentUsername = username;
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USERNAME_KEY, username);
  if (currentUsernameEl) currentUsernameEl.textContent = username;
  if (historyUsernameLabel) historyUsernameLabel.textContent = username;
}

function clearSession() {
  authToken = null;
  currentUsername = null;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USERNAME_KEY);
}

function showApp() {
  authScreen?.classList.add("hidden");
  appShell?.classList.remove("hidden");
}

function showAuthScreen() {
  appShell?.classList.add("hidden");
  authScreen?.classList.remove("hidden");
  historyData = [];
  currentAnalysis = null;
  renderHistory();
  resultEmptyState?.classList.remove("hidden");
  resultContent?.classList.add("hidden");
  xaiEmptyState?.classList.remove("hidden");
  xaiContent?.classList.add("hidden");
}

async function verifySession() {
  if (!authToken) { showAuthScreen(); return; }
  try {
    const res = await fetch(`${API_BASE}/me`, { headers: authHeaders() });
    if (!res.ok) throw new Error("invalid");
    const data = await res.json();
    setSession(authToken, data.username);
    showApp();
    await fetchHistory();
    updateProfileStats(data);
  } catch {
    clearSession();
    showAuthScreen();
  }
}

// Auth tab switching
authTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    authTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.authTab;
    loginForm?.classList.toggle("hidden", target !== "login");
    registerForm?.classList.toggle("hidden", target !== "register");
  });
});

loginForm?.addEventListener("submit", async e => {
  e.preventDefault();
  const username = document.getElementById("loginUsername").value.trim();
  const password = document.getElementById("loginPassword").value;
  try {
    const res = await fetch(`${API_BASE}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Đăng nhập thất bại.");
    setSession(data.token, data.username);
    showApp();
    await fetchHistory();
    await refreshProfileStats();
    showToast(`Chào ${data.username}!`, "success");
    loginForm.reset();
  } catch (err) { showToast(err.message, "danger"); }
});

registerForm?.addEventListener("submit", async e => {
  e.preventDefault();
  const username = document.getElementById("registerUsername").value.trim();
  const password = document.getElementById("registerPassword").value;
  const confirm  = document.getElementById("registerPasswordConfirm").value;
  if (password !== confirm) { showToast("Mật khẩu nhập lại không khớp.", "warning"); return; }
  try {
    const res = await fetch(`${API_BASE}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Đăng ký thất bại.");
    setSession(data.token, data.username);
    showApp();
    await refreshProfileStats();
    showToast(`Tạo tài khoản thành công, chào ${data.username}!`, "success");
    registerForm.reset();
  } catch (err) { showToast(err.message, "danger"); }
});

logoutBtn?.addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/logout`, { method: "POST", headers: authHeaders() });
  } catch {}
  clearSession();
  showAuthScreen();
  showToast("Đã đăng xuất.", "neutral");
});

// ── Change password ───────────────────────────────────────────────────────────
changePwForm?.addEventListener("submit", async e => {
  e.preventDefault();
  const old_pw  = document.getElementById("oldPassword").value;
  const new_pw  = document.getElementById("newPassword").value;
  const confirm = document.getElementById("newPasswordConfirm").value;
  if (new_pw !== confirm) { showToast("Mật khẩu mới nhập lại không khớp.", "warning"); return; }
  try {
    const res = await fetch(`${API_BASE}/change-password`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ old_password: old_pw, new_password: new_pw })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Đổi mật khẩu thất bại.");
    showToast("Đổi mật khẩu thành công.", "success");
    changePwForm.reset();
  } catch (err) { showToast(err.message, "danger"); }
});

// ── Navigation ────────────────────────────────────────────────────────────────
function showPage(pageId) {
  pages.forEach(p => p.classList.remove("active-page"));
  document.getElementById(pageId)?.classList.add("active-page");
  navLinks.forEach(l => l.classList.toggle("active", l.dataset.page === pageId));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (pageId === "profile") refreshProfileStats();
  if (pageId === "history") fetchHistory();
}

navLinks.forEach(l => l.addEventListener("click", () => showPage(l.dataset.page)));
document.querySelectorAll("[data-goto]").forEach(el => {
  el.addEventListener("click", () => showPage(el.dataset.goto));
});

// ── File upload ───────────────────────────────────────────────────────────────
async function handleSelectedFile(file) {
  if (!file) return;
  if (!file.type?.startsWith("image/")) {
    showToast("Vui lòng chọn file ảnh hợp lệ (JPG, PNG, JPEG).", "warning");
    return;
  }
  uploadedFile = file;
  fileName.textContent = file.name;
  uploadedImageData = await new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = e => res(e.target.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
  previewImage.src = uploadedImageData;
  previewImage.style.display = "block";
  previewFrame?.querySelector(".frame-placeholder")?.style.setProperty("display", "none");
}

imageInput?.addEventListener("change", function() { handleSelectedFile(this.files[0]); });
["dragenter","dragover"].forEach(ev => uploadBox?.addEventListener(ev, e => { e.preventDefault(); uploadBox.classList.add("drag-active"); }));
["dragleave","drop"].forEach(ev => uploadBox?.addEventListener(ev, e => { e.preventDefault(); uploadBox.classList.remove("drag-active"); }));
uploadBox?.addEventListener("drop", e => handleSelectedFile(e.dataTransfer?.files?.[0]));

// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn?.addEventListener("click", async () => {
  if (!uploadedFile) { showToast("Vui lòng chọn ảnh X-quang trước khi phân tích.", "warning"); return; }
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Đang phân tích…";
  try {
    const formData = new FormData();
    formData.append("image", uploadedFile);
    formData.append("file", uploadedFile);
    const res = await fetch(API_URL, { method: "POST", body: formData, headers: authHeaders() });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Backend không phản hồi.");
    const normalized = normalizeResult(data);
    currentAnalysis = normalized;
    updateResultPage(normalized);
    updateXaiPage(normalized);
    await fetchHistory();
    showPage("result");
    showToast("Phân tích hoàn tất. Kết quả đã được lưu.", "success");
  } catch (err) {
    console.error(err);
    showToast(`Không phân tích được: ${err.message}`, "danger");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Phân tích ảnh";
  }
});

function normalizeResult(data) {
  const rawPred   = data.prediction || "---";
  const prediction = formatLabel(rawPred);
  let confidence  = Number(data.confidence ?? 0);
  if (confidence <= 1) confidence *= 100;
  const probabilities = normalizeProbabilities(data.probabilities || {});
  const rawXai = data.xai_images || {};
  const xaiImages = {
    gradcam:             rawXai.gradcam || "",
    gradcampp:           rawXai.gradcampp || "",
    integratedGradients: rawXai.integrated_gradients || rawXai.integratedGradients || "",
    occlusion:           rawXai.occlusion || ""
  };
  return {
    id:            data.diagnosis_id || Date.now(),
    fileName:      uploadedFile?.name || "xray-image",
    date:          new Date().toLocaleDateString("vi-VN"),
    model:         data.model || "ResNet101",
    prediction,
    confidence,
    probabilities,
    status:        getStatusText(prediction),
    statusClass:   getStatusClass(prediction),
    originalImage: data.original_image || uploadedImageData || "",
    comparisonImage: data.xai_comparison || "",
    xaiImages
  };
}

function normalizeProbabilities(probs) {
  const out = { "COVID-19":0, Normal:0, Pneumonia:0, Tuberculosis:0 };
  Object.entries(probs).forEach(([k, v]) => {
    const label = formatLabel(k);
    let val = Number(v);
    if (val <= 1) val *= 100;
    if (out[label] !== undefined) out[label] = val;
  });
  return out;
}

function formatLabel(l) {
  const u = String(l || "").toUpperCase();
  if (u === "COVID19" || u === "COVID-19") return "COVID-19";
  if (u === "NORMAL")       return "Normal";
  if (u === "PNEUMONIA")    return "Pneumonia";
  if (u === "TUBERCULOSIS") return "Tuberculosis";
  return l || "---";
}

function getStatusText(l) {
  if (l === "Normal")                      return "Bình thường";
  if (l === "Pneumonia" || l === "COVID-19") return "Cần theo dõi";
  if (l === "Tuberculosis")                return "Cần kiểm tra thêm";
  return "---";
}

function getStatusClass(l) {
  if (l === "Normal")                      return "success";
  if (l === "Pneumonia" || l === "COVID-19") return "warning";
  if (l === "Tuberculosis")                return "danger";
  return "neutral";
}

// ── Result page ───────────────────────────────────────────────────────────────
function updateResultPage(r) {
  resultEmptyState?.classList.add("hidden");
  resultContent?.classList.remove("hidden");
  predictedClass.textContent  = r.prediction;
  confidenceScore.textContent = `${r.confidence.toFixed(1)}%`;
  resultStatusBadge.textContent = `Trạng thái: ${r.status}`;
  resultStatusBadge.className   = `badge ${r.statusClass}`;
  resultCardMain?.style.setProperty("--status-color", STATUS_COLORS[r.statusClass] || STATUS_COLORS.neutral);
  updateGauge(r.confidence);
  resultImageBox.innerHTML = `<img src="${r.originalImage}" alt="Ảnh X-quang đầu vào">`;
  probabilityList.innerHTML = Object.entries(r.probabilities).map(([lbl, val]) => `
    <div class="prob-row">
      <span>${lbl}</span>
      <div class="progress"><b style="width:${val}%"></b></div>
      <strong>${val.toFixed(1)}%</strong>
    </div>`).join("");
}

function updateGauge(confidence) {
  if (!gaugeCircle) return;
  const offset = GAUGE_CIRCUMFERENCE * (1 - Math.max(0, Math.min(confidence, 100)) / 100);
  gaugeCircle.style.strokeDashoffset = String(offset);
}

// ── XAI page ──────────────────────────────────────────────────────────────────
function updateXaiPage(r) {
  xaiEmptyState?.classList.add("hidden");
  xaiContent?.classList.remove("hidden");
  xaiPredictedClass.textContent = r.prediction;
  xaiConfidence.textContent     = `${r.confidence.toFixed(1)}%`;
  xaiInputPreview.innerHTML     = `<img src="${r.originalImage}" alt="XAI input">`;
  if (r.comparisonImage) {
    xaiStripCard?.classList.remove("hidden");
    xaiComparisonImage.src = r.comparisonImage;
  } else {
    xaiStripCard?.classList.add("hidden");
  }
  setXaiImg(gradcamImage,    r.xaiImages.gradcam);
  setXaiImg(gradcamppImage,  r.xaiImages.gradcampp);
  setXaiImg(igImage,         r.xaiImages.integratedGradients);
  setXaiImg(occlusionImage,  r.xaiImages.occlusion);
}

function setXaiImg(el, src) {
  if (!el) return;
  const card = el.closest(".xai-card");
  if (src) { el.src = src; card?.classList.remove("unavailable"); }
  else     { el.removeAttribute("src"); card?.classList.add("unavailable"); }
}

viewXaiBtn?.addEventListener("click", () => {
  if (!currentAnalysis) { showToast("Vui lòng phân tích ảnh trước.", "warning"); return; }
  updateXaiPage(currentAnalysis);
  showPage("xai");
});

// ── Export report ─────────────────────────────────────────────────────────────
exportReportBtn?.addEventListener("click", () => {
  if (!currentAnalysis) { showToast("Chưa có kết quả để xuất báo cáo.", "warning"); return; }
  buildPrintReport(currentAnalysis);
});

function buildPrintReport(r) {
  const xaiPairs = [
    { label: "Grad-CAM",             src: r.xaiImages.gradcam },
    { label: "Grad-CAM++",           src: r.xaiImages.gradcampp },
    { label: "Integrated Gradients", src: r.xaiImages.integratedGradients },
    { label: "Occlusion Sensitivity",src: r.xaiImages.occlusion }
  ].filter(x => x.src);

  printReport.innerHTML = `
    <div class="print-report-header">
      <h2>Báo cáo chẩn đoán — MediAI Chest</h2>
      <button class="print-report-close" id="closePrintBtn">✕ Đóng</button>
    </div>
    <div class="print-report-body">
      <div class="report-row">
        <div class="report-image-box">
          <img src="${r.originalImage}" alt="Ảnh X-quang">
        </div>
        <div class="report-meta">
          <h3>${r.prediction}</h3>
          <div class="report-stat"><span>Độ tin cậy</span><strong>${r.confidence.toFixed(1)}%</strong></div>
          <div class="report-stat"><span>Trạng thái</span><strong>${r.status}</strong></div>
          <div class="report-stat"><span>Mô hình</span><strong>${r.model}</strong></div>
          <div class="report-stat"><span>Tên file</span><strong>${r.fileName}</strong></div>
          <div class="report-stat"><span>Ngày phân tích</span><strong>${r.date}</strong></div>
          <div class="report-stat"><span>Tài khoản</span><strong>${currentUsername || "---"}</strong></div>
        </div>
      </div>
      <div>
        <p style="font-weight:800;margin:0 0 10px">Xác suất từng lớp:</p>
        ${Object.entries(r.probabilities).map(([lbl,val]) => `
          <div class="prob-row" style="margin-bottom:8px">
            <span>${lbl}</span>
            <div class="progress"><b style="width:${val}%"></b></div>
            <strong>${val.toFixed(1)}%</strong>
          </div>`).join("")}
      </div>
      ${xaiPairs.length ? `
      <div>
        <p style="font-weight:800;margin:0 0 10px">Giải thích XAI:</p>
        <div class="report-xai">
          ${xaiPairs.map(x => `<div><img src="${x.src}" alt="${x.label}"><p>${x.label}</p></div>`).join("")}
        </div>
      </div>` : ""}
      <div class="report-disclaimer">
        Kết quả chỉ mang tính hỗ trợ tham khảo. Không thay thế chẩn đoán của bác sĩ.
      </div>
    </div>
    <div class="print-actions">
      <button class="btn primary-btn" onclick="window.print()">
        <svg style="width:16px;height:16px"><use href="#icon-print"></use></svg>
        In / Lưu PDF
      </button>
    </div>
  `;

  printOverlay.classList.remove("hidden");
  document.getElementById("closePrintBtn")?.addEventListener("click", () => {
    printOverlay.classList.add("hidden");
  });
}

printOverlay?.addEventListener("click", e => {
  if (e.target === printOverlay) printOverlay.classList.add("hidden");
});

// ── History ───────────────────────────────────────────────────────────────────
async function fetchHistory() {
  if (!authToken) return;
  try {
    const res = await fetch(`${API_BASE}/history`, { headers: authHeaders() });
    if (res.status === 401) { clearSession(); showAuthScreen(); return; }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Không tải được lịch sử.");
    historyData = (data.history || []).map(item => ({
      id:            item.id,
      fileName:      item.fileName || "xray-image",
      date:          formatDate(item.date),
      model:         "ResNet101",
      prediction:    item.prediction,
      confidence:    Number(item.confidence) || 0,
      probabilities: item.probabilities || {},
      status:        getStatusText(item.prediction),
      statusClass:   getStatusClass(item.prediction),
      originalImage: item.originalImage || "",
      comparisonImage: "",
      xaiImages:     { gradcam:"", gradcampp:"", integratedGradients:"", occlusion:"" }
    }));
    renderHistory();
  } catch (err) {
    showToast(`Lịch sử: ${err.message}`, "danger");
  }
}

async function loadHistoryDetail(id) {
  try {
    const res = await fetch(`${API_BASE}/history/${id}`, { headers: authHeaders() });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Không tải được chi tiết.");
    const r = {
      id:            data.id,
      fileName:      data.fileName || "xray-image",
      date:          formatDate(data.date),
      model:         "ResNet101",
      prediction:    data.prediction,
      confidence:    Number(data.confidence) || 0,
      probabilities: normalizeProbabilities(data.probabilities || {}),
      status:        getStatusText(data.prediction),
      statusClass:   getStatusClass(data.prediction),
      originalImage: data.originalImage || "",
      comparisonImage: data.xai_comparison || "",
      xaiImages: {
        gradcam:             data.xai_images?.gradcam || "",
        gradcampp:           data.xai_images?.gradcampp || "",
        integratedGradients: data.xai_images?.integrated_gradients || "",
        occlusion:           data.xai_images?.occlusion || ""
      }
    };
    currentAnalysis = r;
    updateResultPage(r);
    updateXaiPage(r);
    showPage("result");
  } catch (err) { showToast(err.message, "danger"); }
}

async function deleteHistoryItem(id) {
  if (!confirm("Xóa bản ghi này khỏi lịch sử?")) return;
  try {
    const res = await fetch(`${API_BASE}/history/${id}`, {
      method: "DELETE", headers: authHeaders()
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Xóa thất bại.");
    showToast("Đã xóa bản ghi.", "success");
    await fetchHistory();
  } catch (err) { showToast(err.message, "danger"); }
}

function renderHistory() {
  if (!historyTableBody) return;
  historyEmpty?.classList.toggle("hidden", historyData.length > 0);
  const keyword = searchInput?.value.toLowerCase().trim() || "";
  const filtered = historyData.filter(item => {
    const matchFilter  = currentFilter === "Tất cả" || item.prediction === currentFilter;
    const matchKeyword = [item.fileName, item.date, item.prediction, item.status]
      .join(" ").toLowerCase().includes(keyword);
    return matchFilter && matchKeyword;
  });
  historyTableBody.innerHTML = filtered.map((item, i) => `
    <tr>
      <td>${String(i + 1).padStart(2,"0")}</td>
      <td><img class="history-thumb" src="${item.originalImage}" alt="ảnh" onerror="this.style.background='#0f172a'"></td>
      <td>${item.date}</td>
      <td>${item.model}</td>
      <td>${item.prediction}</td>
      <td class="mono">${item.confidence.toFixed(1)}%</td>
      <td><span class="badge ${item.statusClass}">${item.status}</span></td>
      <td><button class="table-btn history-view-btn" data-id="${item.id}">Xem</button></td>
      <td><button class="delete-btn history-delete-btn" data-id="${item.id}">
        <svg style="width:14px;height:14px"><use href="#icon-trash"></use></svg>
      </button></td>
    </tr>`).join("");

  document.querySelectorAll(".history-view-btn").forEach(btn => {
    btn.addEventListener("click", () => loadHistoryDetail(Number(btn.dataset.id)));
  });
  document.querySelectorAll(".history-delete-btn").forEach(btn => {
    btn.addEventListener("click", () => deleteHistoryItem(Number(btn.dataset.id)));
  });
}

filterButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    filterButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.textContent.trim();
    renderHistory();
  });
});
searchInput?.addEventListener("input", renderHistory);

// ── Profile stats ─────────────────────────────────────────────────────────────
async function refreshProfileStats() {
  if (!authToken) return;
  try {
    const res = await fetch(`${API_BASE}/me`, { headers: authHeaders() });
    if (!res.ok) return;
    const data = await res.json();
    updateProfileStats(data);
  } catch {}
}

function updateProfileStats(data) {
  if (!data) return;
  const stats = data.stats || {};
  if (profileBadge)    profileBadge.textContent  = data.username || "---";
  if (profileUsername) profileUsername.textContent = data.username || "---";
  if (profileJoined)   profileJoined.textContent  = `Ngày tham gia: ${formatDate(data.created_at || "")}`;
  if (profileTotal)    profileTotal.textContent   = data.total_scans ?? 0;
  if (profileCovid)    profileCovid.textContent   = stats["COVID-19"] ?? 0;
  if (profileNormal)   profileNormal.textContent  = stats["Normal"] ?? 0;
  if (profilePneumonia) profilePneumonia.textContent = stats["Pneumonia"] ?? 0;
  if (profileTB)       profileTB.textContent      = stats["Tuberculosis"] ?? 0;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
verifySession();