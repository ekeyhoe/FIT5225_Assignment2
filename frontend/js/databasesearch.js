/* =========================
   STATE
========================= */
let multiSelectMode = false;
const selectedImages = new Set();

/* =========================
   INIT
========================= */
document.addEventListener("DOMContentLoaded", initApp);

function initApp() {
  setupSidebar();
  setupModeButtons();
  setMode("tag-search");
  setInitialActiveButton("tag-btn");
}

function setInitialActiveButton(id) {
  const btn = document.getElementById(id);
  if (!btn) return;

  document.querySelectorAll("[data-mode]").forEach((b) => {
    b.style.backgroundColor = "white";
  });

  btn.style.backgroundColor = "lightgreen";
}

/* =========================
   SIDEBAR
========================= */
function setupSidebar() {
  fetch("sidebar.html")
    .then((res) => res.text())
    .then((data) => {
      const el = document.getElementById("sidebar-container");
      if (el) el.innerHTML = data;
    });
}

/* =========================
   MODE SYSTEM
========================= */
async function setMode(mode) {
  const box = document.getElementById("contentBox");

  const response = await fetch(`searchmodes/${mode}.html`);
  box.innerHTML = await response.text();

  // rebind UI after DOM replacement
  setupGallery();
  syncMultiSelectUI();
}

/* =========================
   MODE BUTTONS
========================= */
function setupModeButtons() {
  document.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      setMode(btn.dataset.mode);
      highlightActive(btn);
    });
  });
}

function highlightActive(activeBtn) {
  document.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.style.backgroundColor = "white";
  });

  activeBtn.style.backgroundColor = "lightgreen";
}

/* =========================
   MULTI SELECT
========================= */
function toggleMultiSelect() {
  multiSelectMode = !multiSelectMode;

  syncMultiSelectUI();

  if (!multiSelectMode) {
    selectedImages.clear();

    document
      .querySelectorAll(".image-gallery img")
      .forEach((img) => img.classList.remove("selected"));
  }
  console.log("actions:", document.getElementById("multiSelectActions"));
}

function syncMultiSelectUI() {
  const btn = document.getElementById("multi-selectbtn");
  const actions = document.getElementById("multiSelectActions");

  if (btn) {
    btn.classList.toggle("active", multiSelectMode);
  }

  if (actions) {
    actions.classList.toggle("visible", multiSelectMode);
  }
}

/* =========================
   GALLERY
========================= */
function setupGallery() {
  document.querySelectorAll(".image-gallery img").forEach((img) => {
    img.onclick = () => {
      if (multiSelectMode) {
        img.classList.toggle("selected");

        const src = img.dataset.full || img.src;

        if (img.classList.contains("selected")) {
          selectedImages.add(src);
        } else {
          selectedImages.delete(src);
        }

        console.log([...selectedImages]);
      } else {
        window.location.href = `imagepage.html?src=${encodeURIComponent(img.src)}`;
      }
    };
  });
}
