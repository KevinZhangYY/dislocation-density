function getTheme(){
  const saved = localStorage.getItem("dd_theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function setTheme(theme){
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("dd_theme", theme);
  const moon = document.getElementById("moon");
  const sun = document.getElementById("sun");
  if(moon && sun){
    if(theme === "dark"){
      moon.style.display = "none";
      sun.style.display = "block";
    } else {
      moon.style.display = "block";
      sun.style.display = "none";
    }
  }
}

function wireThemeToggle(){
  const btn = document.getElementById("themeToggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme || getTheme();
    setTheme(current === "dark" ? "light" : "dark");
  });
  setTheme(getTheme());
}

function wireDropzone(){
  const drop = document.getElementById("drop");
  const fileInput = document.getElementById("file");
  const runBtn = document.getElementById("runBtn");
  const fileName = document.getElementById("fileName");
  const form = document.getElementById("form");
  if (!drop || !fileInput || !runBtn || !form) return;

  function setFile(f){
    if (!f) return;
    fileName.textContent = `${f.name} (${Math.round(f.size/1024)} KB)`;
    runBtn.disabled = false;
    const dt = new DataTransfer();
    dt.items.add(f);
    fileInput.files = dt.files;
  }

  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("dragover");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    setFile(f);
  });
  drop.addEventListener("click", () => fileInput.click()); // Click dropzone to upload
  fileInput.addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    setFile(f);
  });

  form.addEventListener("submit", (e) => {
    if (!fileInput.files || fileInput.files.length === 0) {
      e.preventDefault();
      return;
    }
    runBtn.disabled = true;
    runBtn.textContent = "Processing...";
    const fd = new FormData(form);
    fd.set("file", fileInput.files[0], fileInput.files[0].name);
    fetch("/run", { method: "POST", body: fd })
      .then((r) => r.text())
      .then((html) => {
        document.open();
        document.write(html);
        document.close();
      })
      .catch(() => {
        runBtn.disabled = false;
        runBtn.textContent = "Process Image";
      });
    e.preventDefault();
  });
}

function openModal(img) {
  const modal = document.getElementById("modal");
  const modalImg = document.getElementById("modalImg");
  modal.classList.add("open");
  modalImg.src = img.src;
}

function closeModal(e) {
  if (e.target.id === "modal" || e.target.classList.contains("modal-close")) {
    document.getElementById("modal").classList.remove("open");
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    const modal = document.getElementById("modal");
    if(modal) modal.classList.remove("open");
  }
});

document.addEventListener("DOMContentLoaded", () => {
  wireThemeToggle();
  wireDropzone();
});
