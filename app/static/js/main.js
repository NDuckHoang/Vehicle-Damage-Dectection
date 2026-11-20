// VehiDamage - xu ly giao dien: chon/keo tha anh, xem truoc, loading, phong to anh
(function () {
  const ALLOWED = ["image/jpeg", "image/png", "image/bmp", "image/webp"];

  // ---------- Trang chu: chon anh ----------
  const input = document.getElementById("file-input");
  if (input) {
    const dropzone = document.getElementById("dropzone");
    const preview = document.getElementById("preview");
    const count = document.getElementById("file-count");
    const submit = document.getElementById("submit-btn");
    const form = document.getElementById("upload-form");
    const conf = document.getElementById("conf");
    let files = []; // danh sach file dang chon

    // Dong bo danh sach file vao <input> (dung DataTransfer) va ve lai anh xem truoc
    function sync() {
      const dt = new DataTransfer();
      files.forEach((f) => dt.items.add(f));
      input.files = dt.files;

      preview.innerHTML = "";
      files.forEach((f, i) => {
        const ok = ALLOWED.includes(f.type);
        const div = document.createElement("div");
        div.className = "thumb" + (ok ? "" : " bad");
        div.innerHTML = (ok ? `<img alt="">` : `<div style="height:90px;display:grid;place-items:center">✖</div>`) +
          `<div class="name" title="${f.name}">${ok ? "" : "Không phải ảnh: "}${f.name}</div>` +
          `<button type="button" class="remove" title="Bỏ ảnh">×</button>`;
        if (ok) div.querySelector("img").src = URL.createObjectURL(f);
        div.querySelector(".remove").onclick = () => { files.splice(i, 1); sync(); };
        preview.appendChild(div);
      });

      const tooMany = files.length > MAX_FILES;
      count.textContent = files.length
        ? `Đã chọn ${files.length} tệp` + (tooMany ? ` – vượt quá ${MAX_FILES} ảnh, hãy bỏ bớt.` : "")
        : "";
      count.style.color = tooMany ? "#d23c3c" : "";
      submit.disabled = files.length === 0 || tooMany;
    }

    function addFiles(list) { files = files.concat(Array.from(list)); sync(); }

    input.addEventListener("change", () => addFiles(input.files));
    ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); dropzone.classList.add("drag");
    }));
    ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); dropzone.classList.remove("drag");
    }));
    dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

    conf.addEventListener("input", () => {
      document.getElementById("conf-value").textContent = Number(conf.value).toFixed(2);
    });
    form.addEventListener("submit", () => {
      document.getElementById("loading").hidden = false;
      submit.disabled = true;
    });
    // Khi bam "Back" tu trang ket qua: tat lop loading
    window.addEventListener("pageshow", () => { document.getElementById("loading").hidden = true; sync(); });
  }

  // ---------- Trang ket qua: bam anh de phong to ----------
  const lightbox = document.getElementById("lightbox");
  if (lightbox) {
    document.querySelectorAll(".zoomable").forEach((img) => {
      img.addEventListener("click", () => {
        lightbox.querySelector("img").src = img.src;
        lightbox.hidden = false;
      });
    });
    lightbox.addEventListener("click", () => { lightbox.hidden = true; });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") lightbox.hidden = true; });
  }
})();
