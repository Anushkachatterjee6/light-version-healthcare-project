(() => {
  const caseSelect = document.getElementById("caseSelect");
  const modalitySelect = document.getElementById("modalitySelect");
  const slice1 = document.getElementById("slice1");
  const slice2 = document.getElementById("slice2");
  const runBtn = document.getElementById("runBtn");
  const statusMsg = document.getElementById("statusMsg");
  const sliceHint = document.getElementById("sliceHint");
  const results = document.getElementById("results");
  const stats = document.getElementById("stats");

  let cases = [];

  function setStatus(text, kind = "") {
    statusMsg.textContent = text || "";
    statusMsg.className = `status ${kind}`.trim();
  }

  function b64Img(b64) {
    return `data:image/png;base64,${b64}`;
  }

  function fillModalities(caseObj) {
    modalitySelect.innerHTML = "";
    const mods = caseObj.modalities || [];
    mods.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m.toUpperCase();
      modalitySelect.appendChild(opt);
    });
    if (caseObj.default_modality) {
      modalitySelect.value = caseObj.default_modality;
    }
  }

  function applySliceDefaults(numSlices) {
    const mid = Math.floor(numSlices / 2);
    const s1 = Math.max(0, mid);
    const s2 = Math.min(numSlices - 1, mid + 2);
    slice1.max = numSlices - 1;
    slice2.max = numSlices - 1;
    slice1.value = s1;
    slice2.value = s2;
    sliceHint.textContent = `Volume has ${numSlices} slices (0–${numSlices - 1}). Nearby slices work best.`;
  }

  async function refreshMeta() {
    const caseId = caseSelect.value;
    const modality = modalitySelect.value;
    if (!caseId || !modality) return;
    const res = await fetch(`/api/cases/${encodeURIComponent(caseId)}/meta?modality=${encodeURIComponent(modality)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setStatus(err.detail || "Failed to load case metadata", "err");
      return;
    }
    const meta = await res.json();
    applySliceDefaults(meta.num_slices);
  }

  async function loadCases() {
    setStatus("Loading MRI cases…");
    const res = await fetch("/api/cases");
    if (!res.ok) {
      setStatus("Could not load cases. Is the dataset folder present?", "err");
      return;
    }
    const data = await res.json();
    cases = data.cases || [];
    caseSelect.innerHTML = "";
    if (!cases.length) {
      setStatus("No NIfTI cases found under BraTS2020/. Add MRI volumes first.", "err");
      return;
    }
    cases.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.case_id;
      opt.textContent = c.case_id;
      caseSelect.appendChild(opt);
    });
    fillModalities(cases[0]);
    await refreshMeta();
    setStatus("Ready. Select two nearby slices, then run analysis.", "ok");
  }

  function renderStats(payload) {
    const ratio = ((payload.inlier_ratio || 0) * 100).toFixed(1);
    const items = [
      ["Keypoints in Slice 1", payload.keypoints_image1],
      ["Keypoints in Slice 2", payload.keypoints_image2],
      ["Good SIFT Matches", payload.good_matches],
      ["RANSAC Inliers", payload.ransac_inliers],
      ["RANSAC Outliers", payload.ransac_outliers],
      ["Inlier Ratio", `${ratio}%`],
    ];
    stats.innerHTML = items
      .map(
        ([label, value]) =>
          `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>`
      )
      .join("");
  }

  async function runAnalysis() {
    runBtn.disabled = true;
    setStatus("Running SIFT + RANSAC on MRI slices…");
    try {
      const body = {
        case_id: caseSelect.value,
        modality: modalitySelect.value,
        slice1: Number(slice1.value),
        slice2: Number(slice2.value),
      };
      const res = await fetch("/api/image-analysis/sift-ransac", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await res.json();
      if (!res.ok) {
        const detail = payload.detail;
        const msg = Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : detail || "Analysis failed";
        throw new Error(msg);
      }

      document.getElementById("orig1").src = b64Img(payload.original_image1);
      document.getElementById("orig2").src = b64Img(payload.original_image2);
      document.getElementById("sift1").src = b64Img(payload.sift_visualization_1);
      document.getElementById("sift2").src = b64Img(payload.sift_visualization_2);
      document.getElementById("matches").src = b64Img(payload.matches_visualization);
      document.getElementById("ransac").src = b64Img(payload.ransac_visualization);

      renderStats(payload);
      results.classList.remove("hidden");
      setStatus(payload.message || "Analysis complete.", payload.success ? "ok" : "err");
      document.getElementById("ransac").scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (err) {
      setStatus(err.message || String(err), "err");
    } finally {
      runBtn.disabled = false;
    }
  }

  caseSelect.addEventListener("change", async () => {
    const c = cases.find((x) => x.case_id === caseSelect.value);
    if (c) fillModalities(c);
    await refreshMeta();
  });
  modalitySelect.addEventListener("change", refreshMeta);
  runBtn.addEventListener("click", runAnalysis);

  loadCases().catch(() => setStatus("Cannot connect to backend. Start run_server.bat and open http://127.0.0.1:8000/image-analysis", "err"));
})();
