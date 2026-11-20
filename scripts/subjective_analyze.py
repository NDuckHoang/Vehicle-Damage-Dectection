"""
Danh gia CHU QUAN - Buoc 2: phan tich cac file CSV nguoi cham gui lai.

Dat cac file danh_gia_*.csv vao subjective/responses/ roi chay:
    python scripts/subjective_analyze.py

Tinh:
1. MOS (Mean Opinion Score) chat luong phat hien: trung binh +- do lech chuan, khoang tin cay 95%.
2. Muc do (Khong hu hong / Nhe / Trung binh / Nang):
   - Do dong y he thong vs tung nguoi cham: trung khop, lech <= 1 muc, Cohen's Kappa co trong so tuyen tinh.
   - Y kien so dong (trung vi cua cac nguoi cham) vs he thong + ma tran nham lan.
   - Do dong y GIUA cac nguoi cham: Fleiss' Kappa, Kappa trung binh tung cap.
Ket qua: reports/subjective_evaluation.md, results/subjective/*.png
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBJ_DIR = PROJECT_ROOT / "subjective"
SEV = ["Không hư hỏng", "Nhẹ", "Trung bình", "Nặng"]
SYS_TO_IDX = {"Không phát hiện hư hỏng": 0, "Nhẹ": 1, "Trung bình": 2, "Nặng": 3}
MARKER = "<!-- PHAN TICH VIET TAY: noi dung ben duoi duoc giu nguyen khi chay lai -->"


def load_responses(resp_dir, image_ids):
    """Doc va kiem tra cac file CSV. Tra ve DataFrame: rater, image_id, severity, score, comment."""
    frames, problems = [], []
    for f in sorted(Path(resp_dir).glob("*.csv")):
        try:
            d = pd.read_csv(f, encoding="utf-8-sig", dtype={"comment": str}, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{f.name}: không đọc được ({e})")
            continue
        need = {"rater", "image_id", "human_severity", "detection_score"}
        if not need <= set(d.columns):
            problems.append(f"{f.name}: thiếu cột {need - set(d.columns)}")
            continue
        d["human_severity"] = pd.to_numeric(d["human_severity"], errors="coerce")
        d["detection_score"] = pd.to_numeric(d["detection_score"], errors="coerce")
        bad = (~d["human_severity"].isin([0, 1, 2, 3]) | ~d["detection_score"].isin([1, 2, 3, 4, 5]))
        if set(d["image_id"]) != set(image_ids) or bad.any() or len(d) != len(image_ids):
            problems.append(f"{f.name}: thiếu ảnh hoặc giá trị không hợp lệ (bỏ qua file)")
            continue
        frames.append(d)
    if not frames:
        return None, problems
    df = pd.concat(frames, ignore_index=True)
    df["rater"] = df["rater"].astype(str).str.strip()
    dup = df.groupby("rater")["image_id"].count()
    for r in dup[dup > len(image_ids)].index:
        problems.append(f"Người chấm '{r}' có nhiều hơn 1 file: chỉ giữ file đầu tiên")
        first = df[df["rater"] == r].index[: len(image_ids)]
        df = df.drop(df[(df["rater"] == r) & ~df.index.isin(first)].index)
    df = df.rename(columns={"human_severity": "severity", "detection_score": "score"})
    if "comment" not in df.columns:
        df["comment"] = ""
    return df[["rater", "image_id", "severity", "score", "comment"]], problems


def wkappa(a, b):
    """Cohen's Kappa trong so tuyen tinh (phu hop thang thu tu). NaN neu khong xac dinh."""
    try:
        k = cohen_kappa_score(a, b, labels=[0, 1, 2, 3], weights="linear")
        return float(k) if np.isfinite(k) else np.nan
    except ValueError:
        return np.nan


def fleiss_kappa(table):
    """table: (so anh x so muc) = so nguoi chon moi muc cho moi anh."""
    table = np.asarray(table, dtype=float)
    n = table.sum(axis=1)[0]  # so nguoi cham moi anh
    if n < 2:
        return np.nan
    p_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_j = table.sum(axis=0) / table.sum()
    p_e = (p_j ** 2).sum()
    return np.nan if p_e == 1 else float((p_i.mean() - p_e) / (1 - p_e))


def kappa_text(k):
    """Dien giai theo Landis & Koch (1977)."""
    if k is None or np.isnan(k):
        return "không xác định"
    for th, t in [(0, "kém"), (0.2, "rất thấp"), (0.4, "thấp"), (0.6, "trung bình"), (0.8, "khá cao")]:
        if k <= th:
            return t
    return "rất cao"


def f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", default=str(SUBJ_DIR / "responses"))
    p.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "subjective_evaluation.md"))
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "subjective"))
    args = p.parse_args()
    out_dir, report = Path(args.out_dir), Path(args.report)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = json.loads((SUBJ_DIR / "sample.json").read_text(encoding="utf-8"))
    items = {it["image_id"]: it for it in sample["items"]}
    ids = list(items)
    df, problems = load_responses(args.responses, ids)
    for pr in problems:
        print("[!]", pr)
    if df is None:
        raise SystemExit(f"Không có file CSV hợp lệ trong {args.responses}")

    raters = sorted(df["rater"].unique())
    sys_lv = pd.Series({i: SYS_TO_IDX[items[i]["system_level"]] for i in ids})
    S = df.pivot(index="image_id", columns="rater", values="severity").loc[ids, raters].astype(int)
    Q = df.pivot(index="image_id", columns="rater", values="score").loc[ids, raters].astype(int)
    print(f"{len(raters)} người chấm × {len(ids)} ảnh")

    # ---- 1. MOS chat luong phat hien ----
    allq = Q.values.ravel()
    mos, sd = allq.mean(), allq.std(ddof=1) if allq.size > 1 else 0.0
    ci = 1.96 * sd / np.sqrt(allq.size)
    mos_img = Q.mean(axis=1)
    mos_by_sys = Q.mean(axis=1).groupby(sys_lv.map(lambda v: SEV[v] if v else "Không phát hiện")).agg(["mean", "count"])

    # ---- 2. Muc do: he thong vs nguoi ----
    per_rater = []
    for r in raters:
        h = S[r]
        per_rater.append({"rater": r, "exact": (h == sys_lv).mean(), "within1": ((h - sys_lv).abs() <= 1).mean(),
                          "kappa": wkappa(h, sys_lv), "bias": (sys_lv - h).mean(), "mos": Q[r].mean()})
    pr = pd.DataFrame(per_rater)
    # Y kien so dong = trung vi, lam tron 0.5 len (vd: trung vi 1.5 -> 2)
    consensus = np.floor(S.median(axis=1) + 0.5).astype(int)
    cons_exact = (consensus == sys_lv).mean()
    cons_within1 = ((consensus - sys_lv).abs() <= 1).mean()
    cons_kappa = wkappa(consensus, sys_lv)
    cm = pd.crosstab(pd.Categorical(consensus.map(lambda v: SEV[v]), categories=SEV),
                     pd.Categorical(sys_lv.map(lambda v: SEV[v]), categories=SEV), dropna=False)

    # ---- 3. Dong y giua nguoi cham ----
    pairs = list(itertools.combinations(raters, 2))
    pair_k = [wkappa(S[a], S[b]) for a, b in pairs]
    pair_exact = [(S[a] == S[b]).mean() for a, b in pairs]
    fk = fleiss_kappa(np.stack([(S.values == c).sum(axis=1) for c in range(4)], axis=1))
    mean_pair_k = np.nanmean(pair_k) if pairs and not np.all(np.isnan(pair_k)) else np.nan
    mean_sys_k = np.nanmean(pr["kappa"]) if not pr["kappa"].isna().all() else np.nan

    # ---- Bieu do ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].bar(ids, mos_img, yerr=Q.std(axis=1, ddof=1) if len(raters) > 1 else None, capsize=3, color="#1f5eff")
    ax[0].axhline(mos, color="#d23c3c", ls="--", label=f"MOS chung = {mos:.2f}")
    ax[0].set_ylim(0, 5.5)
    ax[0].set_title("Điểm chất lượng phát hiện theo ảnh (1–5)")
    ax[0].legend()
    bottom = np.zeros(len(ids))
    for c, col in zip(range(4), ["#8a94a3", "#2e9e5b", "#e09b1a", "#d23c3c"]):
        v = (S.values == c).sum(axis=1)
        ax[1].bar(ids, v, bottom=bottom, color=col, label=SEV[c])
        bottom += v
    ax[1].scatter(ids, [len(raters) + 0.4] * len(ids), marker="v", s=60,
                  c=[["#8a94a3", "#2e9e5b", "#e09b1a", "#d23c3c"][v] for v in sys_lv])
    ax[1].set_title("Số phiếu mức độ của người chấm (▼ = mức của hệ thống)")
    ax[1].set_ylim(0, len(raters) + 1)
    ax[1].legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    plt.tight_layout()
    plt.savefig(out_dir / "subjective_summary.png", dpi=120)
    plt.close()

    # ---- Bao cao ----
    L = ["# Đánh giá chủ quan (Subjective evaluation)", "",
         "> Sinh tự động bởi `scripts/subjective_analyze.py` từ phiếu chấm của người đánh giá "
         "(`subjective/responses/*.csv`). Số liệu là **ý kiến thực tế của người chấm**.", "",
         "## 1. Thiết kế", "",
         f"- **{len(raters)} người chấm** ({', '.join(raters)}), mỗi người chấm **cùng {len(ids)} ảnh** "
         f"từ tập validation, chọn ngẫu nhiên (seed {sample['seed']}) theo mức độ hệ thống dự đoán.",
         "- **Bước 1 (chấm mù):** người chấm chỉ xem ảnh gốc, chọn mức độ Không hư hỏng / Nhẹ / Trung bình / Nặng. "
         "Sau khi xác nhận thì không sửa được.",
         "- **Bước 2:** xem vùng hư hỏng do máy phát hiện, chấm điểm chất lượng phát hiện 1–5 (MOS).",
         f"- Ngưỡng confidence của hệ thống: {sample['conf']}.", "",
         "## 2. Chất lượng phát hiện (MOS)", "",
         f"**MOS = {mos:.2f} ± {sd:.2f}** (độ lệch chuẩn; khoảng tin cậy 95%: {mos - ci:.2f}–{mos + ci:.2f}; "
         f"{allq.size} lượt chấm).", "",
         "| Mức hệ thống dự đoán | MOS | Số ảnh |", "|---|---|---|"]
    for k, row in mos_by_sys.iterrows():
        L.append(f"| {k} | {row['mean']:.2f} | {int(row['count'])} |")
    L += ["", "## 3. Mức độ hư hỏng: hệ thống so với người chấm", "",
          "| Người chấm | Trùng khớp | Lệch ≤ 1 mức | Kappa (trọng số) | Độ lệch TB (hệ thống − người) | MOS người này cho |",
          "|---|---|---|---|---|---|"]
    for _, r in pr.iterrows():
        L.append(f"| {r['rater']} | {r['exact'] * 100:.0f}% | {r['within1'] * 100:.0f}% | {f(r['kappa'])} "
                 f"({kappa_text(r['kappa'])}) | {r['bias']:+.2f} | {r['mos']:.2f} |")
    L += ["", f"**Ý kiến số đông** (trung vị các người chấm) so với hệ thống: trùng khớp **{cons_exact * 100:.0f}%**, "
              f"lệch ≤ 1 mức {cons_within1 * 100:.0f}%, Kappa = {f(cons_kappa)} ({kappa_text(cons_kappa)}).", "",
          "Ma trận nhầm lẫn (hàng: ý kiến số đông; cột: hệ thống):", "",
          "| | " + " | ".join(SEV) + " |", "|---" * 5 + "|"]
    for lv in SEV:
        L.append(f"| {lv} | " + " | ".join(str(cm.loc[lv, c]) for c in SEV) + " |")
    bias_all = pr["bias"].mean()
    L += ["", f"Độ lệch trung bình (hệ thống − người) = {bias_all:+.2f}: hệ thống có xu hướng đánh giá "
              + ("**nặng hơn**" if bias_all > 0.15 else "**nhẹ hơn**" if bias_all < -0.15 else "**tương đương**")
              + " so với người chấm.", "",
          "## 4. Độ đồng thuận giữa các người chấm", "",
          f"- Fleiss' Kappa (4 mức, không trọng số): **{f(fk)}** ({kappa_text(fk)}).",
          f"- Kappa trọng số trung bình giữa từng cặp người chấm: **{f(mean_pair_k)}** ({kappa_text(mean_pair_k)}); "
          f"trùng khớp trung bình {np.mean(pair_exact) * 100:.0f}%." if pairs else "- Cần ≥ 2 người chấm.",
          f"- Kappa trọng số trung bình giữa hệ thống và từng người: **{f(mean_sys_k)}**.",
          "- Nếu Kappa (hệ thống, người) gần bằng Kappa (người, người), hệ thống đồng thuận với con người "
          "ở mức tương đương con người đồng thuận với nhau.", "",
          "## 5. Chi tiết từng ảnh", "",
          "| Ảnh | Hệ thống | " + " | ".join(raters) + " | Số đông | MOS |",
          "|---|---|" + "---|" * len(raters) + "---|---|"]
    for i in ids:
        L.append(f"| {i} | {SEV[sys_lv[i]] if sys_lv[i] else 'Không phát hiện'} | "
                 + " | ".join(f"{SEV[S.loc[i, r]]} / {Q.loc[i, r]}đ" for r in raters)
                 + f" | {SEV[consensus[i]]} | {mos_img[i]:.2f} |")
    comments = df[df["comment"].astype(str).str.strip() != ""]
    if len(comments):
        L += ["", "### Nhận xét của người chấm", ""]
        for _, c in comments.sort_values("image_id").iterrows():
            L.append(f"- **{c['image_id']}** ({c['rater']}): {str(c['comment']).replace(chr(10), ' ')}")
    L += ["", "## 6. Lưu ý", "",
          f"- Mẫu nhỏ ({len(ids)} ảnh × {len(raters)} người): Kappa và MOS có sai số lớn, kết quả mang tính tham khảo.",
          "- Người chấm không phải giám định viên chuyên nghiệp; tiêu chí mức độ dựa trên gợi ý chung trong phiếu.",
          "- Biểu đồ: `results/subjective/subjective_summary.png`. Danh sách ảnh và kết quả hệ thống: "
          "`subjective/sample.json`."]

    manual = ""
    if report.exists() and MARKER in report.read_text(encoding="utf-8"):
        manual = report.read_text(encoding="utf-8").split(MARKER, 1)[1]
    report.write_text("\n".join(L) + "\n\n" + MARKER + manual, encoding="utf-8")
    print(f"MOS = {mos:.2f} ± {sd:.2f} | Số đông vs hệ thống: {cons_exact * 100:.0f}% trùng, "
          f"Kappa {f(cons_kappa)} | Fleiss {f(fk)}")
    print(f"Báo cáo: {report}")


if __name__ == "__main__":
    main()
