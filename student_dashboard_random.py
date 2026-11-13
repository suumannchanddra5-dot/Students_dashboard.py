import argparse
import csv
import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_FILE = "students.csv"
DASHBOARD_FILE = "dashboard.png"

GRADE_BOUNDARIES = [(80, "A"), (70, "B"), (60, "C"), (50, "D"), (0, "F")]
GRADE_ORDER = ["A", "B", "C", "D", "F"]


def validate_score(score):
    score = float(score)
    if not 0 <= score <= 100:
        raise ValueError(f"Score must be between 0 and 100, got {score}")
    return score


def score_to_grade(score):
    score = validate_score(score)
    for minimum, grade in GRADE_BOUNDARIES:
        if score >= minimum:
            return grade


def average_score(scores):
    scores = list(scores)
    if not scores:
        raise ValueError("Cannot average an empty list of scores")
    return sum(scores) / len(scores)


def anonymise_id(real_id, salt="change-this-salt"):
    digest = hashlib.sha256((salt + str(real_id)).encode()).hexdigest()
    return "S" + digest[:8]


class StudentRecord:
    MAX_WEEKLY_HOURS = 100

    def __init__(self, student_id, course, study_hours, score):
        study_hours = float(study_hours)
        if not 0 <= study_hours <= self.MAX_WEEKLY_HOURS:
            raise ValueError(f"Study hours out of range: {study_hours}")
        self.student_id = str(student_id)
        self.course = str(course).strip()
        self.study_hours = study_hours
        self.score = validate_score(score)

    @property
    def grade(self):
        return score_to_grade(self.score)

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "course": self.course,
            "study_hours": self.study_hours,
            "score": self.score,
            "grade": self.grade,
        }

    @classmethod
    def from_dict(cls, row):
        return cls(row["student_id"], row["course"], row["study_hours"], row["score"])

    def __repr__(self):
        return (f"StudentRecord(id={self.student_id}, course={self.course}, "
                f"hours={self.study_hours:.1f}, score={self.score:.1f}, grade={self.grade})")


def generate_sample_csv(path, n_students=120, seed=42):
    rng = np.random.default_rng(seed)
    courses = {"Maths": -3, "Physics": 0, "Biology": 3, "History": 5}
    rows = []
    for i in range(n_students):
        course = list(courses)[i % len(courses)]
        hours = float(np.clip(rng.normal(12, 4), 1, 40))

        score = 45 + 1.8 * hours + courses[course] + rng.normal(0, 7)
        rows.append({
            "student_id": anonymise_id(1000 + i),
            "course": course,
            "study_hours": round(hours, 1),
            "score": round(float(np.clip(score, 0, 100)), 1),
        })

    rows[0].update(study_hours=60.0, score=28.0)
    rows[1].update(study_hours=1.0, score=97.0)
    rows[2].update(study_hours=55.0, score=35.0)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["student_id", "course", "study_hours", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created sample data file: {path}")


def load_records(path):
    records = []
    with open(path, newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            try:
                records.append(StudentRecord.from_dict(row))
            except (ValueError, KeyError) as err:
                print(f"  Skipping line {line_no}: {err}")
    return records


def records_to_dict(records):
    return {r.student_id: r.to_dict() for r in records}


def describe_array(values):
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "mean": values.mean(),
        "median": median,
        "std": values.std(ddof=1),
        "min": values.min(),
        "max": values.max(),
        "iqr": q3 - q1,
    }


def pearson_r(x, y):
    return np.corrcoef(x, y)[0, 1]


def spearman_rho(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return np.corrcoef(rx, ry)[0, 1]


def iqr_outlier_mask(values, k=1.5):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (values < q1 - k * iqr) | (values > q3 + k * iqr)


def find_outliers(hours, scores):
    slope, intercept = np.polyfit(hours, scores, 1)
    residuals = scores - (slope * hours + intercept)
    return (iqr_outlier_mask(hours)
            | iqr_outlier_mask(scores)
            | iqr_outlier_mask(residuals))


def association_summary(hours, scores):
    slope, intercept = np.polyfit(hours, scores, 1)
    return {"n": len(hours), "slope": slope, "intercept": intercept,
            "pearson": pearson_r(hours, scores), "spearman": spearman_rho(hours, scores)}


def course_summary(df):
    summary = df.groupby("course").agg(
        students=("student_id", "count"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        std_score=("score", "std"),
        mean_hours=("study_hours", "mean"),
    ).round(2)

    summary["r_all"] = pd.Series(
        {course: g["study_hours"].corr(g["score"]) for course, g in df.groupby("course")}
    ).round(2)

    summary["r_no_outliers"] = pd.Series(
        {course: g.loc[~g["outlier"], "study_hours"].corr(g.loc[~g["outlier"], "score"])
         for course, g in df.groupby("course")}
    ).round(2)
    return summary


def grade_distribution(df):
    table = pd.crosstab(df["course"], df["grade"])
    return table.reindex(columns=GRADE_ORDER, fill_value=0)


def draw_dashboard(df, outlier_mask, all_fit, clean_fit, summary, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_scatter, ax_hist, ax_course_hist, ax_bar = axes.ravel()

    hours = df["study_hours"].to_numpy()
    xs = np.linspace(hours.min(), hours.max(), 100)

    for course, g in df.groupby("course"):
        ax_scatter.scatter(g["study_hours"], g["score"], alpha=0.7, label=course)
    out = df[outlier_mask]
    ax_scatter.scatter(out["study_hours"], out["score"], s=160, facecolors="none",
                       edgecolors="red", linewidths=1.8, label="Outlier")
    ax_scatter.plot(xs, all_fit["slope"] * xs + all_fit["intercept"], "k--",
                    label=f"All data (r={all_fit['pearson']:.2f})")

    clean_hours = hours[~outlier_mask]
    xs_clean = np.linspace(clean_hours.min(), clean_hours.max(), 100)
    ax_scatter.plot(xs_clean, clean_fit["slope"] * xs_clean + clean_fit["intercept"], "g-",
                    label=f"Outliers removed (r={clean_fit['pearson']:.2f})")
    ax_scatter.set(xlabel="Reported study hours per week", ylabel="Score",
                   title="Study hours vs score", ylim=(0, 105))
    ax_scatter.legend(fontsize=8)

    ax_hist.hist(df["score"], bins=15, edgecolor="black", alpha=0.8)
    ax_hist.axvline(df["score"].mean(), color="red", linestyle="--", label="Mean")
    ax_hist.axvline(df["score"].median(), color="green", linestyle=":", label="Median")
    ax_hist.set(xlabel="Score", ylabel="Number of students", title="Score distribution")
    ax_hist.legend()

    bins = np.linspace(0, 100, 21)
    for course, g in df.groupby("course"):
        ax_course_hist.hist(g["score"], bins=bins, alpha=0.5, label=course)
    ax_course_hist.set(xlabel="Score", ylabel="Number of students",
                       title="Score distribution by course")
    ax_course_hist.legend()

    ax_bar.bar(summary.index, summary["mean_score"], yerr=summary["std_score"],
               capsize=5, alpha=0.8)
    ax_bar.set(xlabel="Course", ylabel="Mean score", title="Mean score by course (± 1 SD)")

    fig.suptitle("Student performance and study-hours dashboard", fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved dashboard figure: {save_path}")
    return fig


def evaluate_association(all_fit, clean_fit, n_outliers):
    same_sign = np.sign(all_fit["pearson"]) == np.sign(clean_fit["pearson"])
    change = abs(all_fit["pearson"] - clean_fit["pearson"])
    still_meaningful = abs(clean_fit["pearson"]) >= 0.3
    meaningful_with_all = abs(all_fit["pearson"]) >= 0.3

    print("\n=== ANALYSIS AND EVALUATION ===")
    print(f"Outliers flagged: {n_outliers} of {all_fit['n']} students")
    print(f"Pearson r  : all = {all_fit['pearson']:.2f}  |  without outliers = {clean_fit['pearson']:.2f}")
    print(f"Spearman   : all = {all_fit['spearman']:.2f}  |  without outliers = {clean_fit['spearman']:.2f}")
    print(f"Slope (points per extra weekly hour): all = {all_fit['slope']:.2f}  |  "
          f"without outliers = {clean_fit['slope']:.2f}")

    if same_sign and still_meaningful and change < 0.2:
        print("Verdict: the association SURVIVES outliers (same direction, similar strength).")
    elif same_sign and still_meaningful:
        print("Verdict: the association survives in direction, but its strength depends noticeably "
              "on the outliers.")
    elif still_meaningful and not meaningful_with_all:

        print("Verdict: the outliers HIDE an association. It is weak in the full data but clear "
              "once they are removed.")
        print("         The rank-based Spearman value (less sensitive to extremes) points the same way.")
        print("         Report both results and investigate the outliers instead of silently dropping them.")
    elif meaningful_with_all and not still_meaningful:
        print("Verdict: the association is DRIVEN by the outliers and disappears without them.")
    else:
        print("Verdict: the association is WEAK or NOT ROBUST.")

    print("\nCaution - correlation is not causation:")
    print("  * Study hours are SELF-REPORTED and may be inaccurate.")
    print("  * Other factors (prior knowledge, course difficulty, motivation, sleep) could drive both")
    print("    hours and scores; these data cannot separate them.")
    print("  * Courses differ in average score, so check the within-course correlations too.")
    print("  * The data show that the two variables move together, NOT that more study causes")
    print("    higher scores.")


def main():
    parser = argparse.ArgumentParser(description="Student performance and study-hours dashboard")
    parser.add_argument("--csv", default=CSV_FILE, help="path to the student CSV file")
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        generate_sample_csv(args.csv)

    records = load_records(args.csv)
    by_id = records_to_dict(records)
    print(f"Loaded {len(records)} valid records. Example: {records[0]}")
    first_id = records[0].student_id
    print(f"Dictionary look-up for {first_id}: {by_id[first_id]}")

    df = pd.DataFrame([r.to_dict() for r in records])

    hours = df["study_hours"].to_numpy()
    scores = df["score"].to_numpy()
    print("\n=== NumPy summary statistics ===")
    for name, arr in [("Study hours", hours), ("Score", scores)]:
        stats = describe_array(arr)
        print(f"{name:12s}: " + ", ".join(f"{k}={v:.2f}" for k, v in stats.items()))

    outlier_mask = find_outliers(hours, scores)
    df["outlier"] = outlier_mask

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    summary = course_summary(df)
    print("\n=== Summary by course ===")
    print(summary)
    print("\n=== Grade distribution by course ===")
    print(grade_distribution(df))

    all_fit = association_summary(hours, scores)
    clean_fit = association_summary(hours[~outlier_mask], scores[~outlier_mask])
    print("\nOutlier rows:")
    print(df[outlier_mask][["student_id", "course", "study_hours", "score", "grade"]].to_string(index=False))

    draw_dashboard(df, outlier_mask, all_fit, clean_fit, summary, DASHBOARD_FILE)

    evaluate_association(all_fit, clean_fit, int(outlier_mask.sum()))

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
