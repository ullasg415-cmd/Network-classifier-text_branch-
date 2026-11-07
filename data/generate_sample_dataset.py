"""Generate a SYNTHETIC labelled document set (PDF / DOCX / TXT) for the text branch.

IMPORTANT: this is demo data written by templates, not real documents. A model
trained on it proves the pipeline works end to end; its scores say nothing
about how it will do on real DRDO documents. Replace it with real labelled
files in the same layout (one folder per category) and retrain:

    data/<name>/<category>/<any files .pdf .docx .txt>

Usage (from backend/):
    python -m text.data.generate_sample_dataset --out data/sample --per-class 60
    python -m text.data.generate_sample_dataset --out data/ood --ood --per-class 20
"""
from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path

CATEGORIES = [
    "invoice", "resume", "legal_contract", "research_paper", "news_article", "technical_manual",
]
# Categories the model is never trained on; used to measure how it reacts to unfamiliar documents.
OOD_CATEGORIES = ["recipe", "sports_report"]

FIRST = ["Aarav", "Meera", "Rohan", "Ananya", "Vikram", "Sneha", "Karthik", "Divya", "Arjun", "Priya", "Rahul", "Neha"]
LAST = ["Sharma", "Iyer", "Gowda", "Reddy", "Nair", "Kapoor", "Menon", "Patel", "Rao", "Das", "Bhat", "Joshi"]
COMPANY = ["Apex Systems", "Northwind Traders", "Bluepeak Logistics", "Orion Dynamics", "Kaveri Engineering",
           "Zenith Components", "Lakshya Infotech", "Meridian Supplies", "Sagar Industries", "Vertex Labs"]
CITY = ["Bengaluru", "Hyderabad", "Pune", "Chennai", "Delhi", "Mumbai", "Kochi", "Jaipur", "Lucknow", "Nagpur"]
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _date(r):
    return f"{r.randint(1, 28)} {r.choice(MONTH)} {r.randint(2019, 2026)}"


def _name(r):
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


# ---------------------------------------------------------------------------
# one function per category -> list of lines
# ---------------------------------------------------------------------------
def _invoice(r):
    items = ["steel brackets", "network switches", "printer toner", "server racks", "copper cabling", "laptop batteries",
             "safety helmets", "consulting hours", "cloud hosting", "maintenance contract", "office chairs", "fibre patch cords"]
    lines = [f"INVOICE No. INV-{r.randint(1000, 99999)}", f"Bill To: {r.choice(COMPANY)}, {r.choice(CITY)}",
             f"Invoice date: {_date(r)}", f"Purchase order reference: PO-{r.randint(100, 9999)}", "Description Qty Unit price Amount"]
    sub = 0
    for _ in range(r.randint(2, 6)):
        q, p = r.randint(1, 40), r.randint(50, 9000)
        sub += q * p
        lines.append(f"{r.choice(items)} {q} x Rs {p} = Rs {q * p}")
    rate = r.choice([5, 12, 18, 28])
    lines += [f"Subtotal: Rs {sub}", f"GST ({rate}%): Rs {sub * rate // 100}", f"Total amount due: Rs {sub + sub * rate // 100}",
              f"Payment terms: net {r.choice([15, 30, 45])} days from the invoice date. Late payment attracts interest.",
              "Please remit the balance by bank transfer quoting the invoice number. Remittance advice to accounts receivable."]
    if r.random() < 0.5:
        lines.append("Goods once sold will not be taken back. Subject to local jurisdiction. This is a computer generated invoice.")
    return lines


def _resume(r):
    skills = ["Python", "Java", "SQL", "Docker", "Kubernetes", "TensorFlow", "React", "Linux", "Git", "Spark", "Wireshark", "C++"]
    roles = ["software engineer", "data analyst", "network engineer", "machine learning engineer", "systems administrator", "research assistant"]
    nm = _name(r)
    lines = [nm, f"Email: {nm.split()[0].lower()}@example.com | Phone: +91 {r.randint(70000, 99999)}{r.randint(10000, 99999)} | {r.choice(CITY)}",
             f"Professional summary: motivated {r.choice(roles)} with {r.randint(1, 12)} years of experience in {r.choice(skills)} and {r.choice(skills)}. Seeking a challenging role.",
             "Work experience"]
    for _ in range(r.randint(1, 3)):
        y = r.randint(2014, 2024)
        lines.append(f"{r.choice(roles).title()}, {r.choice(COMPANY)}, {y} - {y + r.randint(1, 3)}")
        for _ in range(r.randint(2, 3)):
            lines.append(f"Developed and deployed {r.choice(['pipelines', 'dashboards', 'services', 'models', 'tools'])} using {r.choice(skills)}, "
                         f"improving {r.choice(['throughput', 'accuracy', 'uptime', 'turnaround time'])} by {r.randint(10, 70)}%.")
    lines += ["Education", f"B.Tech in {r.choice(['Computer Science', 'Electronics', 'Information Science'])}, {r.choice(CITY)} Institute of Technology, {r.randint(2012, 2025)}, CGPA {r.randint(65, 98) / 10}",
              f"Skills: {', '.join(r.sample(skills, 5))}", "Certifications and projects: references available upon request."]
    return lines


def _legal_contract(r):
    a, b = r.choice(COMPANY), r.choice(COMPANY)
    lines = [f"SERVICE AGREEMENT. This Agreement is made on {_date(r)} between {a} (the \"Company\") and {b} (the \"Contractor\").",
             "WHEREAS the Company wishes to engage the Contractor to provide services and the Contractor agrees to do so on the terms set out herein."]
    clauses = [
        "The Contractor shall indemnify and hold harmless the Company against all claims, liabilities, damages and expenses arising from any breach of this Agreement.",
        f"Either party may terminate this Agreement by giving {r.choice([15, 30, 60, 90])} days written notice to the other party.",
        "The Receiving Party shall keep all Confidential Information strictly confidential and shall not disclose it to any third party without prior written consent.",
        f"This Agreement shall be governed by and construed in accordance with the laws of India and the courts of {r.choice(CITY)} shall have exclusive jurisdiction.",
        "Any dispute arising out of this Agreement shall be referred to binding arbitration under the Arbitration and Conciliation Act.",
        "Neither party shall be liable for failure to perform its obligations due to force majeure events beyond its reasonable control.",
        "The Contractor warrants that the services shall be performed with reasonable skill and care and that all intellectual property rights vest in the Company.",
        "No amendment of this Agreement shall be effective unless in writing and signed by both parties. This clause survives termination.",
        f"The Company shall pay the Contractor the fees set out in Schedule {r.randint(1, 4)} within thirty days of receipt of a valid invoice.",
    ]
    for i, c in enumerate(r.sample(clauses, r.randint(5, 8)), 1):
        lines.append(f"{i}. {c}")
    lines.append("IN WITNESS WHEREOF the parties have executed this Agreement on the date first written above. Signed by the authorised signatories.")
    return lines


def _research_paper(r):
    methods = ["a transformer-based encoder", "a gradient boosted ensemble", "a convolutional network", "a graph neural network", "a self-supervised pre-training scheme", "a sparse linear model"]
    tasks = ["anomaly detection", "text classification", "traffic analysis", "image segmentation", "speech recognition", "time series forecasting"]
    data = ["CICIDS2017", "ImageNet", "the 20 Newsgroups corpus", "LibriSpeech", "a proprietary benchmark", "MNIST"]
    m, t = r.choice(methods), r.choice(tasks)
    lines = [f"Abstract. In this paper we propose {m} for {t}. Experiments on {r.choice(data)} show that our approach outperforms strong baselines.",
             f"1 Introduction. {t.capitalize()} remains a challenging problem. Prior work has largely relied on hand-crafted features; we instead learn representations end to end.",
             f"2 Related work. Earlier studies [1], [2] explored {r.choice(methods)}. Our work differs in its training objective and evaluation protocol.",
             f"3 Method. Let x denote the input and y the label. We minimise the cross-entropy loss with weight decay {r.choice(['1e-4', '5e-5', '1e-5'])} using the Adam optimiser.",
             f"4 Experiments. Table {r.randint(1, 4)} reports precision, recall and F1. We achieve an F1 of {r.randint(80, 99) / 100} compared with {r.randint(60, 85) / 100} for the baseline. An ablation study confirms each component contributes.",
             "5 Conclusion. We presented a novel method and showed empirical gains. Future work will study robustness and generalisation to unseen distributions.",
             "References. [1] A. Author et al., Proceedings of the International Conference on Machine Learning. [2] B. Writer, IEEE Transactions on Pattern Analysis."]
    return lines


def _news_article(r):
    events = ["announced a new infrastructure package", "confirmed that talks will resume next week", "said the airport expansion will open ahead of schedule",
              "reported a sharp rise in exports during the quarter", "warned of heavy rainfall across the coastal districts", "unveiled plans to modernise rural schools"]
    org = ["the state government", "the ministry", "the city corporation", "the central bank", "a spokesperson for the railway board"]
    lines = [f"{r.choice(CITY).upper()}, {_date(r)}: Officials said on {r.choice(WEEKDAY)} that {r.choice(org)} {r.choice(events)}.",
             f"\"We are committed to delivering this on time,\" said {_name(r)}, a senior official, told reporters at a press conference.",
             f"The announcement comes after weeks of criticism from opposition leaders, who accused the administration of delays. Residents of {r.choice(CITY)} welcomed the move.",
             f"Analysts expect the decision to affect markets, with the benchmark index closing {r.randint(1, 3)} per cent higher. Experts said the impact would be visible within months.",
             f"\"It is a step in the right direction,\" {_name(r)}, an economist, said. Further details are expected to be released later this week, according to sources."]
    return lines


def _technical_manual(r):
    dev = r.choice(["rack-mount server", "power supply unit", "radio transceiver", "air compressor", "network switch", "diesel generator"])
    lines = [f"{dev.title()} - Installation and Operation Manual, Revision {r.randint(1, 9)}.{r.randint(0, 9)}",
             f"Section {r.randint(1, 6)}: Installation. Step 1. Disconnect the power supply before opening the enclosure. Step 2. Mount the {dev} using the supplied brackets and torque the bolts to {r.randint(5, 30)} Nm.",
             f"WARNING: Do not operate the {dev} with the cover removed. Risk of electric shock. Use only the approved cable.",
             f"Specifications: input voltage {r.choice([110, 230, 415])} V, frequency {r.choice([50, 60])} Hz, operating temperature {r.randint(-10, 0)} to {r.randint(40, 60)} degrees Celsius, weight {r.randint(2, 80)} kg.",
             f"Troubleshooting: if the {r.choice(['status', 'fault', 'power'])} LED blinks red, check the connector, reset the unit and consult the error code table in Appendix {r.choice('ABCD')}.",
             f"Maintenance: inspect the filters every {r.randint(3, 12)} months and replace worn gaskets. Lubricate moving parts according to the maintenance schedule. Refer to the parts list for spare part numbers."]
    return lines


def _recipe(r):
    ing = ["flour", "sugar", "butter", "eggs", "milk", "salt", "garlic", "onions", "tomatoes", "basil", "rice", "lentils", "cumin", "ginger"]
    lines = [f"{r.choice(['Classic', 'Easy', 'Spicy', 'Homestyle'])} {r.choice(['pasta', 'dal', 'cake', 'curry', 'pancakes'])} recipe. Serves {r.randint(2, 8)}. Prep time {r.randint(10, 40)} minutes.",
             f"Ingredients: {r.randint(1, 3)} cups {r.choice(ing)}, {r.randint(1, 4)} tablespoons {r.choice(ing)}, {r.randint(1, 5)} cloves {r.choice(ing)}, a pinch of salt.",
             f"Method: Preheat the oven to {r.randint(160, 220)} degrees. Chop the {r.choice(ing)} and saute in a pan until golden. Stir in the {r.choice(ing)} and simmer for {r.randint(5, 30)} minutes.",
             "Season to taste, garnish with fresh herbs and serve warm."]
    return lines


def _sports_report(r):
    teams = ["Royal Strikers", "City United", "Northern Lions", "Coastal Rangers", "Capital Titans", "Valley Warriors"]
    a, b = r.sample(teams, 2)
    lines = [f"{a} beat {b} {r.randint(1, 5)}-{r.randint(0, 3)} on {r.choice(WEEKDAY)} night, with the striker scoring in the {r.randint(5, 89)}th minute.",
             f"The captain led from the front as the home side dominated possession. The goalkeeper made {r.randint(3, 9)} saves before the final whistle.",
             f"Coach {_name(r)} praised the team's defence after the league match. The victory lifts them to {r.randint(1, 6)}th place in the table.",
             "Fans packed the stadium for the tournament fixture; the next game is scheduled for the weekend."]
    return lines


GENERATORS = {
    "invoice": _invoice, "resume": _resume, "legal_contract": _legal_contract, "research_paper": _research_paper,
    "news_article": _news_article, "technical_manual": _technical_manual, "recipe": _recipe, "sports_report": _sports_report,
}
FILLER = ["Please find the details below.", "Kindly treat this as confidential.", "Thank you for your cooperation.",
          "Contact the office for any clarification.", "Page 1 of 2", "Reference number provided above.", "Regards."]


def make_document(category: str, rng: random.Random, noise: float = 0.25) -> str:
    """Build one document. `noise` = chance of mixing in lines from another category."""
    lines = GENERATORS[category](rng)
    if rng.random() < noise:  # cross-category contamination so classes are not trivially separable
        other = rng.choice([c for c in CATEGORIES if c != category])
        extra = GENERATORS[other](rng)
        for _ in range(rng.randint(1, 2)):
            lines.insert(rng.randrange(len(lines) + 1), rng.choice(extra))
    for _ in range(rng.randint(0, 2)):
        lines.insert(rng.randrange(len(lines) + 1), rng.choice(FILLER))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def _write_txt(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def _write_docx(path: Path, text: str):
    from docx import Document
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(str(path))


def _write_pdf(path: Path, text: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 50
    for line in text.split("\n"):
        for chunk in textwrap.wrap(line, 95) or [""]:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(40, y, chunk)
            y -= 14
    c.save()


def generate(out: Path, categories: list[str], per_class: int, seed: int) -> dict:
    rng = random.Random(seed)
    counts = {}
    try:
        import reportlab  # noqa: F401
        have_pdf = True
    except ImportError:
        have_pdf = False
        print("reportlab not installed: writing .txt/.docx only (pip install reportlab for PDFs)")
    for cat in categories:
        d = out / cat
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            text = make_document(cat, rng)
            kind = rng.choices(["txt", "pdf", "docx"], weights=[2, 1, 1])[0]
            if kind == "pdf" and not have_pdf:
                kind = "txt"
            path = d / f"{cat}_{i:04d}.{kind}"
            {"txt": _write_txt, "pdf": _write_pdf, "docx": _write_docx}[kind](path, text)
        counts[cat] = per_class
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ood", action="store_true", help="generate the out-of-domain categories (recipe, sports_report)")
    a = ap.parse_args()
    cats = OOD_CATEGORIES if a.ood else CATEGORIES
    print(generate(a.out, cats, a.per_class, a.seed))


if __name__ == "__main__":
    main()
