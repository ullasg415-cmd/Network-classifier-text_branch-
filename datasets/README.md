# Datasets for the text branch

```
datasets/
├── real/                 <- labelled documents the model is TRAINED and TESTED on
│   ├── <category_a>/     one folder per category; the folder name IS the label
│   │   ├── file1.pdf
│   │   ├── file2.docx
│   │   └── file3.txt
│   └── <category_b>/ ...
└── ood/                  <- "out-of-domain": documents from categories the model is NOT trained on
    └── <anything>/       used only to measure and tune the "unknown" rejection
```

This folder is git-ignored: **documents are never committed**. Do not put confidential or classified
material here unless you have explicit authorisation to use it with this tool and on this machine.

## Rules for a dataset that gives trustworthy numbers

1. **Real documents, correct labels.** Labels in folder names are the ground truth; wrong labels cap accuracy.
2. **At least 12 documents per category** is the hard minimum (the trainer refuses less); 50+ per category is a sensible start, more is better. Keep categories roughly balanced.
3. **Formats:** `.pdf`, `.docx`, `.txt` (plus `.doc .rtf .odt .html` if a Tika server is running).
4. **Text PDFs only.** Scanned PDFs have no text layer and are skipped (OCR is not set up).
5. **Do not pre-split.** Put everything in the category folder. The trainer removes exact duplicates, groups
   near-duplicates (so versions of the same document never land on both sides of the split) and makes
   the train / validation / test split itself.
6. **Out-of-domain set (`ood/`):** 20+ documents from other kinds of document that the system might meet but
   that are NOT one of your categories (e.g. if your categories are invoice/contract/report, use
   brochures, manuals, letters). One sub-folder per kind, at least 2 documents each. Without this the
   "unknown" threshold is an untuned default.

## Steps

```bash
cd backend
python -m text.data.check_dataset text/datasets/real                # layout, counts, duplicates, problems
python -m text.data.check_dataset text/datasets/ood
python -m text.train --data-dir text/datasets/real --ood-dir text/datasets/ood --tune-threshold --data-source drdo-real
```

`--data-source` is recorded in the model metadata. Use a name that does NOT start with `synthetic` and does
not contain `demo` only for genuinely real data: that is what switches the model's "DEMO MODEL" flag off.
