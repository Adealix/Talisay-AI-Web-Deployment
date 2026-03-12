# PAPER AUDIT REPORT
### TalisayAI Research Paper vs. Actual System Implementation
**Audited by:** GitHub Copilot  
**Date:** March 10, 2026  
**Paper:** Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (Terminalia catappa) Fruits using Morphological Feature Analysis

---

## TABLE OF CONTENTS
1. [Critical Errors (Must Fix)](#critical-errors)
2. [Moderate Errors](#moderate-errors)
3. [Minor Issues](#minor-issues)
4. [How the System Measures Seed-to-Oil Ratio](#how-does-the-system-measure-seed-to-oil-ratio)
5. [Summary Table](#summary-of-all-issues-found)

---

## CRITICAL ERRORS
> These are factual errors that a professor or reviewer will immediately catch. Fix before submission.

---

### ERROR #1 — Wrong Machine Learning Models (Appears Throughout Entire Paper)

**Severity:** 🔴 CRITICAL  
**Locations:** Chapter 1 (Introduction, Scope), Chapter 3 (DFD Level 2), Definition of Terms, everywhere

**What the paper currently says:**
> "Machine Learning models including **Random Forest, Support Vector Regression, and Linear Regression**"

**What the system actually uses** (`ml/models/oil_yield_predictor.py`):
```python
self.rf_model = RandomForestRegressor(...)      # Random Forest ✓
self.gb_model = GradientBoostingRegressor(...)  # Gradient Boosting ✓
# SVR and Linear Regression are NOT in the codebase at all
```

**Fix:** Replace every occurrence of "Support Vector Regression" and "Linear Regression" with **"Gradient Boosting"**.  
The correct phrasing is: *"Machine Learning models including Random Forest and Gradient Boosting"*

---

### ERROR #2 — "SnapShroom System" Copy-Paste Error in DFD Section

**Severity:** 🔴 CRITICAL  
**Location:** Data Flow Diagram (DFD) introductory paragraph

**What the paper currently says:**
> "The Data Flow Diagram (DFD) is a graphical illustration of flow of information in the **SnapShroom system** to show how..."

**What it should say:**
> "The Data Flow Diagram (DFD) is a graphical illustration of flow of information in the **Talisay Oil Yield Prediction system** to show how..."

**Note:** This is leftover text from a different project (a mushroom identification system). It must be corrected — it directly contradicts the paper's title.

---

### ERROR #3 — Figure 9 Use Case Diagram Has Wrong Project Title

**Severity:** 🔴 CRITICAL  
**Location:** Figure 9 caption

**What the paper currently says:**
> "Figure 9: User Case Diagram: **A Machine Learning–Based Mushroom Identification and Edibility Detection System Using Image Processing**"

**What it should say:**
> "Figure 9: Use Case Diagram: **Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (Terminalia catappa) Fruits using Morphological Feature Analysis**"

**Note:** Same copy-paste error as #2. This is the most visible mistake — it references a completely different system in a diagram caption.

---

### ERROR #4 — Figures 6 and 7 Are Both Labeled "Level 0" in Captions

**Severity:** 🔴 CRITICAL  
**Location:** Figure 6 and Figure 7 captions

**What the paper currently says:**
| Figure | Current Caption Label | Correct Label |
|--------|-----------------------|---------------|
| Figure 5 | "DFD - Level 0" | ✅ Correct |
| Figure 6 | "DFD - **Level 0**" | ❌ Should be **Level 1** |
| Figure 7 | "DFD - **Level 0**" | ❌ Should be **Level 2** |

**Fix:** Update Figure 6 caption to say "Level 1" and Figure 7 caption to say "Level 2".

---

### ERROR #5 — ERD Claims "Relational Database" — System Uses MongoDB (NoSQL)

**Severity:** 🔴 CRITICAL / 🟠 HIGH  
**Location:** ERD section description

**What the paper currently says:**
> "The system **relational database structure** used to design the database..."  
> "...the diagram is divided into seven linked and interrelated **tables**..."

**What the system actually uses:** MongoDB via Mongoose — a **document-based NoSQL** database.  
MongoDB does **not** use "tables"; it uses **collections** and **documents**.

**Sub-issue — ERD model count is wrong:**

| ERD in Paper (7 "tables") | Actual MongoDB Collections (5) |
|----------------------------|---------------------------------|
| ACCOUNT | → merged into **User** |
| AUTHSESSION | → merged into **User** (JWT-based, no session table) |
| USERPROFILE | → merged into **User** |
| SCANSESSION | → merged into **History** |
| SCANIMAGE | → merged into **History** |
| MLANALYSISRESULT | → merged into **History** / **Prediction** |
| POST | → **ForumPost** |
| COMMENT | → embedded in **ForumPost** (subdocument) |
| *(not mentioned)* | **Notification** (exists in codebase) |

**Fix:** 
- Change "relational database structure" → "document-based NoSQL database structure"
- Change "tables" → "collections"
- Update ERD to reflect the 5 actual collections: **User, Prediction, History, ForumPost, Notification**

---

## MODERATE ERRORS
> These need to be corrected but won't immediately fail a submission.

---

### ERROR #6 — Oil Yield Range Mismatch in Definition of Terms

**Severity:** 🟠 MODERATE  
**Location:** Definition of Terms

**What the paper currently says:**
> "Oil Yield – the percentage of oil extracted from Talisay seeds, ranging from **48.8% to 53.4%** in Philippine studies"

**What the system actually uses** (`ml/config.py`):
```python
OIL_YIELD_BY_COLOR = {
    "green":  {"mean": 47.0, "min": 45.0, "max": 49.0},
    "yellow": {"mean": 58.5, "min": 57.0, "max": 60.0},  # Up to 60%!
    "brown":  {"mean": 55.5, "min": 54.0, "max": 57.0},
}
# Formula clips final output to range [45%, 65%]
```

**Fix:** Update the definition to:  
> "Oil Yield – the percentage of oil extracted from Talisay seeds, ranging from **45% to 65%** depending on fruit maturity, with Philippine varieties typically at 48.8%–53.4%."

---

### ERROR #7 — Duplicate Literature Entries in Chapter 2

**Severity:** 🟠 MODERATE  
**Location:** Chapter 2, Local Literature section

**Duplicate pair #1:**
- **Entry #37** and **Entry #38** are word-for-word the **same study**:  
  *"Regional Prediction of Crop Yield Success Rate in the Philippines Using Geographic Trend Analysis Algorithm (2023)"* — same title, same authors (Esguerra & Reyes), nearly identical discussion text.  
  ✅ **Action:** Remove one entry entirely.

**Duplicate pair #2:**
- **Entry #31** and **Entry #32** both describe the **same paper**:  
  Abaya (2022), *"Development of Talisay (Terminalia catappa) Butter"*  
  ✅ **Action:** Remove one entry entirely.

---

### ERROR #8 — Duplicate Entries in the References Section

**Severity:** 🟠 MODERATE  
**Location:** References list at the end of the paper

| Duplicate Group | References | Same Study |
|-----------------|------------|------------|
| Group A | #22 and #29 | Both: Orillaneda et al. (2022) Phytochemical Screening |
| Group B | #27, #31, and #50 | All: Esguerra & Reyes (2023) Regional Prediction |
| Group C | #15 and #23 | Both: Alingal & Gibertas (2023) Cookies study |
| Group D | #34 and #9 (Akolade) | Same Akolade et al. 2024 characterization study |

✅ **Action:** Consolidate each group into a single reference entry and renumber accordingly.

---

## MINOR ISSUES
> These are worth fixing for completeness, especially if the paper will be reviewed for technical accuracy.

---

### ISSUE #9 — Missing Technologies in Scope and Tech Stack Description

**Severity:** 🟡 LOW  
**Location:** Scope section (Chapter 1), Research Instrument (Chapter 3)

The paper only mentions Python, Scikit-learn, and TensorFlow. The actual system is a full-stack application with components **not mentioned anywhere in the paper**:

| Technology | Where It's Used | In Paper? |
|------------|-----------------|-----------|
| Node.js / Express | Backend API server (`server/index.js`) | ❌ Not mentioned |
| React Native (Expo) | Mobile frontend (`package.json`) | ❌ Not mentioned |
| MongoDB / Mongoose | Database | Only implied |
| Google Gemini AI | AI Chatbot (`services/geminiService.js`) | ❌ Not mentioned |
| Chatbot feature | `app/chatbot.js` | ❌ Not mentioned |
| Forum/Community feature | `app/forum.js` | ❌ Not mentioned |

✅ **Action:** Add a brief mention of Node.js for the backend API, React Native/Expo for the mobile app, and note the Chatbot and Forum as additional system features in the Scope section.

---

### ISSUE #10 — Paper Mentions EfficientNetB0 for Fruit Detection

**Severity:** 🟡 LOW  
**Location:** Detection Model Configuration in config description / paper methodology

**What the paper implies:** EfficientNetB0 is used for fruit detection.

**What the system actually uses:**
- `ml/models/yolo_detector.py` → **YOLOv8** for object detection (coin + fruit bounding boxes)
- `ml/models/talisay_guard.py` → **Custom CNN guard** (multi-layer validation)
- EfficientNetB0 is referenced only in `config.py` as a config comment — it is **not the active model**

✅ **Action:** Update any reference to the detection architecture to say **YOLOv8** (for object detection) and **custom CNN** (for fruit identity validation).

---

### ISSUE #11 — Literature Entry #10 Describes the Journal, Not a Study

**Severity:** 🟡 LOW  
**Location:** Chapter 2, Foreign Literature, Entry #10

**What the paper currently says:**  
Entry #10 is an entry about the journal *Grasas y Aceites* — describing its history, indexing (JCR, Scopus), and peer-review standards — **not** a study about Talisay or oilseeds.

✅ **Action:** Either:
- Replace it with an **actual published study** from that journal about Terminalia catappa oil, OR
- Remove the entry and renumber subsequent entries

---

## HOW DOES THE SYSTEM MEASURE SEED-TO-OIL RATIO FROM AN IMAGE?

This is the core technical question. Here is exactly how the pipeline works and how it should be validated.

---

### Step-by-Step Prediction Pipeline

The system uses a multi-step pipeline defined in `ml/predict.py` and `ml/models/oil_yield_predictor.py`:

---

#### STEP 1 — Color Classification (Maturity Detection)

The image is classified as **Green / Yellow / Brown** using a CNN (MobileNetV2-based deep learning, or HSV-based fallback). Fruit maturity is the **dominant predictor** — it accounts for the largest portion of the oil yield estimate:

| Color | Maturity | Mean Oil Yield | Range |
|-------|----------|----------------|-------|
| 🟢 Green | Immature | ~47.0% | 45–49% |
| 🟡 Yellow | Mature (peak) | ~58.5% | 57–60% |
| 🟤 Brown | Overripe | ~55.5% | 54–57% |

---

#### STEP 2 — Dimension Estimation (Physical Measurements)

A **₱5 coin** placed beside the fruit serves as a scale reference. YOLOv8 detects both the coin and the fruit bounding box, then converts pixel measurements to real-world centimeters:

- `length_cm` — fruit length from bounding box
- `width_cm` — fruit width from bounding box
- `estimated_kernel_mass_g` — calculated from a volume proxy formula
- `estimated_whole_fruit_weight_g` — estimated from dimensions

---

#### STEP 3 — Oil Yield Prediction Formula / ML Model

If the trained `.joblib` ensemble model (Random Forest + Gradient Boosting) is available, it predicts using all 7 features.

If the model is not loaded, the **formula-based fallback** is used:

```
OilYield = BaseYield(color) + KernelAdj + LengthAdj + AspectAdj + WeightAdj
```

Where each adjustment is:

| Adjustment | Formula | Max Effect |
|------------|---------|------------|
| KernelAdj | `(kernelNorm - 0.5) × 4.0` | ±4% |
| LengthAdj | `(length - 5.0) / 2.5 × 2.5` | ±2.5% |
| AspectAdj | `(L/W - 1.43) × 1.5` | ±2% |
| WeightAdj | `(weight - 35.0) / 20.0 × 1.5` | ±1.5% |

**Final yield is clipped to the range: [45%, 65%]**

---

### How to Verify Accuracy with Real-Life Laboratory Testing

This is called **ground-truth comparison** — the gold standard for validating a predictive model.

---

#### Laboratory Method: Soxhlet Extraction

1. **Weigh** the intact Talisay fruit → `W_fruit` (whole fruit weight in grams)
2. **Shell** the fruit, remove the kernel. Weigh the dry kernel → `W_kernel`
3. **Extract** oil using Soxhlet apparatus with hexane or petroleum ether solvent (6–8 hours)
4. **Evaporate** the solvent completely. Weigh the extracted oil → `W_oil`
5. **Calculate** the true oil yield:

```
OilYield% = (W_oil / W_kernel) × 100
```

---

#### Validation Protocol

1. Collect **20–30 Talisay fruits** with varied maturities (green, yellow, and brown)
2. **Before** extracting, photograph each fruit beside a ₱5 coin and run it through the system → record **predicted oil yield %**
3. **After** extraction in the lab, compute the **actual oil yield %**
4. Compare predicted vs. actual using these metrics:

```
MAE  = (1/n) × Σ|Predicted_i - Actual_i|

RMSE = √[(1/n) × Σ(Predicted_i - Actual_i)²]

R²   = 1 - [Σ(Actual_i - Predicted_i)²] / [Σ(Actual_i - Mean_actual)²]
```

5. **Target accuracy benchmarks** (based on your cited literature):

| Metric | Target | Reference |
|--------|--------|-----------|
| MAE | < 3% | Agu et al. (2022) |
| RMSE | < 4% | Torres et al. (2024) |
| R² | > 0.80 | Jamshidi et al. (2024) achieved R²=0.99 |

---

#### ⚠️ Important Gap for the Paper

> Currently, the oil yield base values in `ml/config.py` are calibrated from **international literature** (Senegal, Vietnam, Indonesia), **not** from actual Philippine Talisay laboratory extractions.
>
> For a scientifically complete paper, the system needs to be trained and validated on **real extraction data from Philippine Talisay samples**.  
> This is **the single most important gap** between the system's claims and what can be proven.
>
> **Recommended action:** Conduct laboratory Soxhlet extraction on at least 20–30 locally collected Talisay fruits, record the results, retrain the model, and report the actual MAE and RMSE values.

---

## SUMMARY OF ALL ISSUES FOUND

| # | Severity | Location in Paper | Issue | Action Required |
|---|----------|-------------------|-------|-----------------|
| 1 | 🔴 CRITICAL | Ch.1, Ch.3, DFD L2, everywhere | ML models listed as SVR & Linear Regression — system actually uses **Gradient Boosting** | Replace all occurrences |
| 2 | 🔴 CRITICAL | DFD section intro | Says "**SnapShroom system**" — wrong project name | Replace with "Talisay Oil Yield Prediction system" |
| 3 | 🔴 CRITICAL | Figure 9 caption | Says "**Mushroom Identification**" — wrong project title | Replace with correct paper title |
| 4 | 🔴 CRITICAL | Figures 6 & 7 captions | Both labeled "**Level 0**" — should be Level 1 and Level 2 | Fix caption numbering |
| 5 | 🔴 HIGH | ERD section | Says "**relational database**" with 7 tables — system uses **MongoDB NoSQL** with 5 collections | Rewrite ERD description and update diagram |
| 6 | 🟠 MODERATE | Definition of Terms | Oil yield range 48.8–53.4% is too narrow — system calibrated to **45–65%** | Clarify the full range with maturity breakdown |
| 7 | 🟠 MODERATE | Ch. 2 Lit entries #37/#38 | **Exact duplicate** literature entry (Esguerra & Reyes 2023) | Remove one entry |
| 8 | 🟠 MODERATE | Ch. 2 Lit entries #31/#32 | **Same Talisay Butter study** cited twice (Abaya 2022) | Remove one entry |
| 9 | 🟠 MODERATE | References list | **Multiple duplicate references** (groups: #22/#29, #27/#31/#50, #15/#23) | Consolidate and renumber |
| 10 | 🟡 LOW | Scope / Tech Stack | **Node.js, Expo/React Native, Gemini AI chatbot, Forum** not mentioned at all | Add to Scope section |
| 11 | 🟡 LOW | Detection model description | Paper implies EfficientNetB0 — system actually uses **YOLOv8 + custom CNN** | Update architecture reference |
| 12 | 🟡 LOW | Ch. 2 Lit entry #10 | Describes the **journal Grasas y Aceites** itself, not an actual Talisay study | Replace with a real study or remove |

---

### Priority Fix Order

```
Fix #1, #2, #3, #4 first  → These are immediately visible factual errors
Fix #5 next               → ERD mismatch affects technical credibility
Fix #6, #7, #8, #9 after  → Literature cleanup
Fix #10, #11, #12 last    → Minor completeness issues
```

---

*Report generated by GitHub Copilot — based on full audit of TalisayAI.txt draft paper against the live system codebase (ml/, server/, app/).*
