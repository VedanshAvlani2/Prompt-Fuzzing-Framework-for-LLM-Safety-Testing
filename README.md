# 🧠 Prompt Fuzzing Framework
*A modular system to stress-test LLM safety and alignment.*

---

## 📘 Overview
The **Prompt Fuzzing Framework** is designed to automatically **mutate, test, and analyze** prompts against large language models (LLMs) to uncover unsafe, misaligned, or policy-violating behaviors.  
It provides a **secure sandbox**, **automated detectors**, and a **triage interface** to benchmark and visualize vulnerabilities across models.

---

## 🎯 Objectives
- Generate diverse, mutated prompts to test model robustness.  
- Detect jailbreaks, bias, sensitive data leaks, and refusal bypasses.  
- Label outputs as **Safe**, **Suspicious**, or **Unsafe** with severity levels.  
- Benchmark and compare model safety across **open-source** and **API-based** LLMs.  
- Ensure **reproducibility** and **auditability** with clear documentation.

---

## 🚀 Key Deliverables
1. **Attack Engine (CLI + Python Library)**  
   - Modular components: mutators, adapters, detectors.  
   - Handles orchestration, rate limiting, and logging.

2. **Benchmark Results**  
   - Runs on ≥2 open-source models (e.g., *LLaMA, Mistral*) and 1 API model (if available).  
   - Produces structured **CSV/JSON artifacts** for analysis.

3. **Minimal HTML Triage UI**  
   - Displays labeled unsafe outputs for human review and analysis.  
   - Includes filtering, search, and export capabilities.

4. **Documentation Package**  
   - `README.md`, `threat_model.md`, `redaction_policy.md`, and `reproduce_steps.md`.

5. **Labeled Evaluation Slice**  
   - Manually annotated subset to evaluate detection accuracy (precision/recall).

---

## 🏗️ System Architecture
Seed Corpus → Mutation Engine → Attack Engine (Sandboxed) → Detector & Scorer →
Triage & Deduplication → Benchmark Reporter (HTML/CSV/JSON)


---

## 🧩 Components
| Module | Description |
|---------|-------------|
| **Engine/** | Core orchestrator that manages runs, logging, and model interfaces. |
| **Mutators/** | Paraphrasing, insertion, and obfuscation-based prompt mutation scripts. |
| **Detectors/** | Rule-based + ML-based classifiers for unsafe or misaligned outputs. |
| **Adapters/** | Connectors for OSS (LLaMA/Mistral) and API-based LLMs. |
| **Sandbox/** | Isolated environment to safely run and log model responses. |
| **Triage/** | Deduplication and HTML report generator for flagged results. |
| **Data/** | Seed prompts, run logs, and labeled evaluation slices. |

---

## 🧰 Tech Stack
- **Python 3.11+**
- **LM Studio** for sandbox isolation  
- **FastAPI / Flask** for sandbox runner  
- **Hugging Face Transformers** for OSS models  
- **scikit-learn / PyTorch** for lightweight classifiers  
- **HTML / Plotly / Bootstrap** for reporting UI

---

## 📅 Project Timeline (6 Weeks)
| Week | Focus Area | Deliverables |
|------|-------------|--------------|
| **1** | Repo setup, CLI, sandbox skeleton | Initial structure, mock runs |
| **2** | Mutation engine + rule detectors | Basic fuzzing pipeline |
| **3** | Orchestrator + 1 OSS model | First benchmark results |
| **4** | 2nd model + ML detector | Combined scoring + labeled slice |
| **5** | HTML triage UI + benchmark runs | Reports + redaction logic |
| **6** | Docs + reproducibility | Final artifacts & evaluation |

---

## 👥 Team & Roles
| Member | Role |
|--------|------|
| **Vedansh** | Project Lead & Attack Engine Developer |
| **Rahul** | Sandbox & Model Integration Specialist |
| **Arbbaz** | Mutation & Detection Engineer |
| **Shivani** | Triage UI & Documentation Lead |
| **Aravind** | Benchmark Analysis & QA |

---

## 🧪 Evaluation Criteria
| Category | Weight | Description |
|-----------|---------|-------------|
| Framework Quality | 40% | Clean, modular, and reproducible system design |
| Attack Efficacy | 30% | % of unsafe behaviors discovered per model |
| Detection Accuracy | 20% | Precision/recall on labeled evaluation slice |
| Reporting & Docs | 10% | Clarity, usability, and reproducibility |

---

## 🛡️ Safety & Redaction
- All model interactions run **inside the sandbox**.  
- Logs and outputs undergo **regex-based redaction** for PII and API keys.  
- A **threat model** defines boundaries for safe testing and model isolation.

---

## ⚙️ Setup & Run
```bash
# 1️⃣ Clone the repo
git clone https://github.com/<org>/prompt-fuzzing-framework.git
cd prompt-fuzzing-framework

# 2️⃣ Build the sandbox
docker build -t prompt-sandbox .

# 3️⃣ Run tests or experiments
python cli.py --mode sandbox --model llama --seeds data/seeds.csv

# 4️⃣ Generate report
python triage/report_generator.py --input data/results/ --output reports/
```

---

## 🧾 License
For academic and research use only.
Do not use for real-world attack deployment or unauthorized testing of external APIs.
