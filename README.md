# 🛡️ AI Claims Assessments System

An AI-powered multi-agent system designed to assist insurance adjusters by automatically processing, evaluating, and summarizing insurance claims safely and accurately.

---

## 📌 Project Overview

This project automates the initial workflow of handling insurance claims using Gemini AI models and Python safety rules. The system processes claim descriptions written in natural language, extracts structured facts, and runs checks across policies, past history, and required documents.

### Key Highlights
* **Human-in-the-Loop:** The AI never approves or rejects claims automatically—it prepares clear, factual summaries to help human adjusters decide.
* **Deterministic Guardrails:** Built-in safety filters block prompt injection attacks, attempted claim manipulation, and unverified outcome promises.
* **Smart Escalation:** Claims involving high monetary values, inactive policies, or suspicious patterns are automatically flagged and routed for manual human review.

---

## 🏗️ How It Works

1. **Intake & Validation:** Receives claim text and extracts key facts (type, incident date, claimed amount, description).
2. **Safety Check & Risk Routing:** Checks for prompt injections and flags high-risk claims (e.g., policy expired, large amount) for human review.
3. **Parallel Evidence Workers:**
   * **Policy Worker:** Verifies coverage limits, deductibles, and policy status.
   * **History Worker:** Checks previous claims to spot suspicious frequency patterns.
   * **Docs Worker:** Compares submitted documents against required documents.
4. **Drafting & Revision Loop:** Generates a structured summary for the adjuster, reviewing and refining the draft to ensure accuracy.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
* **Python 3.10+**
* A valid **Gemini API Key**

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/USRAZz/AI-Claims-Assessments--Depi.git](https://github.com/USRAZz/AI-Claims-Assessments--Depi.git)
   cd AI-Claims-Assessments--Depi
