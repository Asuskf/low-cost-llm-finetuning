Markdown
# Low-Cost LLM Fine-Tuning & Inference Labs 🚀

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![MLOps](https://img.shields.io/badge/MLOps-Ready-brightgreen)

This repository contains hands-on laboratories designed to democratize the deployment and specialization of Large Language Models (LLMs). The primary focus is executing and training billion-parameter models in resource-constrained environments, such as the Google Colab free tier (NVIDIA T4 GPU).

## 🧪 Laboratory Contents

The project is structured around two critical phases of the AI model lifecycle:

### Lab 01: Standard LLM vs. AirLLM (Optimized Inference) 💻
This lab focuses on memory efficiency during the deployment phase.
* **Objective:** Compare the performance and resource consumption when loading a 7B parameter model (`Platypus2-7B`).
* **Technologies:** `transformers` vs. `airllm`.
* **Technical Highlight:** An architectural evaluation demonstrating how `airllm` enables the execution of models that would typically exceed a T4's VRAM by utilizing layered loading and low-memory inference optimization.

### Lab 02: Fine-Tuning Gemma with QLoRA (4-bit) 🧠
A complete Supervised Fine-Tuning (SFT) workflow for model specialization.
* **Objective:** Fine-tune the `Gemma-4-E2B-it` model for specific tasks using Parameter-Efficient Fine-Tuning (PEFT) techniques.
* **Technologies:** PEFT, LoRA, `bitsandbytes` (4-bit quantization).
* **Technical Highlight:** Implements QLoRA to drastically reduce memory footprint during training without significantly sacrificing model accuracy. It includes a complete workflow for persisting the model checkpoints and tokenizer to Google Drive.

## 🏗 Architecture & Methodology

To ensure these labs translate effectively into production and **MLOps** environments, the repository is built on the following pillars:

1. **Extreme Quantization:** Utilizing 4-bit precision to allow cutting-edge models to run on consumer hardware.
2. **Low-Rank Adaptation (LoRA):** Training only a small set of external adapter weights to prevent catastrophic forgetting and enable efficient model versioning.
3. **Data Quality & Governance:** Emphasizing the importance of structured datasets for training (aligned with **DAMA** lifecycle principles) to ensure high-quality outputs.
4. **Advanced RAG Integration:** The fine-tuned models are optimized to serve as the reasoning engine within Retrieval-Augmented Generation architectures.

## ⚙️ Environment Setup

Each notebook is strictly designed to be executed on **Google Colab** with an NVIDIA T4 GPU.

1. **Dependencies:**
   ```bash
   pip install -q -U transformers peft accelerate bitsandbytes trl airllm
   ```
2. * **Model Access:** Ensure you have your Hugging Face token configured to access gated models like Gemma.
3. * **Storage:** Lab 02 requires mounting Google Drive to persistently save model adapters and checkpoints.

## 📂 Repository Structure

```text
low-cost-llm-finetuning/
├── lab_01_standard_llm_vs_airllm_fixed.ipynb  # Inference comparison and memory optimization
supervised-finetuning/
├── lab_02_gemma4_sft_fixed_2.ipynb           # QLoRA Fine-Tuning workflow
```
## 🤝 Contributions & Support

If you encounter performance issues or notice signs of model degradation (data drift) in your specific use cases, please open an issue. Building robust AI systems relies heavily on continuous evaluation and high data quality standards.

---
**Developed by [Kevin Farinango C.](https://www.linkedin.com/in/david-farinango/) ([@Asuskf](https://github.com/Asuskf))**
*Data Scientist & Machine Learning Engineer*
