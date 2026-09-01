# DeepDomain

**Metamorphic Testing of Deep Neural Network-based Autonomous Driving Systems Using Behavioral Domain Adequacy**

DeepDomain is a **gray-box, multi-objective test generation approach** for testing deep neural network (DNN)-based autonomous driving systems. It adaptively selects diverse source inputs and generates **domain-oriented follow-up tests** to expose model misbehaviours.

> **Core idea:** DeepDomain guides metamorphic test generation using **behavioral domain adequacy**, rather than relying solely on conventional neural coverage criteria.

---

## 🔎 Overview

Testing DNN-based autonomous driving systems is challenging because the correct output for an arbitrary input is often difficult to determine. **Metamorphic Testing (MT)** addresses this oracle problem by generating follow-up inputs from source inputs according to predefined **metamorphic relations (MRs)**.

DeepDomain extends this idea by considering the **behavioral domains** explored during metamorphic test generation.

The approach combines:

* **Metamorphic Testing**
* **Search-based test generation**
* **Neuron contribution analysis**
* **Behavioral domain adequacy**
* **Multi-objective optimization**
* **Adaptive source-input selection**
* **Gray-box access to DNN internals**

DeepDomain is designed particularly for **DNN predictor models with continuous outputs**, such as steering-angle prediction in autonomous driving, rather than classification models.

---

## 💡 Behavioral Domain Adequacy

DeepDomain introduces two complementary dimensions for guiding test generation.

### Inter-behavioural Domain

Generated follow-up tests should explore behavioral regions that are **broader than those covered by their source tests**.

This encourages exploration of new behavioral regions rather than repeatedly generating tests with similar behavior.

### Intra-behavioural Domain

Once a misbehaviour-inducing region has been identified, generated tests should further explore its **neural boundary**.

This allows the testing process to investigate regions around previously detected misbehaviours more systematically.

Together, these objectives guide the search from:

```text
New Behavioral Regions
          │
          ▼
Misbehaviour-Inducing Regions
          │
          ▼
Their Behavioral Boundaries
```

---

## 🧠 Approach

At a high level, DeepDomain follows the workflow below:

```text
Source Input Selection
        │
        ▼
Metamorphic Relation
        │
        ▼
Follow-up Test Generation
        │
        ▼
Neuron Contribution Analysis
        │
        ▼
Multi-objective Search
   ┌────┴───────────────────┐
   │                        │
   ▼                        ▼
Inter-behavioural      Intra-behavioural
Domain Adequacy        Domain Adequacy
   │                        │
   └──────────┬─────────────┘
              ▼
       Candidate Tests
              │
              ▼
      Misbehaviour Detection
              │
              ▼
     Behavioral Domain Analysis
```

The detailed formulation of the objectives, neural pathway analysis, metamorphic relations, and search process is described in the accompanying paper and dissertation.

---

## 🎯 Key Contributions

### Behavioral-domain-guided test generation

DeepDomain uses behavioral-domain information to guide the search for effective test cases instead of treating neuron coverage as the primary objective.

### Critical neural pathway exploration

The approach extracts **critical neural pathways using neuron contribution** and incorporates them into follow-up test generation.

### Multi-objective search

DeepDomain formulates test generation as a multi-objective optimization problem, balancing complementary objectives related to behavioral-domain exploration.

### Adaptive source-input selection

Source inputs are selected adaptively to maintain diversity and improve exploration of the input and behavioral spaces.

### Gray-box metamorphic testing

DeepDomain combines metamorphic testing with internal DNN information to guide test generation while ultimately evaluating the observable behavior of the model.

---

## 🧪 Experimental Evaluation

DeepDomain was evaluated on:

| Component                 | Setting                                                  |
| ------------------------- | -------------------------------------------------------- |
| **DNN models**            | 3 autonomous-driving predictor models                    |
| **Application**           | Udacity Self-Driving Car Challenge                       |
| **Metamorphic Relations** | 18 MRs                                                   |
| **Search strategy**       | DNSGA-II                                                 |
| **Testing approach**      | Gray-box, search-based metamorphic testing               |
| **Primary focus**         | Misbehaviour detection and behavioral-domain exploration |

---

## 📊 Key Results

The empirical evaluation indicates that **behavioral domain adequacy is a more reliable indicator of test-generation effectiveness than conventional coverage criteria** for the evaluated DNN-based autonomous driving systems.

Compared with the evaluated baselines, DeepDomain achieved improvements of up to:

| Metric                                     |              Improvement |
| ------------------------------------------ | -----------------------: |
| **Misbehaviour detection**                 |                  **94×** |
| **Fault-revealing capability**             |                  **79%** |
| **Output diversity**                       |                  **71%** |
| **Corner-case detection**                  |                 **187×** |
| **MR robustness subdomain identification** | **33 percentage points** |
| **Naturalness**                            |                   **2×** |

The results further suggest that:

* High neural/structural coverage does not necessarily imply effective misbehaviour detection.
* **Black-box diversity-based** test generation can be less effective than the proposed gray-box approach.
* Behavioral-domain information can provide more useful guidance for exploring failure-prone regions of DNN behavior.

> **Main finding:** Effective DNN testing requires not only exploring internal model structure, but also adequately exploring the model's **behavioral domain**.

---

## 📁 Repository Structure

```text
DeepDomain/
│
├── constants.py
├── configs/
│
├── dataset/
│
├── models/
│
├── pathway_grad/
│
├── tools/
│   └── DEEPDOMAIN/
│       ├── deep_domain.py
│       └── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

### Main components

| Component           | Description                                  |
| ------------------- | -------------------------------------------- |
| `constants.py`      | Global project configurations and paths      |
| `configs/`          | Optional custom configuration                |
| `dataset/`          | Input datasets and labels                    |
| `models/`           | Trained DNN models and related scripts       |
| `pathway_grad/`     | Extraction of critical neural pathways       |
| `tools/DEEPDOMAIN/` | Main DeepDomain implementation               |
| `results/`          | Generated test data and experimental outputs |

---

## ⚙️ System Requirements

The original experiments were developed and tested with:

* **Operating System:** Ubuntu 18.04
* **Python:** 3.8
* **RAM:** 8 GB or higher, depending on model size
* **GPU:** Recommended for TensorFlow/PyTorch-based models

> The repository reflects the environment used for the original experiments. Newer Python, TensorFlow, or PyTorch versions may require additional compatibility adjustments.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR-USERNAME>/DeepDomain.git
cd DeepDomain
```

Create and activate a Python 3.8 environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running DeepDomain

### 1. Configure the project path

Open:

```text
constants.py
```

and set:

```python
SRC_DIR = "/path/to/project_root"
```

### 2. Configure the experiment

The main configuration parameters are defined in:

```text
tools/DEEPDOMAIN/deep_domain.py
```

For example:

```python
gama_list = [0.2, 0.3, 0.4]
search_algorithm = "DNSGAII"
mutant_generator_budget = 10 * 60
max_iteration = 6
start_epoch = 1
max_epoch = 6
n_variables = 2
m = 10
```

These parameters control, among other things:

* metamorphic transformation error bounds (`gama_list`)
* search strategy
* mutant-generation budget
* number of search iterations
* number of epochs
* chromosome dimensionality
* source-test candidate pool size

### 3. Run the tool

```bash
python tools/DEEPDOMAIN/deep_domain.py
```

---

## 📤 Output

DeepDomain stores generated results under:

```text
tools/DEEPDOMAIN/results/
```

A typical experiment produces a structure such as:

```text
results/
└── dave2/
    └── DNSGAII/
        └── final_gama_0.2/
            └── epoch_1/
                ├── cache/
                ├── executed_images/
                ├── executed_images_features/
                ├── inconsistent_pool/
                ├── database.csv
                ├── executed_tests.npy
                ├── follow_ups_executed_info.csv
                └── source_executed_info.csv
```

### Output files

| File / Directory               | Description                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------- |
| `cache/`                       | Per-source test generation results, generated populations, follow-ups, metrics, and inconsistencies |
| `executed_images/`             | Images executed on the DNN model                                                                    |
| `executed_images_features/`    | Extracted features of generated test images                                                         |
| `inconsistent_pool/`           | Detected inconsistencies grouped by steering-angle categories                                       |
| `database.csv`                 | Test metadata, including MR index, parameter value, and consistency status                          |
| `executed_tests.npy`           | NumPy representation of executed tests                                                              |
| `follow_ups_executed_info.csv` | Information about follow-up test executions                                                         |
| `source_executed_info.csv`     | Information about source test executions                                                            |

In `database.csv`:

```text
x0          → metamorphic function index
x1          → metamorphic transformation parameter
consistent  → 0 indicates a detected inconsistency
```

---

## 📚 Paper

**Metamorphic Testing of Deep Neural Network-based Autonomous Driving Systems Using Behavioral Domain Adequacy**

**Akram Kalaee and Saeed Parsa**

The paper introduces DeepDomain and presents its behavioral-domain adequacy criteria, test-generation approach, and empirical evaluation.

---

## 🎓 Dissertation

DeepDomain was developed as part of the Ph.D. research:

**Domain Analysis and Its Effect on Improving Testability and Explainability of Learning-based Cyber Physical Systems**

The dissertation provides additional technical details, formulations, figures, tables, experimental analyses, and background on domain-oriented testing and behavioral-domain adequacy.

---

## 🔬 Research Trajectory

DeepDomain represents a step in a broader research trajectory from **model-based testing** and **search-based test generation** toward testing intelligent systems through their behavioral domains:

```text
Model-Based Testing
        │
        ▼
Search-Based Test Generation
        │
        ▼
Domain-Oriented Testing
        │
        ▼
Behavioral Domain Adequacy
        │
        ▼
Testing Intelligent & Autonomous Systems
```

---

## 📖 Citation

If you use DeepDomain in your research, please cite:

```bibtex
@article{KalaeeDeepDomain,
  author  = {Kalaee, Akram and Parsa, Saeed},
  title   = {Metamorphic Testing of Deep Neural Network-based Autonomous Driving Systems Using Behavioral Domain Adequacy},
  journal = {Neural Computing and Applications},
  year    = {2025}
}
```

---

## 📜 License

This project is released under the license specified in [`LICENSE`](LICENSE).

---

## 👤 Authors

**Akram Kalaee**
Ph.D. in Software Engineering
Iran University of Science and Technology

**Saeed Parsa**
Iran University of Science and Technology

---

⭐ If you find this research useful, consider starring the repository.

