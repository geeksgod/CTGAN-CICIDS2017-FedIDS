* **Official Dataset Page:** [UNB CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
* **Description:** The dataset contains benign network traffic and up-to-date, realistic cyberattacks (such as DDoS, Brute Force, XSS, and Infiltration) captured over a 5-day period. It is provided in both raw packet capture (pcap) format and pre-processed CSV format containing calculated network flow features.

### Run the CTGAN
1. Upload dataset to you google drive
2. Run CTGAN > 600epochCTGAN.ipnyb file in google collab

### Run the FED-IDS
1. Download training data and Synthetic data form CTGAN run
2. Add it to dataset folder 
3. run code 
```bash
python federated_ids_ctgan.py
```





## 🤖 AI Refactoring & Attribution Disclosure

A significant portion of this codebase was optimized and refactored using generative AI assistance. While the core research logic, algorithmic requirements, and system design were defined by the author, **[OpenAI ChatGPT, Anthropic Claude, Gemini]** was utilized to improve code quality, performance, and structure.

### Areas of AI Assistance
* **Code Refactoring:** Modularizing monolithic scripts into clean, reusable functions and classes.
* **Performance Optimization:** Enhancing loop structures, vectorizing data operations, and improving memory efficiency.
* **Documentation & Testing:** Automatically generating docstrings, inline comments, and writing unit test structures.

### Human Oversight Statement
In alignment with academic integrity guidelines, all AI-generated refactoring was critically reviewed, manually modified, and rigorously tested by the author. The final codebase was verified to ensure that the underlying research logic remains accurate, secure, and entirely representative of the author's intended methodologies.

## Citation

```bibtex
@software{federated_cybersecurity_2024,
  title        = {Federated Learning for Cybersecurity Threat Detection},
  author       = {{Cybersecurity Research Team}},
  year         = {2024},
  url          = {[https://github.com/yourusername/federated-cybersecurity](https://github.com/yourusername/federated-cybersecurity)},
  note         = {GitHub Repository}
}


@inproceedings{ctgan,
  title        = {Modeling Tabular data using Conditional GAN},
  author       = {Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle    = {Advances in Neural Information Processing Systems},
  year         = {2019}
}