<div align = "center" > <h1>🧠 Constraint-Based Story Generator</h1> </div>

An AI-powered story generation system that creates meaningful and coherent stories from a set of input images, guided by user-defined constraints such as word limit, tone, and story type.
This project integrates computer vision, natural language processing, and large language models (LLMs) to produce human-like storytelling experiences.

## 🚀 Project Overview

The system takes multiple input images, analyzes them to detect visual themes and emotions, sorts them based on similarity, and then generates a contextually connected story that satisfies user constraints.

## 🏗️ Core Idea

“We provide a set of images → the system sorts them by themes → then creates a story using an LLM model, respecting given constraints such as word limit, story type, and tone.”

## ✨ Key Features

 **Image Understanding:** Extracts captions, objects, and emotions from each image.
 

**Theme-based Sorting:** Groups and orders images based on semantic similarity using CLIP embeddings and Clustering.

**Constraint-based Story Generation:** Users can define constraints such as:
<br> 1. Word count or sentence limit .<br>2. Story type (romantic, thriller, fantasy, etc.) <br> 3. Tone (happy, sad, mysterious, inspirational) <br> 4. Narrative voice (first-person, third-person)

## 🗣️ Interactive User Control: 
Modify constraints dynamically or generate alternative story versions.

## 📊 Story Evaluation: 
Checks story coherence, creativity, and constraint satisfaction using NLP metrics.

```
                                            
                                            +-----------------------+
                                            |    Image Dataset      |
                                            +----------+------------+
                                                       |
                                                       v
                                            +-----------------------+
                                            | Image Preprocessing   |
                                            | (Resizing, Cleaning)  |
                                            +----------+------------+
                                                       |
                                                       v
                                            +-----------------------+
                                            | Image Captioning      |
                                            | (BLIP / OFA)          |
                                            +----------+------------+
                                                       |
                                                       v
                                            +-----------------------+
                                            | Embedding Extraction  |
                                            | (CLIP Model)          |
                                            +----------+------------+
                                                       |
                                                       v
                                            +-----------------------+
                                            | Theme Sorting /       |
                                            | Clustering (K-Means)  |
                                            +----------+------------+
                                                       |
                                                       v
                                            +-----------------------+
                                            | Constraint Controller |
                                            | (Word limit, Tone,    |
                                            |  Genre, POV)          |
                                            +----------+------------+
                                                       |
                                                       v
                                            +-----------------------+
                                            | LLM Story Generator   |
                                            | (GPT / LLaMA / T5)    |
                                            +----------+------------+
                                                       |
                                                       v
                                            +-----------------------+
                                            | Story Evaluation      |
                                            | (BLEU, ROUGE, etc.)   |
                                            +----------+------------+

``` 
## 🧰 Technologies Used

| **Category** | **Technologies / Tools** | **Purpose** |
|---------------|---------------------------|--------------|
| 🖼️ **Computer Vision** | - OpenAI CLIP  <br> - BLIP / InstructBLIP /| - Semantic embedding & image similarity <br> - Image caption generation <br>|
| 💬 **Natural Language Processing** | - Qwen2.5 <br> - BERTScore, BLEU, ROUGE | - Constraint-based story generation <br> - Story quality evaluation |
| 🧠 **Backend & Data Processing** | - Python 3.10  <br> - PyTorch / TensorFlow  <br> - NumPy / Pandas / scikit-learn  <br> Streamlit | - Model inference <br> - Data preprocessing & clustering <br> 
---

## ⚙️ Installation

| **Step** | **Command / Description** |
|---------|----------------------------|
| **1. Clone Repository** | <pre>git clone https://github.com/8429shishir/Constraint-based-Story-Generator.git|
|Move inside Project | <pre>cd constraint-based-story-generator</pre> |
| **2. Create Virtual Environment** | <pre>python -m venv venv
| Activate Environment(Linux/Mac)|<pre>source venv/bin/activate  </pre> 
|  Activate Environment(Windows)|<pre>venv\Scripts\activate    </pre> |
| **3. Install Dependencies** | <pre>pip install -r requirements.txt</pre> |
| **4. Run the Application** | <pre>streamlit run app.py</pre> |
| **5. Access the Web UI** | <pre>Open `http://localhost:5000/` or Streamlit URL</pre> |


## Modules
| Step | Module                             | Description                                                |
| ---- | ---------------------------------- | ---------------------------------------------------------- |
| 1    | **Input & Preprocessing**          | User uploads images, system resizes and normalizes them.   |
| 2    | **Caption & Embedding Extraction** | Generate captions & embeddings for semantic understanding. |
| 3    | **Theme Sorting**                  | Cluster similar images based on CLIP embeddings.           |
| 4    | **Constraint Handling**            | User specifies story type, tone, and word limit.           |
| 5    | **Story Generation**               | LLM generates coherent narrative based on sorted images.   |
| 6    | **Story Evaluation**               | Evaluate coherence & constraint satisfaction.              |

## 📈 Evaluation Metrics

**BLEU / ROUGE:** Measures linguistic similarity.

**BERTScore:** Evaluates semantic similarity.

**Constraint Accuracy:** Checks whether the generated story meets user-defined conditions.



## 🌱 Future Improvements

1.Fine-tune smaller LLMs on visual storytelling datasets (like VIST).

2.Introduce reinforcement learning for better constraint adherence.

3.Enable multi-lingual story generation.

4.Implement real-time storytelling assistant for educational or entertainment use.

