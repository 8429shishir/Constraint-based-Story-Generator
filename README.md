🧠 Constraint-Based Story Generator

An AI-powered story generation system that creates meaningful and coherent stories from a set of input images, guided by user-defined constraints such as word limit, tone, and story type.
This project integrates computer vision, natural language processing, and large language models (LLMs) to produce human-like storytelling experiences.

🚀 Project Overview

The system takes multiple input images, analyzes them to detect visual themes and emotions, sorts them based on similarity, and then generates a contextually connected story that satisfies user constraints.

🏗️ Core Idea

“We provide a set of images → the system sorts them by themes → then creates a story using an LLM model, respecting given constraints such as word limit, story type, and tone.”

✨ Key Features

🖼️ Image Understanding: Extracts captions, objects, and emotions from each image.

🧩 Theme-based Sorting: Groups and orders images based on semantic similarity using CLIP embeddings.

🧠 Constraint-based Story Generation: Users can define constraints such as:

Word count or sentence limit

Story type (romantic, thriller, fantasy, etc.)

Tone (happy, sad, mysterious, inspirational)

Narrative voice (first-person, third-person)

🔄 Plot Consistency Layer: Ensures logical story flow by generating outline → expanding scenes.

🗣️ Interactive User Control: Modify constraints dynamically or generate alternative story versions.

📊 Story Evaluation: Checks story coherence, creativity, and constraint satisfaction using NLP metrics.

🎬 Optional Multi-modal Output: Combines story text, input images, and voice narration into a storytelling video.

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
           |
           v
+-----------------------+
| Optional: Story Video |
| (Images + VoiceOver)  |
+-----------------------+

🧰 Technologies Used
🖼️ Computer Vision

OpenAI CLIP – for semantic embedding & image similarity

BLIP / InstructBLIP / OFA – for image caption generation

YOLOv8 / DETR – for object detection (optional)

DeepFace / FER+ – for emotion detection (optional)

💬 Natural Language Processing

GPT-4 / LLaMA 3 / FLAN-T5 – for constraint-based story generation

LangChain – for prompt chaining & constraint control

BERTScore, BLEU, ROUGE – for story evaluation

🧠 Backend & Data

Python 3.x

PyTorch / TensorFlow – for model inference

NumPy / Pandas / scikit-learn – for clustering and preprocessing

Flask / Streamlit – for building interactive UI

🎬 Multimedia (Optional)

gTTS / ElevenLabs API – for voice narration

MoviePy / OpenCV – for generating storytelling videos

| Step | Module                             | Description                                                |
| ---- | ---------------------------------- | ---------------------------------------------------------- |
| 1    | **Input & Preprocessing**          | User uploads images, system resizes and normalizes them.   |
| 2    | **Caption & Embedding Extraction** | Generate captions & embeddings for semantic understanding. |
| 3    | **Theme Sorting**                  | Cluster similar images based on CLIP embeddings.           |
| 4    | **Constraint Handling**            | User specifies story type, tone, and word limit.           |
| 5    | **Story Generation**               | LLM generates coherent narrative based on sorted images.   |
| 6    | **Story Evaluation**               | Evaluate coherence & constraint satisfaction.              |
| 7    | **Optional Multimedia Output**     | Generate narrated video story from text and images.        |


📈 Evaluation Metrics

BLEU / ROUGE: Measures linguistic similarity.

BERTScore: Evaluates semantic similarity.

Constraint Accuracy: Checks whether the generated story meets user-defined conditions.

Human Evaluation: Coherence, creativity, and engagement.

🌱 Future Improvements

Fine-tune smaller LLMs on visual storytelling datasets (like VIST).

Introduce reinforcement learning for better constraint adherence.

Enable multi-lingual story generation.

Implement real-time storytelling assistant for educational or entertainment use.

| Role          | Member          | Responsibility                            |
| ------------- | --------------- | ----------------------------------------- |
| Vision Module | (Your Name)     | Image processing & theme sorting          |
| NLP Module    | (Teammate Name) | Story generation & constraint enforcement |
| UI/Backend    | (Teammate Name) | Web interface & integration               |
| Evaluation    | (Teammate Name) | Story quality analysis                    |
