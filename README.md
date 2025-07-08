---
title: AI Azure Architect
emoji: 💡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# Starting Point for the Final Project of the "From Beginner to Advanced LLM Developer" course

## Overview

This repository contains the code of the final "Part 4; Building Your Own advanced LLM + RAG Project to receive certification" lesson of the "From Beginner to Advanced LLM Developer" course.

Congrats, you are at the last step of the course! In this final project you'll have the possibility to practice with all the techniques that you learned and earn your certification.

If you want, you can use this repository as starting point for your final project. The code here is the same as in the "Building and Deploying a Gradio UI on Hugging Face Spaces" lesson, so you should be already familiar with it. If you want to use it for your project, fork this repository here on GitHub. By doing so, you'll create a copy of this repository in your GitHub account that you can modify as you want.

## Setup

1. Create a `.env` file and add there your OpenAI API key. Its content should be something like:

```bash
OPENAI_API_KEY="sk-..."
COHERE_API_KEY="...."
```

<b>Note: Open AI and Cohere Keys are manually enter in a textbox</b>

2. Create a local virtual environment, for example using the `venv` module. Then, activate it.

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

4. Launch the Gradio app.

```bash
python app.py
```

# Data Collection and curation

Check this [Data Collection](/scripts/00_DataCollection.md) file for information about collection and curation information.

# Cost 
The user can try all the functionalities with $0.50 or less.

# Optional functionalities implemented

1. Implement streaming responses. ✅
2. There's code for RAG evaluation in the [folder](/scripts/), and the README contains the evaluation results. The folder must also contain the evaluation dataset and the evaluation scripts. ✅
3. The app is designed for a specific goal/domain that is not a tutor about AI. This app is focused on Azure engineering ✅
4. You have shown evidence of collecting at least two data sources beyond those provided in our course. ([Five datasources collected](/scripts/)) ✅
5. Use a reranker in your RAG pipeline. It can be a fine-tuned version (your choice). ✅
6. Use a fine-tuned embedding model in your app. ✅

# Example questions
- when to use Azure functions vs app service
- how do I keep microservices decoupled and independent and achive HA
- Use the many-models architecture approach to scale machine learning models
