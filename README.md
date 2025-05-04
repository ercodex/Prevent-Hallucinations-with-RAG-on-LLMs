# Formal Languages and Automata Semester Project: Prevent Hallucinations with RAG on LLMs
## Improving LLM accuracy with RAG implementation.

---

### Group Members
- Eren Çil (Coordinator)
- Server Ahıskalı
- Merve Taş
- Furkan Gemici

---

### Project Description

With my project group, we applied RAG on a small language model to improve its accuracy. Here is some information about our project:

---

### Technical Details

- **LLM and Tokenizer:** TinyLlama-1.1B-Chat-v1.0  
- **Embedding Model:** MiniLM-L6-v2  
- **Similarity Measurement:** Cosine Similarity

---

### Runtime Environment

We run our LLM with CUDA, on GTX 1650 GPU. It generates answers within 5-6 seconds.

---

### Knowledge Base Construction

First we created a knowledge-base, which is a small XML file containing information about some topics such as history, geography, art, technology etc. By "information", we meant sentences about specific topics. For example:  
> The first mechanical calculator was invented by Blaise Pascal.

---

### Use Case: Preventing Hallucinations

We picked information about potential hallucination cases. Consider given example above. We asked:

> Who invented the first mechanical calculator?

- **Without RAG:**  
  The first mechanical calculator was invented by Charles Harrison in 1805.

- **With RAG:**  
  Blaise Pascal invented the first mechanical calculator in the year 1642. He was a French mathematician and inventor.

Even though we didn't mention it in the knowledge base, the invention year is true! So, we choose topics which our model is prone to see hallucinations. This was one of them.

---

### XML and DTD

Then we needed to define a DTD for our XML file so we don't make any mistakes while adding information to knowledge-base.  
You can take a look at both XML and DTD file in this repo.

---

### Similarity Check Mechanism

In our similarity-check process, embedding model iterates through each sentence while turning them into vectors, every time user asks a question.

We are aware that another possible (probably more efficient) solution for this step was creating an extra entry called "vector value" inside each sentence, and not running embedding model every time. Since our intention was just creating a demo and understanding how RAG works, we decided that this update is not necessary in our case.

---

### Application Logic

After completing knowledge-base, we wrote a Python code which acts as an application. You can find it in this repo too. We use our model via this code.

---

### Prompt Strategy

As far as I know, TinyLlama doesn't have Q-A mode, it just generates text. So, we needed to trick it into answering questions. For this, I created a scheme that gives the model an insight above previous context.

There are four inputs called `system`, `additional-information`, `user-question`, and `answer`.

- `"system"` input says to the model, "You are a chatbot that answers questions."  
- Other inputs gets filled one by one via some functions coded for this purpose (`addQuestionToPrompt()`, `addInfoToPrompt()`).  
- Of course, we leave `"answer"` part blank.  
- And model starts to generate text on this pre-defined, created-along-the-way string.

---

### We can summarize the whole process like this. You can find all files mentioned in this repo. Thanks!
