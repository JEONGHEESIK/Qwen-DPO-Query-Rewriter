# Korean Query Rewriter SLM for RAG Systems

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Page-FFD21E?logo=huggingface)](https://huggingface.co/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3--0.6B-blue)](#)
[![Method](https://img.shields.io/badge/Fine--Tuning-DPO%20%2B%20DoRA-green)](#)

A Small Language Model (SLM) tailored for **Korean synonym and query rewriting**, designed to maximize search coverage in Retrieval-Augmented Generation (RAG) systems.

** Developed a Qwen3-based Query Rewriter and optimized performance through DPO (Direct Preference Optimization) training, with the final model deployed via the corporate Hugging Face repository.

<br/>

## Why Query Rewriter?
In the retrieval phase of a RAG system, if the user's input expression differs from the terminology used in the stored documents (e.g., the user searches for "BEP 분석" but the document uses "손익 분기점"), traditional keyword-matching methods like BM25 often fail to retrieve the correct documents.

This issue is particularly prominent in Korean due to its complex word conjugations and postpositions. This model solves this problem by **automatically transforming and expanding user queries into various semantic equivalents**, drastically improving the matching rate between user intents and document representations.

<br/>

## Model Architecture & Methodology
* **Base Model:** `Qwen 3-0.6B`
* [cite_start]**Fine-Tuning Method:** * Applied **DoRA (Rank: 32, Alpha: 64)** for Parameter-Efficient Fine-Tuning (PEFT)[cite: 313, 976, 977].
  * [cite_start]Conducted rigorous hyperparameter tuning across **23 experimental rounds** to identify optimal weights for context-aware rewriting.
* **Optimization Strategy:** * Leveraged **DPO (Direct Preference Optimization)** to align model outputs with human-preferred semantic expansions.
* **Deployment Experience:** * Successfully optimized and deployed as a production-ready SLM within the corporate Hugging Face environment to enhance internal RAG service performance.

<br/>

## Fine-Tuning Experiment Logs
The hyperparameter tuning and dataset optimization process across 23 experimental rounds has been fully documented. 
**(Click the image below to view the log PDF.)**

[![Experiment Log](https://github.com/user-attachments/assets/8325ad70-fd55-4f18-a1ac-056ba443a0bf)](./docs/develop_log.pdf)

<br/>

## Dataset (DPO Structure)
The model was trained on a custom DPO dataset designed to teach the model how to choose the most natural and contextually appropriate synonyms while preserving the original intent of the Korean text.

| prompt (User Input) | chosen (Preferred Output) | rejected (Dispreferred Output) |
| :--- | :--- | :--- |
| 가장자리에 있는 저 물건은 뭐야? | 가에 있는 저 물건은 뭐야? / 모퉁이에 있는... | 한가운데에 있는 저 물건은 뭐야? |
| 가냘프다는 건 얇다는 거야 약하다는 거야? | 가냘프다는 건 연약하다는 의미야 | 튼튼하다는 건 얇다는 거야 강하다는 거야? |
| 사과를 쪼개어 줄래? | 사과를 갈라줄래? / 사과를 나누어 줄래? | 사과를 통째로 줄래? |

<br/>

## Usage (Inference Code)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "NEXTITS/QUANTUS-L-SLM-2509-v0.9.1"

# 1. Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 2. Prepare the model input
prompt = "이 햄버거는 존맛탱이야"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False 
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 3. Conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128,
    temperature=0.2,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 4. Parsing thinking content
try:
    index = len(output_ids) - output_ids[::-1].index(151668) # 151668 is </think>
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```

## Citation
Thanks to Qwen

### Qwen
```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report},
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={[https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)},
}
```
