### Feb 6, 2026
Tried 4b, 7b, and 13b models. 
The 7b performed significantly better than the 4b, but the 13b offered little improvment with much slower runtime. Ideally we'd run a MUCH larger model on cloud hardware, but this may have to do for now.

### Feb 8, 2026
mistral:7b-instruct is performing much better than gemma3:4b
Another concern may be that repeated runs from the same model could create nearly identical output between runs (or multiple similar titles for the same labels)

### Feb 14th, 2026
Using gemini 3 flash preview through the API is looking like the best option

### Feb 15th, 2026
Having multiple class counts is not viable for training or validation, but I am keeping it as it allows the model to build more nuanced responses

I also am considering using a BERT model for validating the emotions assigned by the LLM, however
1. This could introduce the bias of said BERT model, which itself might be lower quality then the LLM
2. Most of these models don't map 1-1 to with our dataset of emotions, and include emotions we don't classify
