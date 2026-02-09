### Feb 6, 2026
Tried 4b, 7b, and 13b models. 
The 7b performed significantly better than the 4b, but the 13b offered little improvment with much slower runtime. Ideally we'd run a MUCH larger model on cloud hardware, but this may have to do for now.

### Feb 8, 2026
mistral:7b-instruct is performing much better than gemma3:4b
Another concern may be that repeated runs from the same model could create nearly identical output between runs (or multiple similar titles for the same labels)