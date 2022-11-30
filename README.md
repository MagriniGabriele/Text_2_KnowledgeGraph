# Text_2_KnowledgeGraph

This software, written in Python, let the user create a 
knowledge graph starting from a text file, and then display it.
It uses libraries such as SpaCy and NeuralCOref to elaborate the text file, 
and a much more in depth explaination can be found here:
https://www.overleaf.com/read/ggdbqnpsxtjp

## Usage
For instruction about using this program, run  
```python Text2Knowledge.py -h```
For running a benchmark execute
```python Text2Knowledge.py {path_to_texts_folder} --output {path_to_output} --score```
Note the ground truth count must be placed in the GT.txt file inside the source folder
