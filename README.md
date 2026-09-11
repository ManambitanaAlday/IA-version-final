Nécessite Ollama installé en locale avec un modèle LLM téléchargé depuis Hugging Face au moins
=> lancer ollama via cmd : ollama serve

Nécessite python installé sur la machine locale

Finetuning est à executer sur google colab puis le dossier qwen_pharma_lora va être créé
=> Lancer ensuite sur google collab !zip -r qwen_pharma_lora.zip qwen_pharma_lora

Télécharger le zip et à placer dans le dossier pricipale avec les scripts

Objectifs : 
TAF 
Training data et nettoyage (processdat_py)
1) Algorithme Script : tokenisation et embedding (data_set_prep.py)
Attention layer => MLP => Unembedding
2) Hugging face Modele GPT léger (1.5B ou 2.5B) (qwen2,5B-1.5B)
3) Finetuning du modèle reentrainement (google collab - finetuning.py)
4) Intregrer le modèle dans du logique métier (cas chatbot pharmacie assistant pharmacien et gestionnaire de pharmacie)

