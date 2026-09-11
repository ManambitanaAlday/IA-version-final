import os
import glob
import json
import re
import ast

def force_json_repair(raw_text):
    """
    Moteur de réparation renforcé (Forcé) pour résoudre les erreurs de syntaxe :
    - Noms de clés sans guillemets ou avec guillemets simples (A.json, R.json)
    - Délimiteur virgule manquant (D.json, L.json, X.json)
    - Virgules traînantes en fin d'objet
    - Booléens et None au format Python (True, False, None)
    - Chaines entre apostrophes au lieu de guillemets
    """
    # 1. Conversion des littéraux Python -> JSON
    text = re.sub(r'\bTrue\b', 'true', raw_text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)

    # 2. Réparation des clés sans guillemets ou entre apostrophes (ex: A.json, R.json)
    text = re.sub(r"'([a-zA-Z0-9_]+)'\s*:", r'"\1":', text)
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', text)

    # 3. Suppression des virgules superflues avant } ou ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # 4. Insertion systématique des virgules manquantes entre propriétés / éléments (ex: D.json, L.json, X.json)
    text = re.sub(r'}\s*{', '},{', text)
    text = re.sub(r']\s*\[', '],[', text)
    text = re.sub(r'([0-9"true|false|null])\s*\n*\s*(")', r'\1,\2', text)
    text = re.sub(r'}\s*\n*\s*(")', r'},\1', text)
    text = re.sub(r']\s*\n*\s*(")', r'],\1', text)

    # Essai de parsing n°1 : JSON réparé
    try:
        return json.loads(text)
    except Exception:
        pass

    # Essai de parsing n°2 : Normalisation des guillemets simples sur les valeurs
    try:
        def replace_single_quotes(m):
            val = m.group(1).replace('"', '\"')
            return f'"{val}"'
        text_sq = re.sub(r"'([^'\n\r]*)'", replace_single_quotes, text)
        return json.loads(text_sq)
    except Exception:
        pass

    # Essai de parsing n°3 : Évaluation AST Python (si la structure est un dict Python)
    try:
        py_obj = ast.literal_eval(raw_text)
        return json.loads(json.dumps(py_obj))
    except Exception:
        pass

    # Essai de parsing n°4 : Extraction forcée par blocs (Regex Chunking)
    extracted = []
    blocks = re.findall(r'\{[^{}]+\}', text)
    for b in blocks:
        try:
            b_clean = re.sub(r',\s*}', '}', b)
            b_clean = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', b_clean)
            extracted.append(json.loads(b_clean))
        except Exception:
            continue
            
    if extracted:
        return extracted

    return None

def repair_and_load_json(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Tentative standard rapide
    try:
        return json.loads(content)
    except Exception:
        pass

    # Lancement de la réparation forcée
    return force_json_repair(content)

def remove_emplacement_keys(data):
    """Supprime récursivement la clé 'emplacement' dans toute la structure JSON."""
    if isinstance(data, dict):
        return {k: remove_emplacement_keys(v) for k, v in data.items() if k != 'emplacement'}
    elif isinstance(data, list):
        return [remove_emplacement_keys(item) for item in data]
    return data

def extract_medicaments(data):
    """Extrait les médicaments de la structure."""
    meds = []
    if isinstance(data, dict):
        if 'medicaments' in data and isinstance(data['medicaments'], list):
            meds.extend(data['medicaments'])
        elif 'nom' in data:
            meds.append(data)
        else:
            for v in data.values():
                meds.extend(extract_medicaments(v))
    elif isinstance(data, list):
        for item in data:
            meds.extend(extract_medicaments(item))
    return meds

def build_finetuning_dataset(medicaments):
    """Génère le dataset au format JSONL ChatML/OpenAI."""
    dataset = []
    
    for med in medicaments:
        if not isinstance(med, dict):
            continue
            
        nom = med.get('nom', 'Médicament')
        famille = med.get('famille', 'Non renseignée')
        dosage = med.get('dosage', '')
        posologie = med.get('posologie_standard', 'Consulter la notice.')
        contre_ind = med.get('contre_indications', [])
        effets_sec = med.get('effets_secondaires', [])
        substituts = med.get('substituts', [])
        
        # Exemple 1 : Fiche d'information
        prompt_1 = f"Quelles sont les informations clés, la posologie et les contre-indications du médicament {nom} ({dosage}) ?"
        resp_1 = (
            f"**{nom}** ({famille}, {dosage})\n"
            f"- **Posologie standard** : {posologie}\n"
            f"- **Contre-indications** : {', '.join(contre_ind) if isinstance(contre_ind, list) and contre_ind else 'Aucune répertoriée.'}\n"
            f"- **Effets secondaires** : {', '.join(effets_sec) if isinstance(effets_sec, list) and effets_sec else 'Aucun répertorié.'}"
        )
        dataset.append({
            "messages": [
                {"role": "system", "content": "Vous êtes un assistant pharmacien expert en pharmacologie et substituts médicamenteux."},
                {"role": "user", "content": prompt_1},
                {"role": "assistant", "content": resp_1}
            ]
        })
        
        # Exemple 2 : Substituts
        if substituts and isinstance(substituts, list):
            sub_details = []
            for s in substituts:
                if isinstance(s, dict):
                    s_nom = s.get('nom', '')
                    s_raison = s.get('raison', '')
                    sub_details.append(f"- **{s_nom}** : {s_raison}")
            
            if sub_details:
                prompt_2 = f"Quel substitut puis-je proposer pour le {nom} et pour quelle raison ?"
                resp_2 = f"Voici les alternatives recommandées pour {nom} :\n" + "\n".join(sub_details)
                
                dataset.append({
                    "messages": [
                        {"role": "system", "content": "Vous êtes un assistant pharmacien expert en pharmacologie et substituts médicamenteux."},
                        {"role": "user", "content": prompt_2},
                        {"role": "assistant", "content": resp_2}
                    ]
                })
            
    return dataset

def run_pipeline(input_folder="./json_files", output_clean_folder="./json_cleaned", output_ft_file="finetuning_dataset.jsonl"):
    os.makedirs(output_clean_folder, exist_ok=True)
    all_medicaments = []
    
    # Parcours des fichiers A.json à Z.json
    for letter in [chr(i) for i in range(ord('A'), ord('Z') + 1)]:
        file_path = os.path.join(input_folder, f"{letter}.json")
        if not os.path.exists(file_path):
            continue
            
        print(f"[+] Traitement de {file_path}...")
        data = repair_and_load_json(file_path)
        
        if data is not None:
            # 1. Suppression de 'emplacement'
            cleaned_data = remove_emplacement_keys(data)
            
            # Sauvegarde du JSON valide et nettoyé
            clean_path = os.path.join(output_clean_folder, f"{letter}_cleaned.json")
            with open(clean_path, 'w', encoding='utf-8') as f_out:
                json.dump(cleaned_data, f_out, ensure_ascii=False, indent=2)
                
            # 2. Accumulation pour le dataset
            meds = extract_medicaments(cleaned_data)
            all_medicaments.extend(meds)
            print(f"  [✓] {file_path} réparé et extrait avec succès ({len(meds)} médicaments).")
        else:
            print(f"  [X] Impossible de rattraper {file_path} malgré les correctifs forcés.")
            
    # 3. Exportation du jeu de fine-tuning final
    ft_dataset = build_finetuning_dataset(all_medicaments)
    with open(output_ft_file, 'w', encoding='utf-8') as f_ft:
        for entry in ft_dataset:
            f_ft.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\n[✓] Traitement terminé !")
    print(f"    - JSON nettoyés sauvegardés dans : {output_clean_folder}")
    print(f"    - Dataset de fine-tuning créé : {output_ft_file} ({len(ft_dataset)} paires d'instructions)")

if __name__ == "__main__":
    run_pipeline()