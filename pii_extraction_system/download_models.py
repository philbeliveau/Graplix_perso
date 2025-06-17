#!/usr/bin/env python3
"""
Script de téléchargement automatique des modèles ML
Exécutez ce script après le clone pour télécharger les modèles requis
"""

import os
import sys
from pathlib import Path

def download_models():
    """Télécharge automatiquement les modèles ML requis."""
    print("🔄 Téléchargement des modèles ML...")
    print("📍 Cela peut prendre quelques minutes...")
    
    try:
        # Import transformers
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        print("✅ Transformers library importée")
        
        models_to_download = [
            "dbmdz/bert-large-cased-finetuned-conll03-english",
        ]
        
        for model_name in models_to_download:
            print(f"📥 Téléchargement: {model_name}")
            try:
                # Download tokenizer and model - they'll be cached automatically
                print("  📝 Téléchargement du tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("  🧠 Téléchargement du modèle...")
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                print(f"✅ {model_name} téléchargé avec succès")
            except Exception as e:
                print(f"❌ Erreur lors du téléchargement de {model_name}: {e}")
                print("💡 Conseil: Vérifiez votre connexion internet et réessayez")
                return False
        
        print("\n🎉 Tous les modèles ont été téléchargés!")
        print("🚀 Vous pouvez maintenant utiliser le système PII Extraction")
        return True
        
    except ImportError as e:
        print(f"❌ Dépendances manquantes: {e}")
        print("💡 Installez d'abord les dépendances avec:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

if __name__ == "__main__":
    success = download_models()
    if success:
        print("\n✅ Installation terminée! Vous pouvez maintenant:")
        print("   python validation_test.py  # Pour tester le système")
        print("   cd src/dashboard && streamlit run main.py  # Pour démarrer le dashboard")
    sys.exit(0 if success else 1)
