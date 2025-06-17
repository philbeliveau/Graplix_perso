#!/bin/bash

echo "🚀 Configuration Rapide - Système PII Extraction"
echo "==============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "🐍 Python version: $python_version"

# Install dependencies
echo "📦 Installation des dépendances..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'installation des dépendances"
    echo "💡 Essayez: pip install -r requirements.txt"
    exit 1
fi

# Download models
echo "📥 Téléchargement des modèles ML..."
python3 download_models.py

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du téléchargement des modèles"
    echo "💡 Vérifiez votre connexion internet et réessayez"
    exit 1
fi

# Run validation
echo "🧪 Validation du système..."
python3 validation_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCÈS! Le système PII Extraction est prêt!"
    echo "============================================="
    echo ""
    echo "🚀 Pour démarrer le dashboard:"
    echo "   cd src/dashboard"
    echo "   streamlit run main.py"
    echo ""
    echo "📚 Documentation disponible dans: docs/"
    echo "🧪 Tests disponibles dans: tests/"
    echo ""
    echo "✅ Le système est entièrement opérationnel!"
else
    echo ""
    echo "⚠️ Des problèmes ont été détectés lors de la validation"
    echo "📖 Consultez docs/TROUBLESHOOTING.md pour l'aide"
fi
