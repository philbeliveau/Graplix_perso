# How OCR + PII Detection Works - Complete Pipeline Explained

## 🔍 **The Complete Journey: From Image to PII Detection**

Let me walk you through exactly how your system identifies PII from OCR text using a real example from your French document.

---

## 📄 **Step 1: Document Input**
```
Input: JPG image with French medical document
```

## 🔍 **Step 2: OCR Text Extraction**
Your OCR produces text like:
```
"Québec MÉDECINE
Laval Hôpital de la Santé  
175, boul. René Lévesque
Date de naissance: 14/05/2024
Nom: Jean Dupont
Téléphone: (450) 555-1234
Numéro d'assurance: 123-456-789"
```

## 🧠 **Step 3: Multiple PII Detection Methods**

The system runs **3 different extractors** simultaneously on this text:

### **🔢 1. Rule-Based Extractor (Regex Patterns)**

**How it works:** Uses pre-written patterns to find specific formats

```python
# French phone number pattern
pattern = r'\b0[1-9](?:[-.\s]?\d{2}){4}\b'
# Matches: (450) 555-1234

# Date pattern  
pattern = r'\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(19|20)\d{2}\b'
# Matches: 14/05/2024

# Credit card pattern
pattern = r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'
# Matches: 1234-5678-9012-3456
```

**What it finds in your text:**
- `📞 (450) 555-1234` → **PHONE_NUMBER** (confidence: 0.95)
- `📅 14/05/2024` → **DATE_OF_BIRTH** (confidence: 0.90)
- `🔢 123-456-789` → **SOCIAL_SECURITY_NUMBER** (confidence: 0.85)

### **🤖 2. NER Extractor (AI Models)**

**How it works:** Uses pre-trained AI models (like BERT) that learned from millions of examples

```python
# The model processes: "Nom: Jean Dupont"
# AI model thinks: "Jean Dupont follows 'Nom:' which means 'Name:' in French"
# → This is likely a PERSON name

# The model processes: "Laval Hôpital de la Santé"  
# AI model thinks: "Hôpital means hospital, this is a healthcare organization"
# → This is likely an ORGANIZATION
```

**What it finds in your text:**
- `👤 Jean Dupont` → **PERSON_NAME** (confidence: 0.92)
- `🏥 Laval Hôpital de la Santé` → **ORGANIZATION** (confidence: 0.88)
- `📍 175, boul. René Lévesque` → **ADDRESS** (confidence: 0.75)

### **📍 3. Layout-Aware Extractor**

**How it works:** Looks at the structure and context around text

```python
# Analyzes: "Date de naissance: 14/05/2024"
# Layout awareness: "This date follows the label 'Date de naissance' (Date of birth)"
# → This is specifically a BIRTH DATE, not just any date

# Analyzes: "Nom: Jean Dupont"
# Layout awareness: "This name follows the label 'Nom:' (Name:)"
# → This is specifically a PATIENT NAME
```

**What it finds in your text:**
- `📅 14/05/2024` → **DATE_OF_BIRTH** (confidence: 0.93) *(higher confidence than rule-based)*
- `👤 Jean Dupont` → **PATIENT_NAME** (confidence: 0.96) *(more specific than generic person)*

---

## 🔗 **Step 4: Results Combination & Confidence Scoring**

The system combines all three results:

### **For "Jean Dupont":**
- Rule-based: ❌ (no match - it only does patterns, not names)
- NER: ✅ PERSON_NAME (confidence: 0.92)
- Layout-aware: ✅ PATIENT_NAME (confidence: 0.96)

**Final result:** `PERSON_NAME` with confidence `0.94` (averaged and weighted)

### **For "(450) 555-1234":**
- Rule-based: ✅ PHONE_NUMBER (confidence: 0.95)
- NER: ✅ PHONE_NUMBER (confidence: 0.87)
- Layout-aware: ✅ CONTACT_PHONE (confidence: 0.91)

**Final result:** `PHONE_NUMBER` with confidence `0.91`

### **For "14/05/2024":**
- Rule-based: ✅ DATE (confidence: 0.90)
- NER: ❌ (AI didn't recognize the format)
- Layout-aware: ✅ DATE_OF_BIRTH (confidence: 0.93)

**Final result:** `DATE_OF_BIRTH` with confidence `0.91`

---

## 🎯 **Step 5: Final PII Entities**

Your document produces these PII detections:

```json
[
  {
    "text": "Jean Dupont",
    "type": "PERSON_NAME", 
    "confidence": 0.94,
    "start_pos": 89,
    "end_pos": 100,
    "context": "Nom: Jean Dupont",
    "extractor": "ensemble"
  },
  {
    "text": "(450) 555-1234",
    "type": "PHONE_NUMBER",
    "confidence": 0.91, 
    "start_pos": 156,
    "end_pos": 170,
    "context": "Téléphone: (450) 555-1234",
    "extractor": "ensemble"
  },
  {
    "text": "14/05/2024", 
    "type": "DATE_OF_BIRTH",
    "confidence": 0.91,
    "start_pos": 45,
    "end_pos": 55,
    "context": "Date de naissance: 14/05/2024",
    "extractor": "ensemble"
  }
]
```

---

## 🧪 **Why Each Method is Important**

### **🔢 Rule-Based (Regex)**
- **Best for:** Structured data (phone numbers, credit cards, dates)
- **Advantage:** 100% consistent, fast, no AI required
- **Limitation:** Can't understand context or names

### **🤖 NER (AI Models)**  
- **Best for:** Names, organizations, locations
- **Advantage:** Understands language and context
- **Limitation:** Can miss unusual formats, needs training data

### **📍 Layout-Aware**
- **Best for:** Document-specific context (forms, medical records)
- **Advantage:** Uses document structure for better accuracy
- **Limitation:** More complex, slower processing

---

## ⚡ **Why the Ensemble Approach Works**

**Single method limitations:**
- Rule-based alone: Misses "Jean Dupont" (no pattern for names)
- NER alone: Misses "14/05/2024" (unusual date format)
- Layout-aware alone: Slower and computationally intensive

**Combined ensemble:**
- ✅ Catches everything each method is good at
- ✅ Higher confidence through consensus  
- ✅ More robust to OCR errors
- ✅ Better accuracy across different document types

This is why your system can reliably find PII even in messy OCR text from JPG files!