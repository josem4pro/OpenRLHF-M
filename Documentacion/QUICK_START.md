# 🚀 QUICK START - Modelo DPO Entrenado

## ⚡ Ejecución Rápida (1 Comando)

```bash
cd /home/jose/Repositorios/OpenRLHF-M
source .venv-cpu-rlhf/bin/activate
python3 dpo_training_cpu.py
```

**Resultado**: Modelo entrenado en `dpo_output/final_model/` en ~5 minutos

---

## 🔄 Re-entrenar con Diferentes Parámetros

### Más Épocas (Mejor Calidad)
```python
# Editar dpo_training_cpu.py línea 184
num_train_epochs=5,  # En vez de 3
```

### Dataset Más Grande
```python
# Agregar más pares en línea 66-75
preference_data = [
    # ... pares existentes ...
    {
        "prompt": "Nueva pregunta",
        "chosen": "Respuesta preferida",
        "rejected": "Respuesta rechazada"
    },
]
```

---

## 🧪 Probar el Modelo Entrenado

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cargar modelo entrenado
model = AutoModelForCausalLM.from_pretrained(
    "./dpo_output/final_model",
    torch_dtype=torch.float32,
    device_map=None
)

tokenizer = AutoTokenizer.from_pretrained("./dpo_output/final_model")

# Generar respuesta
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## 📊 Ver Resultados

### Reporte JSON
```bash
cat dpo_output/training_report.json | jq
```

### Log Completo
```bash
less dpo_training.log
```

### Métricas de Entrenamiento
```bash
grep "loss" dpo_training.log | tail -10
```

---

## 🔧 Troubleshooting

### Error: "Out of Memory"
**Solución**: Reducir batch size
```python
# Línea 185
per_device_train_batch_size=1,  # En vez de 2
```

### Error: "CUDA not available"
**Solución**: Ya configurado para CPU - verificar:
```python
# Líneas 191-194
fp16=False,
bf16=False,
use_cpu=True,
no_cuda=True,
```

### Error: "Model not found"
**Solución**: Descargar manualmente
```bash
# Configurar token (si no está en ~/.env)
export HF_TOKEN="your_huggingface_token_here"

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --token $HF_TOKEN
```

---

## 📁 Estructura de Archivos

```
OpenRLHF-M/
├── dpo_training_cpu.py         # ← Script principal
├── dpo_training.log            # ← Log de ejecución
├── dpo_output/
│   ├── final_model/            # ← Modelo entrenado (1.9GB)
│   └── training_report.json    # ← Métricas completas
├── ✅-MISION-COMPLETA-100%.md  # ← Documentación exhaustiva
└── QUICK_START.md              # ← Esta guía
```

---

## 🎯 Comandos Útiles

### Ver progreso en tiempo real
```bash
tail -f dpo_training.log
```

### Verificar espacio en disco
```bash
du -sh dpo_output/
```

### Limpiar outputs anteriores
```bash
rm -rf dpo_output/
```

### Backup del modelo entrenado
```bash
tar -czf dpo_model_$(date +%Y%m%d).tar.gz dpo_output/final_model/
```

---

**¿Necesitas más ayuda?** Ver: [✅-MISION-COMPLETA-100%.md](✅-MISION-COMPLETA-100%.md)
