# Estrategia EntropyMaster: Julián y Tania

## Descripción General
Nuestra IA, `EntropyMaster_julian_tania`, utiliza una estrategia híbrida basada en la Teoría de la Información (Entropía de Shannon) y el concepto de Score Esperado.

Lo optimizamos con operaciones matriciales en `numpy` para respetar el límite de 5 segundos por juego. 

Nuestro algoritmo divide el juego en 3 fases estratégicas:

### 1. Fase 1: Openers (Precomputación)
Primero precalculamos el máximo global de entropía para cada configuración del juego de manera offline. Así, en el primer turno ya tenemos el opener inmediato. Esto nos garantiza la mayor ganancia de información en 0.001 segundos, guardando nuestro presupuesto de CPU para los turnos siguientes.

### 2. Fase 2: Aprendizaje y Exploración
Utilizamos la ecuación de Shannon con NumPy para evaluar la ganancia de información (en *bits*) de posibles jugadas.
- **Evitar el timeout:** Si hay más de 150 candidatos, tomamos una muestra representativa (`random.sample`) para mantener la eficiencia.
- **Estrategia de exploración:** Si hay entre 3 y 15 palabras restantes que comparten casi todas las letras, sacrificamos un turno para ganar información al inyectar palabras de todo el vocabulario.
- **Bono de decisión:** Si la palabra a probar es candidata a ser la solución, al evaluar si es opción para el siguiente turno, se le otorga un bono matemático de `+0.1`. De esta manera, si hay empate en cuanto a la entropía o ganancia de información con una palabra no candidata, gane la que tiene mayor ganancia de información **y** tiene probabilidad de ser la solución.

### 3. Fase 3: Score Esperado
Cuando el espacio de búsqueda se reduce a 2 palabras o menos, o si estamos en nuestro último intento (turno 6), apagamos el motor de Shannon. Se ordenan las opciones restantes basándose puramente en su probabilidad de ocurrencia en español (sigmoide de frecuencias) y va por la más probable.


## Copy this directory to create your team workspace:

```bash
cp -r estudiantes/_template estudiantes/your_team_name
```

## Directory Structure

```
estudiantes/your_team_name/
    strategy.py          # YOUR STRATEGY (this is what gets submitted)
    results/             # Auto-created: experiment and tournament outputs
    ...                  # Anything else you want (notebooks, scripts, data)
```

## Quick Start

1. **Edit** `strategy.py` — change the class name, the `name` property, and implement `guess()`.

2. **Test** your strategy:
   ```bash
   python experiment.py --strategy "YourName_teamname" --num-games 10 --verbose
   ```

3. **Compare** against benchmarks (Random, MaxProb, Entropy):
   ```bash
   python tournament.py --team your_team_name --num-games 20
   ```

4. **Run specific configurations:**
   ```bash
   # 6-letter words, frequency mode
   python experiment.py --strategy "YourName_teamname" --length 6 --mode frequency --num-games 20 --verbose

   # Full local tournament (all configs)
   python tournament.py --team your_team_name --official --num-games 10
   ```

## Rules

- Your strategy must work for **all** word lengths (4, 5, 6) and **both** modes (uniform, frequency).
- **5 seconds** max per game (one secret word, up to 6 guesses).
- **1 CPU core** during tournament.
- Only `numpy` + standard library allowed (no extra dependencies).
- No machine learning or reinforcement learning. Use goal-based or utility-based approaches. Simulations are allowed.
- The `name` property must be unique: `"StrategyName_teamname"`.

## Useful Utilities

```python
from wordle_env import feedback, filter_candidates
from lexicon import load_lexicon

# Compute feedback for a guess against a secret
fb = feedback("canto", "arcos")  # -> (1, 0, 1, 1, 0)

# Filter candidates by feedback
remaining = filter_candidates(word_list, "arcos", (1, 0, 1, 1, 0))

# Load word list with probabilities
lex = load_lexicon(word_length=5, mode="frequency")
print(lex.words[:5], lex.probs)
```
