
from __future__ import annotations
import numpy as np
import random
from collections import defaultdict
from strategy import Strategy, GameConfig
from wordle_env import feedback, filter_candidates


class MyStrategy(Strategy):
    """Nuestra estrategia será híbrida, basada en la Entropía (Numpy) y Score Esperado (W.5+W.6)."""

    @property
    def name(self) -> str:
        # Convention: "StrategyName_teamname"
        return "EntropyMaster_julian_tania"  # <-- CHANGE THIS

    def begin_game(self, config: GameConfig) -> None:
        """Called once at the start of each game.

        Available information in config:
          - config.word_length    (int)  — 4, 5, or 6
          - config.vocabulary     (tuple) — all valid words (secret is from here)
          - config.mode           (str)  — "uniform" or "frequency"
          - config.probabilities  (dict) — word -> probability (sums to 1)
          - config.max_guesses    (int)  — maximum guesses allowed (typically 6)
          - config.allow_non_words (bool) — True = you can guess ANY letter combo
        """
        self._vocab = list(config.vocabulary)
        self._config = config

        # Openers pre-computados: Las mejores aperturas para cada variante del juego (uniform o frecuency, longitud 4, 5 o 6)
        self._openers = {
            (4, "uniform"): "roia",
            (4, "frequency"): "cora",
            (5, "uniform"): "careo",
            (5, "frequency"): "careo",
            (6, "uniform"): "careto",
            (6, "frequency"): "cerito",
        }

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        """Return the next guess.

        Parameters
        ----------
        history : list of (guess, feedback) tuples
            Each feedback is a tuple of ints:
              2 = green (correct letter, correct position)
              1 = yellow (correct letter, wrong position)
              0 = gray (letter not in word)

        Returns
        -------
        str
            A lowercase string of length config.word_length.
            Can be ANY letter combination (not restricted to vocabulary)
            for better information discovery.
        """
        # Filter candidates consistent with all feedback so far
        candidates = self._vocab
        for past_guess, pattern in history:
            candidates = filter_candidates(candidates, past_guess, pattern)

        if not candidates:
            return self._vocab[0]

        # Caso de un solo candidato restante por si acaso
        if len(candidates) == 1:
            return candidates[0]
        
        # Si es el turno 1: usamos el opener precalculado
        if not history:
            key = (self._config.word_length, self._config.mode)
            return self._openers.get(key, self._vocab[0])
        
        # Si estamos en el último turno (6) o quedan 2 palabras o menos, disparamos a la más probable
        # Utilizamos entropía
        if len(candidates) <= 2 or len(history) >= 5:
            # Ordenamos de mayor a menor probabilidad y devolvemos la más probable
            candidates.sort(key=lambda word: self._config.probabilities.get(word, 0), reverse=True)
            return candidates[0]
        
        # Si tenemos de 3 a 15 opciones posibles, inyectamos todo el vocabulario
        # Gastamos un intento en un guess para reducir el espacio de búsqueda
        if 2 < len(candidates) <= 15:
            vocab_sample = random.sample(self._vocab, min(200, len(self._vocab)))
            words_to_test = list(set(candidates + vocab_sample))

        # Si hay más de 150 palabras, tomamos un sample para ahorrar tiempo
        elif len(candidates) > 150:
            words_to_test = random.sample(candidates, 150)
        else:
            words_to_test = candidates

        best_guess = candidates[0]
        max_score = -1.0

        # Suma para normalizar probabilidades
        total_probability = sum(self._config.probabilities.get(candidate, 1.0) for candidate in candidates)
        if total_probability == 0: total_probability = 1.0

        # Calculamos la entropía de cada posible guess sobre los candidatos restantes
        for test_word in words_to_test:
            pattern_probabilities = defaultdict(float)
            
            # Simulación contra posibles respuestas
            for c in candidates:
                simmulated_pattern = feedback(c, test_word)
                candidate_probability = self._config.probabilities.get(c, 1.0) / total_probability
                pattern_probabilities[simmulated_pattern] += candidate_probability
            
            # Ecuación de Shannon con NumPy
            probabilities_array = np.array(list(pattern_probabilities.values()))
            entropy = -np.sum(probabilities_array * np.log2(probabilities_array + 1e-9))
            
            # Si la palabra es un candidato real, le damos un bono de 0.1
            is_valid_candidate = 1.0 if test_word in candidates else 0.0
            final_score = entropy + (0.1 * is_valid_candidate)

            # Evaluamos contra el max_score en lugar de solo la entropía
            if final_score > max_score:
                max_score = final_score
                best_guess = test_word
        
        return best_guess