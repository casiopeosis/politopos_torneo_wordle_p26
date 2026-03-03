import math
from collections import defaultdict
from strategy import Strategy, GameConfig
from wordle_env import feedback, filter_candidates

class MiEstrategiaPro(Strategy):
    @property
    def name(self) -> str:
        return "Strategy_ValeNuria" # Asegúrate que sea único

    def begin_game(self, config: GameConfig) -> None:
        """Inicialización con selección de Openers por Modo y Longitud."""
        self._vocab = list(config.vocabulary)
        self._config = config
        self._priors = config.probabilities
        
        # 1. Definición de estrategias de apertura
        # Modo Uniforme: Priorizamos 100% la reducción del espacio (Entropía pura)
        openers_uniform = {
            4: "rosa", 
            5: "noria", 
            6: "cernia"
        }
        
        # Modo Frequency: Priorizamos palabras con alto Prior P(w) y alta Entropía (H)
        openers_freq = {
            4: "sera", 
            5: "corea", 
            6: "contra"
        }

        # 2. Selección lógica según el contexto del torneo
        if config.mode == "frequency":
            # Elegimos el opener de frecuencia para la longitud actual
            self.first_guess = openers_freq.get(config.word_length, self._vocab[0])
        else:
            # Elegimos el opener uniforme para la longitud actual
            self.first_guess = openers_uniform.get(config.word_length, self._vocab[0])

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        # 1. Filtrado de consistencia
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        # 2. Caso base: Solo queda una posibilidad
        if len(candidates) == 1:
            return candidates[0]

        # 3. TURNO 1: Usar palabra precomputada (Ahorra CPU)
        if not history:
            return self.first_guess

        # 4. TURNOS SIGUIENTES: Entropía + Modificación C
        return self.entropia_pesada(candidates)

    def entropia_pesada(self, candidates):
        """
        Calcula Entropía Pesada con ajuste dinámico de recursos por longitud 
        y ponderación de probabilidad según el modo de juego.
        """
        # 1. CONFIGURACIÓN DINÁMICA POR LONGITUD
        # Optimizamos pool y sample para cumplir con los 5s
        length = self._config.word_length
        num_c = len(candidates)

        if length == 4:
            p_limit, e_limit = 500, 600  # Universo pequeño: Precisión máxima
        elif length == 5:
            p_limit, e_limit = 300, 500  # Estándar Wordle
        else: # length == 6
            p_limit, e_limit = 200, 400  # Universo grande: Cuidado con el tiempo

        # Aplicamos los límites de forma segura (min entre el límite y lo que hay)
        pool_size = min(num_c, p_limit)
        eval_size = min(num_c, e_limit)
        
        guess_pool = candidates[:pool_size]
        eval_sample = candidates[:eval_size]

        # 2. PREPARACIÓN DE PESOS
        total_p_candidates = sum(self._priors.get(c, 0) for c in candidates)
        if total_p_candidates == 0: total_p_candidates = 1 

        best_word = candidates[0]
        best_score = -1.0
        
        # 3. DETERMINACIÓN DEL PESO DE PROBABILIDAD (Modificación C)
        # En 'frequency' subimos el peso a 0.5 para priorizar el gane rápido.
        # En 'uniform' lo dejamos en 0.1 para que sea solo un desempate técnico.
        prob_weight = 0.5 if self._config.mode == "frequency" else 0.1

        for g in guess_pool:
            weighted_counts = defaultdict(float)
            
            for target in eval_sample:
                pat = feedback(target, g)
                # Sumamos la masa de probabilidad (Entropía Pesada)
                weighted_counts[pat] += self._priors.get(target, 0)
            
            # Cálculo de Entropía de Shannon
            ent = 0.0
            for weight in weighted_counts.values():
                p = weight / total_p_candidates
                if p > 0:
                    ent -= p * math.log2(p)
            
            # 4. CÁLCULO DEL SCORE FINAL (H + P)
            # individual_p es la probabilidad de que 'g' sea la respuesta secreta.
            individual_p = self._priors.get(g, 0) / total_p_candidates
            score = ent + (individual_p * prob_weight)

            if score > best_score:
                best_score = score
                best_word = g
        
        return best_word