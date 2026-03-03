from strategy import Strategy, GameConfig
from strategies.entropy_strat import EntropyStrategy


class EntropyNoRepeat_compus(Strategy):
    """Wrapper around EntropyStrategy that avoids repeating secret words within a tournament run."""

    def __init__(self) -> None:
        self._entropy = EntropyStrategy()
        self._guessed_secrets: set[str] = set()
        self._session_key: tuple[int, str, int] | None = None

    @property
    def name(self) -> str:
        return "EntropyNoRepeat_compus"

    def _make_session_key(self, config: GameConfig) -> tuple[int, str, int]:
        # Use a simple signature of the round: (length, mode, vocab size)
        return (config.word_length, config.mode, len(config.vocabulary))

    def begin_game(self, config: GameConfig) -> None:
        key = self._make_session_key(config)

        # Detect cambio de ronda / configuración y reiniciar la memoria si hace falta.
        if self._session_key is None or key != self._session_key:
            self._guessed_secrets = set()
        else:
            vocab_size = len(config.vocabulary)
            if vocab_size > 0 and len(self._guessed_secrets) >= 0.9 * vocab_size:
                self._guessed_secrets = set()

        self._session_key = key

        # Filtrar vocabulario excluyendo secretos ya usados en esta sesión.
        original_vocab = config.vocabulary
        filtered_vocab = tuple(
            w for w in original_vocab if w not in self._guessed_secrets
        )

        # Si el filtrado deja sin vocabulario, caer al original para no romper nada.
        if not filtered_vocab:
            filtered_vocab = original_vocab

        # Filtrar y renormalizar probabilidades.
        if filtered_vocab is original_vocab:
            filtered_probs = dict(config.probabilities)
        else:
            filtered_probs = {
                w: config.probabilities.get(w, 0.0) for w in filtered_vocab
            }
            total = sum(filtered_probs.values())
            if total > 0.0:
                for w in filtered_vocab:
                    filtered_probs[w] /= total
            else:
                # Si por alguna razón todo quedó en cero, usar uniforme sobre el vocab filtrado.
                uniform_p = 1.0 / len(filtered_vocab)
                filtered_probs = {w: uniform_p for w in filtered_vocab}

        filtered_config = GameConfig(
            word_length=config.word_length,
            vocabulary=filtered_vocab,
            mode=config.mode,
            probabilities=filtered_probs,
            max_guesses=config.max_guesses,
            allow_non_words=config.allow_non_words,
        )

        self._entropy.begin_game(filtered_config)

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        return self._entropy.guess(history)

    def end_game(self, secret: str, solved: bool, num_guesses: int) -> None:
        self._guessed_secrets.add(secret)
        # Propagar a la estrategia base por compatibilidad futura.
        end_game = getattr(self._entropy, "end_game", None)
        if callable(end_game):
            end_game(secret, solved, num_guesses)