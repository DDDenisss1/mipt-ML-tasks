import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


DATA_DIR = Path("data_train")
CATALOG_PATH = Path("botify/data/tracks.json")
OUTPUT_PATH = Path("botify/data/recommendations_als.json")

FACTORS = 64
REGULARIZATION = 0.1
ITERATIONS = 30
ALPHA = 40.0

TOP_N = 100
BATCH_SIZE = 128
MIN_TIME = 0.2
SEED = 42


def load_data():
    # Загружаем каталог треков
    with CATALOG_PATH.open(encoding="utf-8") as f:
        tracks = sorted({
            int(json.loads(line)["track"])
            for line in f
            if line.strip()
        })

    track_set = set(tracks)
    interactions = []
    popularity = Counter()

    # Объединяем обучающие логи из двух контейнеров Botify
    for path in sorted(DATA_DIR.glob("recommender_*.json")):
        print(f"Чтение: {path}")

        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                event = json.loads(line)

                user = int(event["user"])
                track = int(event["track"])
                listen_time = event.get("time")

                if track not in track_set or listen_time is None:
                    continue

                listen_time = float(listen_time)

                interactions.append((user, track, listen_time))

                if listen_time > MIN_TIME:
                    popularity[track] += listen_time

    return tracks, interactions, popularity


def main():
    tracks, interactions, popularity = load_data()

    users = sorted({
        user for user, _, _ in interactions
    })

    # Преобразование исходных ID в индексы матрицы user-item
    user_to_idx = {
        user: idx for idx, user in enumerate(users)
    }

    track_to_idx = {
        track: idx for idx, track in enumerate(tracks)
    }

    print("Взаимодействий:", len(interactions))
    print("Пользователей:", len(users))
    print("Треков:", len(tracks))

    pair_weights = defaultdict(float)
    listened = defaultdict(set)

    # Суммируем положительный сигнал для каждой пары юзер-трек
    # Прослушивания короче MIN_TIME не учитываю
    for user, track, listen_time in interactions:
        listened[user].add(track)

        weight = max(0.0, listen_time - MIN_TIME)

        if weight > 0:
            pair_weights[(user, track)] += weight

    rows = []
    cols = []
    values = []

    # Преобразуем накопленный сигнал в confidence для implicit ALS
    for (user, track), weight in pair_weights.items():
        rows.append(user_to_idx[user])
        cols.append(track_to_idx[track])
        values.append(1.0 + ALPHA * weight)

    # храним только наблюдаемые взаимодействия
    user_items = csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (rows, cols),
        ),
        shape=(len(users), len(tracks)),
        dtype=np.float32,
    )

    print("Положительных пар:", user_items.nnz)

    print("Обучение ALS...")

    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        random_state=SEED,
        num_threads=4,
    )

    model.fit(user_items, show_progress=True)

    print("ALS обучена")

    user_factors = np.asarray(
        model.user_factors, dtype=np.float32
    )
    item_factors = np.asarray(
        model.item_factors, dtype=np.float32
    )

    # Список популярных треков используем как fallback
    popular_tracks = [
        track for track, _ in popularity.most_common()
    ]

    popular_set = set(popular_tracks)

    popular_tracks.extend(
        track for track in tracks
        if track not in popular_set
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    print("Формирование рекомендаций...")

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for start in range(0, len(users), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(users))

            # Скаляр скрытых факторов даёт оценки
            # предпочтения юзера для каждого трека
            scores = user_factors[start:end] @ item_factors.T

            for local_idx, user in enumerate(users[start:end]):
                user_scores = scores[local_idx]

                # Исключаю треки из обучающей истории пользователя
                for track in listened[user]:
                    user_scores[track_to_idx[track]] = -np.inf

                top_indices = np.argpartition(
                    user_scores, -TOP_N
                )[-TOP_N:]

                top_indices = top_indices[
                    np.argsort(user_scores[top_indices])[::-1]
                ]

                recommendations = [
                    tracks[idx]
                    for idx in top_indices
                    if np.isfinite(user_scores[idx])
                ]

                # Если рекомендаций недостаточно, дополняем их
                # популярными треками, исключая уже прослушанные
                if len(recommendations) < TOP_N:
                    selected = set(recommendations)

                    for track in popular_tracks:
                        if track in selected or track in listened[user]:
                            continue

                        recommendations.append(track)
                        selected.add(track)

                        if len(recommendations) == TOP_N:
                            break

                out.write(json.dumps({
                    "user": user,
                    "tracks": recommendations,
                }) + "\n")

            print(
                f"Обработано пользователей: {end}/{len(users)}"
            )

    print("Готово:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
