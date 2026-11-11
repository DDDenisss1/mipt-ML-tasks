import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


DATA_DIR = Path("data_train")
CATALOG_PATH = Path("botify/data/tracks.json")
OUTPUT_PATH = Path("botify/data/recommendations_ub.json")

TOP_N = 100
BATCH_SIZE = 32


def main():
    # Загружаем идентификаторы треков из каталога Botify
    with CATALOG_PATH.open(encoding="utf-8") as f:
        catalog = {
            int(json.loads(line)["track"])
            for line in f
            if line.strip()
        }

    user_stats = defaultdict(lambda: [0.0, 0])
    interactions = []
    popularity = Counter()

    for path in sorted(DATA_DIR.glob("recommender_*.json")):
        print(f"Чтение {path}")

        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                event = json.loads(line)
                user = int(event["user"])
                track = int(event["track"])
                listen_time = event.get("time")

                if track not in catalog or listen_time is None:
                    continue

                listen_time = float(listen_time)

                interactions.append((user, track, listen_time))
                user_stats[user][0] += listen_time
                user_stats[user][1] += 1
                popularity[track] += listen_time

    users = sorted(user_stats)
    tracks = sorted(catalog)

    user_to_idx = {user: i for i, user in enumerate(users)}
    track_to_idx = {track: i for i, track in enumerate(tracks)}

    print("Взаимодействий:", len(interactions))
    print("Пользователей:", len(users))
    print("Треков в каталоге:", len(tracks))

    # Вычитаем среднее время прослушивания каждого юзера
    user_means = {
        user: total / count
        for user, (total, count) in user_stats.items()
    }

    pair_values = defaultdict(lambda: [0.0, 0])
    listened = defaultdict(set)

    for user, track, listen_time in interactions:
        centered = listen_time - user_means[user]

        pair_values[(user, track)][0] += centered
        pair_values[(user, track)][1] += 1
        listened[user].add(track)

    rows = []
    cols = []
    values = []

    for (user, track), (total, count) in pair_values.items():
        value = total / count

        if value == 0:
            continue

        rows.append(user_to_idx[user])
        cols.append(track_to_idx[track])
        values.append(value)

    # Разреженная матрица юзер - трек
    X = csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (rows, cols),
        ),
        shape=(len(users), len(tracks)),
        dtype=np.float32,
    )

    X = normalize(X, norm="l2", axis=1).tocsr()

    popular_tracks = [
        track for track, _ in popularity.most_common()
    ]
    popular_tracks += sorted(catalog - set(popular_tracks))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Обработка пользователей блоками
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for start in range(0, len(users), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(users))

            similarity = (X[start:end] @ X.T).tolil()

            # Исключаем сходство пользователя с самим собой
            for local_idx in range(end - start):
                similarity[local_idx, start + local_idx] = 0.0

            similarity = similarity.tocsr()
            similarity.eliminate_zeros()

            # Оценки треков по предпочтениям похожих пользователей
            scores = (similarity @ X).tocsr()

            for local_idx, user in enumerate(users[start:end]):
                row = scores.getrow(local_idx)

                candidates = [
                    (tracks[idx], float(score))
                    for idx, score in zip(row.indices, row.data)
                    if score > 0 and tracks[idx] not in listened[user]
                ]

                candidates.sort(key=lambda x: x[1], reverse=True)

                recommendations = [
                    track for track, _ in candidates[:TOP_N]
                ]

                # Дополняем список популярными треками, если
                # рекомендаций недостаточно
                selected = set(recommendations)

                if len(recommendations) < TOP_N:
                    for track in popular_tracks:
                        if track in listened[user] or track in selected:
                            continue

                        recommendations.append(track)
                        selected.add(track)

                        if len(recommendations) == TOP_N:
                            break

                out.write(json.dumps({
                    "user": user,
                    "tracks": recommendations,
                }) + "\n")

            if end % 1000 < BATCH_SIZE or end == len(users):
                print(f"Обработано пользователей: {end}/{len(users)}")

    print(f"Готово: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
