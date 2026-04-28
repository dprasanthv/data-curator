import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

from faker import Faker
from kafka import KafkaProducer


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


class SocialGraph:
    def __init__(self, max_users: int = 200, max_posts: int = 1000, max_comments: int = 2000) -> None:
        self.max_users = max_users
        self.max_posts = max_posts
        self.max_comments = max_comments
        self.users: list[dict] = []
        self.posts: list[dict] = []
        self.comments: list[dict] = []
        self.follow_edges: set[tuple[str, str]] = set()

    def _cap(self) -> None:
        if len(self.users) > self.max_users:
            self.users = self.users[-self.max_users :]
        if len(self.posts) > self.max_posts:
            self.posts = self.posts[-self.max_posts :]
        if len(self.comments) > self.max_comments:
            self.comments = self.comments[-self.max_comments :]


def ensure_user(fake: Faker, graph: SocialGraph) -> dict:
    if graph.users and random.random() < 0.90:
        return random.choice(graph.users)

    user = {"user_id": str(uuid.uuid4()), "name": fake.name()}
    graph.users.append(user)
    graph._cap()
    return user


def build_post(fake: Faker, graph: SocialGraph) -> dict:
    actor = ensure_user(fake, graph)
    post = {
        "post_id": str(uuid.uuid4()),
        "author_user_id": actor["user_id"],
        "text": fake.paragraph(nb_sentences=random.randint(1, 4)),
        "privacy": random.choice(["public", "friends", "only_me"]),
        "client": random.choice(["ios", "android", "web"]),
        "location": random.choice([None, fake.city(), fake.country()]),
        "media": [
            {
                "type": random.choice(["image", "video"]),
                "url": fake.image_url(),
            }
        ]
        if random.random() < 0.25
        else [],
    }
    graph.posts.append(post)
    graph._cap()
    return {
        "event_type": "post",
        "event_id": str(uuid.uuid4()),
        "created_at": utc_now_iso(),
        "actor": actor,
        "post": post,
    }


def build_comment(fake: Faker, graph: SocialGraph) -> dict:
    if not graph.posts:
        build_post(fake, graph)

    actor = ensure_user(fake, graph)
    post = random.choice(graph.posts)
    comment = {
        "comment_id": str(uuid.uuid4()),
        "author_user_id": actor["user_id"],
        "post_id": post["post_id"],
        "post_author_user_id": post["author_user_id"],
        "text": fake.sentence(nb_words=random.randint(6, 20)),
    }
    graph.comments.append(comment)
    graph._cap()
    return {
        "event_type": "comment",
        "event_id": str(uuid.uuid4()),
        "created_at": utc_now_iso(),
        "actor": actor,
        "comment": comment,
    }


def build_like(fake: Faker, graph: SocialGraph) -> dict:
    if not graph.posts:
        build_post(fake, graph)

    actor = ensure_user(fake, graph)

    if graph.comments and random.random() < 0.40:
        target_type = "comment"
        target = random.choice(graph.comments)
        target_id = target["comment_id"]
        target_author_user_id = target.get("author_user_id")
    else:
        target_type = "post"
        target = random.choice(graph.posts)
        target_id = target["post_id"]
        target_author_user_id = target["author_user_id"]

    reaction = {
        "target_type": target_type,
        "target_id": target_id,
        "target_author_user_id": target_author_user_id,
        "reaction_type": random.choice(["like", "love", "haha", "wow", "sad", "angry"]),
    }

    return {
        "event_type": "reaction",
        "event_id": str(uuid.uuid4()),
        "created_at": utc_now_iso(),
        "actor": actor,
        "reaction": reaction,
    }


def build_follow(fake: Faker, graph: SocialGraph) -> dict:
    while len(graph.users) < 2:
        ensure_user(fake, graph)

    actor = ensure_user(fake, graph)
    followed = random.choice([u for u in graph.users if u["user_id"] != actor["user_id"]])

    edge = (actor["user_id"], followed["user_id"])
    if edge in graph.follow_edges and len(graph.users) > 2:
        candidates = [
            u
            for u in graph.users
            if u["user_id"] != actor["user_id"] and (actor["user_id"], u["user_id"]) not in graph.follow_edges
        ]
        if candidates:
            followed = random.choice(candidates)
            edge = (actor["user_id"], followed["user_id"])

    graph.follow_edges.add(edge)

    return {
        "event_type": "follow",
        "event_id": str(uuid.uuid4()),
        "created_at": utc_now_iso(),
        "actor": actor,
        "follow": {
            "followed_user_id": followed["user_id"],
            "followed_user_name": followed["name"],
        },
    }


def _send_event(producer: KafkaProducer, topic: str, event: dict) -> None:
    key = event.get("event_type", "event")
    producer.send(topic, key=key, value=event)


def bulk_generate(producer: KafkaProducer, topic: str, fake: Faker) -> None:
    users_count = int(os.environ.get("BULK_USERS", "1000"))
    posts_per_user = int(os.environ.get("BULK_POSTS_PER_USER", "10"))
    comments_min = int(os.environ.get("BULK_COMMENTS_MIN", "5"))
    comments_max = int(os.environ.get("BULK_COMMENTS_MAX", "20"))
    likes_min = int(os.environ.get("BULK_LIKES_MIN", "15"))
    likes_max = int(os.environ.get("BULK_LIKES_MAX", "25"))

    flush_every = int(os.environ.get("BULK_FLUSH_EVERY", "5000"))
    rng_seed = os.environ.get("BULK_SEED")
    if rng_seed is not None:
        random.seed(int(rng_seed))
        fake.seed_instance(int(rng_seed))

    users: list[dict[str, str]] = []
    for _ in range(users_count):
        users.append({"user_id": str(uuid.uuid4()), "name": fake.name()})

    sent = 0
    created_at = utc_now_iso()
    batch_ts = utc_now().isoformat()

    for user in users:
        _send_event(
            producer,
            topic,
            {
                "event_type": "user",
                "event_id": str(uuid.uuid4()),
                "created_at": created_at,
                "user": user,
                "_batch": {"ingested_at": batch_ts},
            },
        )
        sent += 1

        for _ in range(posts_per_user):
            post_id = str(uuid.uuid4())
            post_event = {
                "event_type": "post",
                "event_id": str(uuid.uuid4()),
                "created_at": utc_now_iso(),
                "actor": user,
                "post": {
                    "post_id": post_id,
                    "author_user_id": user["user_id"],
                    "text": fake.paragraph(nb_sentences=random.randint(1, 4)),
                    "privacy": random.choice(["public", "friends", "only_me"]),
                    "client": random.choice(["ios", "android", "web"]),
                    "location": random.choice([None, fake.city(), fake.country()]),
                },
            }
            _send_event(producer, topic, post_event)
            sent += 1

            comments_n = random.randint(comments_min, comments_max)
            for _ in range(comments_n):
                commenter = random.choice(users)
                comment_event = {
                    "event_type": "comment",
                    "event_id": str(uuid.uuid4()),
                    "created_at": utc_now_iso(),
                    "actor": commenter,
                    "comment": {
                        "comment_id": str(uuid.uuid4()),
                        "author_user_id": commenter["user_id"],
                        "post_id": post_id,
                        "post_author_user_id": user["user_id"],
                        "text": fake.sentence(nb_words=random.randint(6, 20)),
                    },
                }
                _send_event(producer, topic, comment_event)
                sent += 1

                if flush_every > 0 and sent % flush_every == 0:
                    producer.flush(timeout=30)

            likes_n = random.randint(likes_min, likes_max)
            for _ in range(likes_n):
                liker = random.choice(users)
                reaction_event = {
                    "event_type": "reaction",
                    "event_id": str(uuid.uuid4()),
                    "created_at": utc_now_iso(),
                    "actor": liker,
                    "reaction": {
                        "target_type": "post",
                        "target_id": post_id,
                        "target_author_user_id": user["user_id"],
                        "reaction_type": random.choice(["like", "love", "haha", "wow", "sad", "angry"]),
                    },
                }
                _send_event(producer, topic, reaction_event)
                sent += 1

                if flush_every > 0 and sent % flush_every == 0:
                    producer.flush(timeout=30)

    producer.flush(timeout=60)


def main() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    topic = os.environ.get("KAFKA_TOPIC", "facebook.events")
    delay_ms = int(os.environ.get("PRODUCER_DELAY_MS", "250"))
    bulk_mode = os.environ.get("BULK_MODE", "false").lower() in {"1", "true", "yes"}

    fake = Faker()
    graph = SocialGraph(
        max_users=int(os.environ.get("MAX_USERS", "200")),
        max_posts=int(os.environ.get("MAX_POSTS", "1000")),
        max_comments=int(os.environ.get("MAX_COMMENTS", "2000")),
    )
    for _ in range(int(os.environ.get("INITIAL_USERS", "25"))):
        ensure_user(fake, graph)

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8"),
        acks="all",
        retries=10,
        linger_ms=50,
    )

    if bulk_mode:
        bulk_generate(producer, topic, fake)
        return

    builders = [
        (0.50, lambda: build_post(fake, graph)),
        (0.25, lambda: build_comment(fake, graph)),
        (0.20, lambda: build_like(fake, graph)),
        (0.05, lambda: build_follow(fake, graph)),
    ]

    weights, funcs = zip(*builders)

    while True:
        event = random.choices(funcs, weights=weights, k=1)[0]()
        key = event.get("event_type", "event")

        producer.send(topic, key=key, value=event)
        producer.flush(timeout=10)

        time.sleep(delay_ms / 1000.0)


if __name__ == "__main__":
    main()
