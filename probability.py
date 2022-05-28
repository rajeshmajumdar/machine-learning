import random

def random_kid():
   return random.choice(["boy", "girl"])

both_girls = 0
older_girl = 0
either_girl = 0

for _ in range(100000):
    younger = random_kid()
    older = random_kid()
    if older == 'girl':
        older_girl += 1
    if older == 'girl' and younger == 'girl':
        both_girls += 1
    if older == 'girl' or younger == 'girl':
        either_girl += 1

print(f"P(both | older) : {both_girls/older_girl}")
print(f"P(both | either) : {both_girls/either_girl}")
