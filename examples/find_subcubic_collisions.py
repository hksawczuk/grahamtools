from grahamtools.search.collisions_subcubic import find_collision_subcubic

# Requires nauty tools in PATH: geng, shortg
res = find_collision_subcubic(
    n_min=1,
    n_max=20,
    k_cap=6,
    N_cap=200_000,
    processes=4,
    batch_size=200,
)
print(res)
