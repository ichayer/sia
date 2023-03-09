import json

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect


if __name__ == "__main__":
    with open('config.json') as file:
        config = json.load(file)

    factory = PokemonFactory("pokemon.json")
    pokemon_name = config.get('pokemon_name')
    pokemon = factory.create(pokemon_name, 100, StatusEffect.NONE, 1)

    pokeball_name = config.get('pokeball_name')

    print("No noise: ", attempt_catch(pokemon, pokeball_name))
    for _ in range(10):
        print("Noisy: ", attempt_catch(pokemon, pokeball_name, 0.15))
