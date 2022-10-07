coin.level = 75
coin2 = coin.data[0:45000]
import copy

coins = copy.deepcopy(coin)
coins.data[15000:15000 + len(coin2)] = coin2
coins.write(data_dir / 'sounds' / 'coins.wav')